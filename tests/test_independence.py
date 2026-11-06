from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from chromorph import Scene, FreeBandSED, FSPSStellarSED, SEDContext, PointSource, Parameter
from chromorph.parameters import Registry
from conftest import observations, clump


class ToyPopulation:
    """Analytic testing SED, NEVER represented as FSPS or a science model."""
    def __init__(self, age=1., dust=.2, mass=1.):
        self.parameters = dict(age=Parameter(age, .05, 4), dust=Parameter(dust, 0, 2),
                               mass=Parameter(mass, .01, 20))
    def register(self, registry, name, bands):
        for key, p in self.parameters.items():
            registry.add(name+'.sed.'+key, p)
    def fluxes(self, values, name, observations, context):
        age, dust, mass = [values[name+'.sed.'+key] for key in ['age', 'dust', 'mass']]
        # Distinct non-collinear spectral derivatives across >=4 bands.
        spectrum = mass*np.array([np.exp(-age-2*dust), np.exp(-.3*age-dust),
                                  (1+.3*age)*np.exp(-.3*dust), (1+age)*np.exp(-.1*dust)])
        return {o.band: spectrum[i] for i, o in enumerate(observations)}


def test_independent_population_changes_one_component_only(renderer):
    scene = Scene(observations(('b', 'v', 'r', 'i')), renderer=renderer)
    scene.add_component(clump('one', sed=ToyPopulation(.2, .1, 2)))
    scene.add_component(clump('two', sed=ToyPopulation(2, .8, 3), x=17))
    v = scene.registry.unpack()
    _, before, flux_before = scene.predict(v)
    for parameter, replacement in [('age', 1.), ('dust', .5), ('mass', 5.)]:
        change = dict(v); change['one.sed.'+parameter] = replacement
        _, after, flux_after = scene.predict(change)
        assert flux_before['one'] != flux_after['one']
        assert flux_before['two'] == flux_after['two']
        for o in scene.observations:
            np.testing.assert_array_equal(before[o.name]['two'], after[o.name]['two'])


def test_separate_fsps_configuration_and_cache():
    a = FSPSStellarSED(sfh='young', nebular=True, dust_type=2)
    b = FSPSStellarSED(sfh='old', nebular=False, dust_type=0)
    assert a.parameters['age'].value != b.parameters['age'].value
    a.configuration['dust_type'] = 4
    a._cache['probe'] = (1, 2)
    assert b.configuration['dust_type'] == 0 and not b._cache
    a.close(); b.close()


def test_fsps_complete_parameter_mapping_no_cross_contamination():
    class SpectrumProbe(FSPSStellarSED):
        def _spectrum(self, parameters):
            self.last = parameters.copy()
            wave = np.array([3500., 5000., 8000.])
            return wave, np.full(3, np.exp(parameters['logzsol']-parameters['dust2'])*parameters['tage'])
    scene = Scene(observations(), context=SEDContext(distance_mpc=10))
    a, b = SpectrumProbe(sfh='young'), SpectrumProbe(sfh='old')
    scene.add_component(clump('a', sed=a)); scene.add_component(clump('b', sed=b))
    v = scene.registry.unpack()
    wa, fa = a.spectrum(v, 'a', scene.context)
    wb, fb = b.spectrum(v, 'b', scene.context)
    changed = dict(v); changed['b.sed.metallicity'] = -.9
    b.spectrum(changed, 'b', scene.context)
    _, fa2 = a.spectrum(changed, 'a', scene.context)
    np.testing.assert_array_equal(fa, fa2)
    assert b.last['logzsol'] == -.9 and a.last['logzsol'] == -.3


def test_nuclear_fluxes_are_unconstrained_and_not_tied(renderer):
    scene = Scene(observations(), renderer=renderer)
    scene.add_component(PointSource('nucleus', FreeBandSED({'blue': 10, 'red': .01}), x=12, y=12))
    scene.add_component(clump('clump', {'blue': .01, 'red': 10}))
    with pytest.raises(ValueError, match='Nuclear fluxes'):
        scene.tie('nucleus.flux.blue', 'clump.flux.blue')
    with pytest.raises(ValueError, match='free per-band'):
        Scene(observations()).add_component(PointSource('nucleus', FSPSStellarSED()))
    p = Parameter(1., 0, 10, prior=(1, .1))
    proper=Scene(observations())
    proper.add_component(PointSource('nucleus', FreeBandSED(parameters={'blue': p})))
    assert proper.registry.parameters['nucleus.flux.blue'].prior==(1.,.1)
    assert proper.registry.parameters['nucleus.flux.red'].prior is None


def test_sed_instance_reuse_is_rejected(renderer):
    scene = Scene(observations(), renderer=renderer); sed = FreeBandSED()
    scene.add_component(clump('one', sed=sed))
    with pytest.raises(ValueError, match='distinct SED'):
        scene.add_component(clump('two', sed=sed))


def test_explicit_centroid_groups_and_cycles(renderer):
    scene = Scene(observations(), renderer=renderer)
    scene.add_component(clump('one', moving=True))
    scene.add_component(clump('two', moving=True))
    scene.tie_centroids(['one', 'two'])
    assert 'two.morphology.x' not in scene.registry.names()
    v = scene.registry.unpack(); assert v['one.morphology.x'] == v['two.morphology.x']
    with pytest.raises(ValueError, match='Cyclic'):
        scene.tie('one.morphology.x', 'two.morphology.x')


def test_registry_collision_and_invalid_parameter():
    r = Registry(); r.add('a', Parameter(1, 0, 2))
    with pytest.raises(ValueError): r.add('a', Parameter(1))
    with pytest.raises(ValueError): Parameter(2, 0, 1)


def test_tie_to_fixed_root_respects_alias_bounds():
    registry = Registry()
    registry.add('fixed', Parameter(1., vary=False))
    registry.add('alias', Parameter(2.5, 2., 3.))
    with pytest.raises(ValueError, match='bounds'):
        registry.tie('alias', 'fixed')
    assert not registry.ties


def test_mixed_free_and_population_seds_share_one_likelihood(renderer):
    obs = observations(('b', 'v', 'r', 'i'))
    scene = Scene(obs, renderer=renderer)
    scene.add_component(clump('stellar', sed=ToyPopulation(.2, .1, 2)))
    scene.add_component(clump('free', {'b': 1, 'v': 3, 'r': .1, 'i': 2}, x=17))
    values = scene.registry.unpack()
    model, images, fluxes = scene.predict(values)
    for o in obs:
        np.testing.assert_allclose(model[o.name], images[o.name]['stellar']+images[o.name]['free'])
    assert 'stellar.sed.age' in values and 'free.flux.b' in values
    assert 'free.sed.age' not in values


def test_fsps_worker_configuration_change_restarts_state(monkeypatch):
    import chromorph.sed as module
    class Executor:
        created = []
        def __init__(self, **kwargs):
            self.configuration = kwargs['initargs'][0]
            self.closed = False
            Executor.created.append(self)
        def submit(self, function, params):
            from concurrent.futures import Future
            future = Future()
            future.set_result((np.array([4000., 5000.]),
                               np.full(2, self.configuration['dust_type']+1.), 1.))
            return future
        def shutdown(self, wait):
            self.closed = True
    monkeypatch.setattr(module, 'ProcessPoolExecutor', Executor)
    sed = FSPSStellarSED()
    params = dict(tage=1., logzsol=0., dust2=.2)
    first = sed._spectrum(params)[1]
    sed.configuration['dust_type'] = 4
    second = sed._spectrum(params)[1]
    assert Executor.created[0].closed and len(Executor.created) == 2
    assert not np.array_equal(first, second)
    sed.close()


def test_stellar_flux_cache_includes_mass_context_and_throughput():
    class Probe(FSPSStellarSED):
        calls = 0
        def _spectrum(self, parameters):
            self.calls += 1
            return np.array([4000., 5000.]), np.full(2, parameters['tage'])
    class Band:
        def __init__(self, scale):
            self.scale = scale; self.fingerprint = str(scale)
        def microjy(self, wavelength, flam):
            return float(flam.sum()*self.scale)
    obs = observations(); obs[0].throughput = Band(1.); obs[1].throughput = Band(2.)
    sed = Probe(); scene = Scene(obs, context=SEDContext(distance_mpc=10))
    scene.add_component(clump('a', sed=sed)); values = scene.registry.unpack()
    first = sed.fluxes(values, 'a', obs, scene.context)
    sed.fluxes(values, 'a', obs, scene.context)
    assert sed.calls == 1
    changed = dict(values); changed['a.sed.log_mass'] += 1
    scaled = sed.fluxes(changed, 'a', obs, scene.context)
    np.testing.assert_allclose(scaled['blue'], 10*first['blue'])
    farther = sed.fluxes(values, 'a', obs, SEDContext(distance_mpc=20))
    np.testing.assert_allclose(farther['blue'], first['blue']/4)
    obs[0].throughput = Band(3.)
    different_filter = sed.fluxes(values, 'a', obs, scene.context)
    np.testing.assert_allclose(different_filter['blue'], 3*first['blue'])


def test_spatial_cache_reuses_only_unchanged_component_morphology(renderer):
    scene = Scene(observations(), renderer=renderer)
    scene.add_component(clump('one', {'blue': 2, 'red': 1}, moving=True))
    values = scene.registry.unpack()
    initial = scene.predict(values)[1]
    changed_sed = dict(values); changed_sed['one.flux.blue'] = 4
    sed_images = scene.predict(changed_sed)[1]
    np.testing.assert_allclose(sed_images['blue']['one'], 2*initial['blue']['one'])
    shifted = dict(values); shifted['one.morphology.x'] += .5
    shifted_images = scene.predict(shifted)[1]
    assert not np.array_equal(initial['blue']['one'], shifted_images['blue']['one'])
    # Returning to the old vector must recover the old basis exactly.
    np.testing.assert_array_equal(scene.predict(values)[1]['blue']['one'], initial['blue']['one'])
