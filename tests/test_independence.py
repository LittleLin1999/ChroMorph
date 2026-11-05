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
    with pytest.raises(ValueError, match='Nuclear fluxes'):
        Scene(observations()).add_component(PointSource('nucleus', FreeBandSED(parameters={'blue': p})))


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
