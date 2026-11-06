from chromorph import FSPSStellarSED, Scene, SEDContext
from chromorph import Parameter
import numpy as np
from conftest import observations, clump


def test_constant_sfh_uses_only_constant_component_and_independent_ages():
    one = FSPSStellarSED(sfh='constant', nebular=True)
    two = FSPSStellarSED(sfh='constant', nebular=True)
    assert one.configuration['sfh'] == 1
    assert one.configuration['const'] == 1.
    assert one.configuration['fburst'] == 0.
    assert 'tau' not in one.parameters
    scene = Scene(observations(), context=SEDContext(distance_mpc=10))
    scene.add_component(clump('one', sed=one))
    scene.add_component(clump('two', x=17, sed=two))
    assert 'one.sed.age' in scene.registry.names()
    assert 'two.sed.age' in scene.registry.names()
    assert not scene.registry.ties
    one.configuration['const'] = .5
    assert two.configuration['const'] == 1.


def test_default_sfh_and_nebular_ionization_are_explicit():
    default = FSPSStellarSED(nebular=True)
    assert default.sfh == 'constant'
    assert not default.parameters['gas_logu'].vary
    assert default.configuration['add_neb_emission']
    assert default.configuration['add_neb_continuum']
    freed = FSPSStellarSED(nebular=True, parameters={'gas_logu': Parameter(-2., -4., -1., vary=True)})
    assert freed.parameters['gas_logu'].vary
    assert FSPSStellarSED(sfh='ssp').configuration['sfh'] == 0
    exponential = FSPSStellarSED(sfh='exponential')
    assert exponential.configuration['sfh'] == 1 and exponential.configuration['const'] == 0.
    assert 'tau' in exponential.parameters


def test_component_sfh_and_filter_predictions_have_independent_caches():
    class Probe(FSPSStellarSED):
        calls = 0
        def _spectrum(self, parameters):
            self.calls += 1
            return np.array([3000., 5000., 9000.]), np.full(3,
                parameters['tage']*(1+self.configuration.get('const', 0.)))
    class Band:
        fingerprint = 'test-band'
        calls = 0
        def microjy(self, wave, flux):
            self.calls += 1
            return float(flux.sum())
    obs = observations();band = Band()
    for o in obs:o.throughput = band
    one, two = Probe(sfh='constant'), Probe(sfh='ssp')
    scene = Scene(obs, context=SEDContext(distance_mpc=10))
    scene.add_component(clump('one', sed=one));scene.add_component(clump('two', x=17, sed=two))
    values = scene.registry.unpack()
    first = one.fluxes(values, 'one', obs, scene.context)
    other = two.fluxes(values, 'two', obs, scene.context)
    count = band.calls
    mass = dict(values);mass['one.sed.log_mass'] += 1.
    scaled = one.fluxes(mass, 'one', obs, scene.context)
    np.testing.assert_allclose(scaled['blue'], 10*first['blue'])
    assert band.calls == count and one.calls == two.calls == 1
    changed = dict(values);changed['one.sed.age'] = .1
    assert one.fluxes(changed, 'one', obs, scene.context) != first
    assert two.fluxes(changed, 'two', obs, scene.context) == other
    assert two.calls == 1
