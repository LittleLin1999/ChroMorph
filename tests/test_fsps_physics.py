"""Live stellar physics tests with synthetic filters, no private target data.

Requires public FSPS reference libraries. These are distinct from the opt-in
HST/STSynphot calibration-data tests.
"""
import os
import numpy as np
import pytest
if os.environ.get('CHROMORPH_FSPS_TESTS') != '1':
    pytest.skip('Set CHROMORPH_FSPS_TESTS=1 with public FSPS libraries', allow_module_level=True)
pytest.importorskip('fsps'); pytest.importorskip('synphot'); pytest.importorskip('galsim')
from astropy import units as u
from synphot import SpectralElement
from synphot.models import Box1D
from chromorph import Scene, FSPSStellarSED, FreeBandSED, PointSource, Sersic, SEDContext, Parameter
from chromorph.photometry import HSTThroughput
from chromorph.simulation import simulate
from conftest import observations


def stellar_observations():
    obs = observations(('u', 'b', 'v', 'r', 'j', 'h'), size=31, noise=.0001)
    for o, wave in zip(obs, [3500., 4400., 5500., 7000., 11000., 16000.]):
        # Exercise the production synphot integration on explicitly synthetic
        # test filters; these are not claimed to be calibrated HST responses.
        throughput = HSTThroughput.__new__(HSTThroughput)
        throughput.bandpass = SpectralElement(Box1D, amplitude=1., x_0=wave*u.AA, width=wave*.15*u.AA)
        throughput.obsmode = 'synthetic-box'
        throughput.fingerprint = str(wave)
        o.throughput = throughput
        o.pivot_angstrom = wave
    return obs


def population(age, av, mass, metallicity):
    return FSPSStellarSED(parameters={
        'age': Parameter(age, .05, 5), 'av': Parameter(av, 0., 2.),
        'log_mass': Parameter(mass, 5., 11.),
        'metallicity': Parameter(metallicity, -2., .3, vary=False)})


def component(name, sed, x):
    return Sersic(name, sed, parameters={
        'x': Parameter(x, vary=False), 'y': Parameter(15., vary=False),
        're': Parameter(1.5, .1, 5, vary=False), 'n': Parameter(.8, .3, 6.2, vary=False),
        'q': Parameter(.8, .1, 1, vary=False), 'pa': Parameter(20., -180, 180, vary=False)})


def test_live_process_isolation_mass_redshift_and_distinct_populations():
    obs = stellar_observations()
    with Scene(obs, context=SEDContext(redshift=.2)) as scene:
        a, b = population(.2, .1, 8., -.5), population(3., .7, 8.5, -.1)
        scene.add_component(component('a', a, 9.)); scene.add_component(component('b', b, 21.))
        values = scene.registry.unpack()
        fa = a.fluxes(values, 'a', obs, scene.context); fb = b.fluxes(values, 'b', obs, scene.context)
        assert fa != fb
        changed = dict(values); changed['a.sed.av'] = 1.
        assert a.fluxes(changed, 'a', obs, scene.context) != fa
        assert b.fluxes(changed, 'b', obs, scene.context) == fb
        changed['b.sed.metallicity'] = -.8
        a_again = a.fluxes(changed, 'a', obs, scene.context)
        assert b.fluxes(changed, 'b', obs, scene.context) != fb
        assert a.fluxes(changed, 'a', obs, scene.context) == a_again
        mass = dict(values); mass['a.sed.log_mass'] += 1
        np.testing.assert_allclose(list(a.fluxes(mass, 'a', obs, scene.context).values()),
                                   10*np.array(list(fa.values())), rtol=1e-8)
        assert a._executor is not b._executor


@pytest.mark.parametrize('separation', [12., 4.])
def test_live_joint_age_dust_mass_recovery(separation):
    obs = stellar_observations(); context = SEDContext(redshift=.2)
    with Scene(obs, context=context) as truth:
        truth.add_component(component('young', population(.2, .1, 8., -.5), 9.))
        truth.add_component(component('old', population(3., .7, 8.5, -.1), 9.+separation))
        data = simulate(truth, seed=1042)
        with Scene(data, context=context) as scene:
            scene.add_component(component('young', population(.25, .15, 8.05, -.5), 9.))
            scene.add_component(component('old', population(2.8, .6, 8.45, -.1), 9.+separation))
            result = scene.fit(multistart=1, max_nfev=100)
            assert min(result.runs, key=lambda r: r['objective'])['success']
            # High-S/N, known morphology and fixed independent metallicities
            # deliberately make this a constrained recovery case.
            for name, target in [('young', [.2, .1, 8.]), ('old', [3., .7, 8.5])]:
                actual = [result.values[name+'.sed.'+k] for k in ['age', 'av', 'log_mass']]
                np.testing.assert_allclose(actual, target, rtol=.03, atol=.03)
