"""Independent validation against the documented native constant-SFR model."""
import os
import numpy as np
import pytest
if os.environ.get('CHROMORPH_FSPS_TESTS') != '1':
    pytest.skip('Set CHROMORPH_FSPS_TESTS=1 with public FSPS libraries', allow_module_level=True)
fsps = pytest.importorskip('fsps')
from astropy.constants import L_sun
from chromorph import FSPSStellarSED, SEDContext, Scene, Parameter
from conftest import observations, clump


def test_constant_duration_spectrum_and_native_sfr_normalization():
    context = SEDContext(distance_mpc=10)
    sed = FSPSStellarSED(nebular=True, parameters={'age': Parameter(.1, .001, 6.)})
    with Scene(observations(), context=context) as scene:
        scene.add_component(clump('clump', sed=sed))
        values = scene.registry.unpack()
        wave, per_mass = sed.spectrum_per_mass(values, 'clump', context)
        direct = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
            sfh=1, const=1., fburst=0., sf_start=0., sf_trunc=0., dust_type=2,
            add_neb_emission=True, add_neb_continuum=True, nebemlineinspec=True,
            logzsol=-.3, dust2=.2/1.0857362047581296, dust1=0., gas_logz=-.3, gas_logu=-2.)
        native_wave, native_luminosity = direct.get_spectrum(tage=.1, peraa=True)
        expected = native_luminosity/direct.formed_mass*L_sun.cgs.value/(4*np.pi*context.distance_cm()**2)
        np.testing.assert_array_equal(wave, native_wave)
        np.testing.assert_allclose(per_mass, expected, rtol=1e-12, atol=0.)
        np.testing.assert_allclose(direct.sfr*.1*1e9/direct.formed_mass, 1., rtol=.005)
