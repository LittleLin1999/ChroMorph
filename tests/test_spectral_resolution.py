import numpy as np
from chromorph.spectral_resolution import desi_resolution_sigma, resample_flux, subtract_matched_extended


def test_desi_diagonal_order_and_measured_second_moment():
    wave = np.arange(101, dtype=float)+5000
    offset = np.arange(5, -6, -1)
    weights = np.exp(-.5*(offset/2)**2);weights /= weights.sum()
    diagonal = np.repeat(weights[:, None], len(wave), axis=1)
    sigma = desi_resolution_sigma(wave, diagonal)
    np.testing.assert_allclose(sigma[10:-10], np.sqrt(weights@offset**2), rtol=1e-13)
    np.testing.assert_allclose(resample_flux(wave, np.full_like(wave, 2.), wave[2:-2]), 2.)


def test_resolution_subtraction_does_not_double_broaden_template():
    wave = np.arange(4970., 5030., .02)
    model = np.exp(-.5*((wave-5000)/1.)**2)/np.sqrt(2*np.pi)
    total = np.exp(-.5*((wave-5000)/2.)**2)/(2*np.sqrt(2*np.pi))
    result = subtract_matched_extended(wave,total,np.ones(len(wave),bool),np.ones(len(wave)),
        wave,model,np.full(len(wave),2.),299792.458/wave)
    good = result['good'] & (abs(wave-5000)<10)
    np.testing.assert_allclose(result['extended'][good], total[good], atol=1e-5)
    assert result['fraction_data_broadened'] == 0.
