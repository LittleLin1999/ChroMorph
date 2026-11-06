"""The cached integration must reproduce synphot, including line spectra."""
import numpy as np
import pytest

synphot = pytest.importorskip('synphot')
from astropy import units as u
from synphot import SpectralElement
from synphot.models import Empirical1D
from chromorph.photometry import HSTThroughput


def throughput():
    obj = object.__new__(HSTThroughput)
    obj.bandpass = SpectralElement(Empirical1D, points=np.array([1000, 2000, 2300, 4200, 5000])*u.AA,
                                  lookup_table=np.array([0., .1, 1., .7, 0.]))
    obj._quadrature = None
    return obj


@pytest.mark.parametrize('spectrum', ['flat_fnu', 'sloped', 'lines'])
def test_operator_matches_reference(spectrum):
    band = throughput()
    for count in [201, 1007]:
        wave = np.geomspace(500., 10000., count)
        flux = 1e-16*(wave/3000.)**(-2 if spectrum == 'flat_fnu' else 1.3)
        if spectrum == 'lines':
            flux[70] += 1e-11
            flux[100] += 1e-12
        expected = band.microjy_reference(wave, flux)
        np.testing.assert_allclose(band.microjy(wave, flux), expected, rtol=2e-13)
        np.testing.assert_allclose(band.microjy(wave, flux*3), expected*3, rtol=2e-13)


def test_operator_preserves_coverage_check():
    from synphot.exceptions import PartialOverlap
    with pytest.raises(PartialOverlap):
        throughput().microjy(np.linspace(2000, 4000, 100), np.ones(100))
