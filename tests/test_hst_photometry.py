"""Live STSynphot calibration smoke test, independent of private target data."""
import os
import numpy as np
import pytest
if os.environ.get('CHROMORPH_HST_TESTS') != '1':
    pytest.skip('Set CHROMORPH_HST_TESTS=1 with HST reference tables', allow_module_level=True)
pytest.importorskip('stsynphot')
from chromorph.photometry import HSTThroughput


@pytest.mark.parametrize('band', ['F275W', 'F336W', 'F621M', 'F689M'])
def test_flat_microjy_spectrum_through_real_hst_bandpass(band):
    throughput = HSTThroughput('wfc3,uvis2,'+band.lower())
    wave = np.geomspace(500., 26000., 10000)
    flam = 1e-29*2.99792458e18/wave**2
    assert throughput.pivot_angstrom > 2000
    assert throughput.fingerprint
    np.testing.assert_allclose(throughput.microjy(wave, flam), 1., rtol=2e-4)
