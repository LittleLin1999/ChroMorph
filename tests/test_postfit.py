"""Fixed-parameter post-fit export; synthetic backend used only in these tests."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from chromorph import Scene
from chromorph.postfit import export_postfit_spectra
from conftest import observations, clump
from test_sfh_comparison import ReportPopulation


def backend(tmp_path, monkeypatch, library='c3k_hr'):
    (tmp_path/'_fsps_test.so').write_bytes(b'synthetic backend identity')
    wave = np.linspace(1000., 20000., 100)
    probe = SimpleNamespace(libraries=[b'mist', library.encode(), b'DL07'],
                            wavelengths=wave, resolutions=np.full_like(wave, 100.))
    monkeypatch.setitem(sys.modules, 'fsps', SimpleNamespace(__file__=str(tmp_path/'__init__.py'),
                        __version__='synthetic', StellarPopulation=lambda: probe))


def test_postfit_preserves_parameters_and_caches_by_backend(tmp_path, monkeypatch, renderer):
    backend(tmp_path, monkeypatch)
    obs = observations()
    for o, high in zip(obs, [2.1, 3.2]):
        o.throughput = SimpleNamespace(fingerprint=o.band, microjy=lambda w, f, value=high: value)
    scene = Scene(obs, renderer=renderer)
    sed = ReportPopulation(sfh='constant')
    calls = []
    native = sed.spectrum
    monkeypatch.setattr(sed, 'spectrum', lambda *args: (calls.append(1) or native(*args)))
    scene.add_component(clump('one', sed=sed))
    values = scene.registry.unpack();unchanged = dict(values)
    reference = {'one': {'blue': 2., 'red': 3.}}
    spectra, report = export_postfit_spectra(scene, values, tmp_path/'output', reference)
    assert values == unchanged and not report['parameters_refitted']
    assert not report['likelihood_backend_changed']
    assert len(calls) == 1
    np.testing.assert_allclose(report['maximum_absolute_band_flux_fractional_difference'], 3.2/3-1)
    export_postfit_spectra(scene, values, tmp_path/'output', reference)
    assert len(calls) == 1
    assert (tmp_path/'output/one.fits').exists()


def test_postfit_rejects_wrong_compiled_library(tmp_path, monkeypatch, renderer):
    backend(tmp_path, monkeypatch, 'c3k_lr')
    with pytest.raises(ValueError, match='expected c3k_hr'):
        export_postfit_spectra(Scene(observations(), renderer=renderer), {}, tmp_path/'output', {})
