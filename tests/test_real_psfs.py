"""Injection/recovery with copied, existing J1042 empirical_v2 kernels.

No science pixels are bundled. Synthetic images use measured per-band noise
levels; no claim is made that diagonal noise reproduces drizzle covariance.
"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
pytest.importorskip('galsim')
from chromorph import Scene, Observation, EmpiricalPSF, FreeBandSED, PointSource, Parameter, Sersic
from chromorph.rendering import GalSimRenderer
from chromorph.simulation import simulate

import os
DATA = Path(os.environ.get('CHROMORPH_J1042_FIXTURES', Path(__file__).parent/'data'))
if not (DATA/'j1042_empirical_v2_psfs.npz').exists():
    pytest.skip('Local J1042 PSF fixtures are not bundled for publication', allow_module_level=True)


def actual_psf_observations():
    info = json.loads((DATA/'j1042_empirical_v2_metadata.json').read_text())
    result = []
    with np.load(DATA/'j1042_empirical_v2_psfs.npz') as arrays:
        for band, m in info.items():
            result.append(Observation(band, band, np.zeros((31, 31)), np.full((31, 31), m['sigma_rate']),
                                      EmpiricalPSF(arrays[band], m['pixel_scale'], tuple(m['center']),
                                                   provenance=m), m['pixel_scale'], m['ujy_per_rate'],
                                      pivot_angstrom=m['pivot_angstrom']))
    return result


def make_scene(obs, case, *, injection):
    scene = Scene(obs, fit_registration=case == 'astrometry')
    nuclear = dict(zip(scene.bands, [1., .2, 3., .1]))
    scene.add_component(PointSource('nucleus', FreeBandSED(nuclear if injection else None),
                                     parameters={'x': Parameter(15.2, 14, 16.5, vary=False),
                                                 'y': Parameter(15.3, 14, 16.5, vary=False)}))
    if case == 'point_only':
        return scene
    centers = [(11., 18.)]
    if case in ['several', 'overlap', 'centroid_offsets', 'astrometry', 'psf_mismatch', 'sed_mismatch']:
        centers += [(12.5, 17.) if case == 'overlap' else (21., 13.)]
    for i, (x, y) in enumerate(centers):
        flux = dict(zip(scene.bands, [2., 1.5, .4, .25] if i == 0 else [.2, .35, 2., 3.]))
        if case == 'sed_mismatch' and injection and i == 0:
            flux[scene.bands[1]] = 6.  # strong feature beyond a constant-color family
        pars = {'x': Parameter(x, x-1, x+1, vary=case == 'centroid_offsets'),
                'y': Parameter(y, y-1, y+1, vary=case == 'centroid_offsets'),
                're': Parameter(.25 if case == 'compact' else 1.5, .1, 4, vary=False),
                'n': Parameter(.8, .3, 6.2, vary=False), 'q': Parameter(.7, .1, 1, vary=False),
                'pa': Parameter(25., -180, 180, vary=False)}
        scene.add_component(Sersic('clump'+str(i+1), FreeBandSED(flux if injection else None), parameters=pars))
    return scene


@pytest.mark.parametrize('case', ['point_only', 'one_sersic', 'several', 'overlap', 'compact', 'centroid_offsets', 'astrometry'])
def test_actual_psf_injection_recovery(case):
    obs = actual_psf_observations()
    with make_scene(obs, case, injection=True) as truth:
        values = truth.registry.unpack()
        if case == 'astrometry':
            for o in obs[1:]:
                values[f'observation.{o.name}.dx'] = .2
                values[f'observation.{o.name}.dy'] = -.15
        data = simulate(truth, values, seed=1042)
        with make_scene(data, case, injection=False) as scene:
            result = scene.fit(multistart=1, max_nfev=80)
            assert min(result.runs, key=lambda r: r['objective'])['success']
            assert .7 < result.chi2/result.n_pixels < 1.3
            for component in truth.components:
                for band in scene.bands:
                    expected = truth.predict(values)[2][component.name][band]
                    assert abs(result.fluxes[component.name][band]-expected) < .2


def test_actual_psf_sampling_for_every_filter():
    renderer = GalSimRenderer()
    for obs in actual_psf_observations():
        qa = renderer.validate_psf_sampling(obs)
        assert qa['pass_same_grid'], (obs.band, qa)


def test_psf_mismatch_is_detected_in_residuals():
    obs = actual_psf_observations()
    with make_scene(obs, 'point_only', injection=True) as truth:
        data = simulate(truth, seed=1042)
        # Controlled test mismatch, not a reconstructed empirical science PSF.
        from scipy.ndimage import gaussian_filter
        wrong = [replace(o, psf=EmpiricalPSF(gaussian_filter(o.psf.image, 1.2), o.pixel_scale, o.psf.center)) for o in data]
        with make_scene(wrong, 'point_only', injection=False) as scene:
            result = scene.fit(multistart=1)
            assert result.chi2/result.n_pixels > 1.5
