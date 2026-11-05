"""Discover J1042 data and retain reduction provenance without re-drizzling."""
from pathlib import Path
import csv
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from .observations import Observation, EmpiricalPSF, sha256
from .sed import SEDContext

TARGET = 'J104242.43+372147.63'


def _redshift(root, metadata=None):
    candidates = [Path(metadata)] if metadata else [p/'lrd2-data/targets/targets.csv' for p in root.parents]
    for path in candidates:
        if path.exists():
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    if row.get('full_name') == TARGET:
                        return float(row['z']), dict(path=str(path), sha256=sha256(path))
    return None, None


def discover_j1042(root, *, metadata=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HST root not accessible: {root}")
    analysis = root/'analysis'
    pointer = analysis/TARGET/'fit/empirical_v2/hstpipe/current_fit.json'
    if not pointer.exists():
        raise FileNotFoundError("Selected fit pointer is required to recover validated masks/noise and initial geometry")
    selected = json.loads(pointer.read_text())
    fitpath = (analysis/selected['fit']).resolve()
    if analysis.resolve() not in fitpath.parents:
        raise ValueError("Selected fit escapes analysis root")
    saved = json.loads(fitpath.read_text())
    if saved.get('psf_model') != 'drizzled empirical_v2':
        raise ValueError("Selected fit does not use drizzled empirical_v2 PSFs")
    psfroot = root/'products/uvis/psfs'
    manifest = json.loads((psfroot/'manifest.json').read_text())
    records = {r['filter']: r for r in manifest['records'] if r['target'] == TARGET and r['product'] == 'hstpipe'}
    report = dict(target=TARGET, root=str(root), selected_fit=str(fitpath),
                  selected_json_sha256=sha256(fitpath), selected_fits_sha256=sha256(fitpath.with_suffix('.fits')),
                  filters=[], blockers=[], warnings=[
                      'Science replay has per-band sky offsets; saved sky levels are retained.',
                      'Saved diagonal noise maps omit drizzle pixel covariance.',
                      'Finite PSF stamps cannot establish the unobserved far-wing energy.',
                      'GalSim same-grid/subpixel tests and injection recovery must pass before science interpretation.'])
    observations = []
    with fits.open(fitpath.with_suffix('.fits'), checksum=True) as hd:
        if not all(h.verify_checksum() == 1 and h.verify_datasum() == 1 for h in hd):
            raise ValueError("Selected FITS checksum mismatch")
        reference = WCS(hd[saved['reference_filter']+'_DATA'].header).celestial
        for band in saved['filters']:
            meta = saved['bands'][band]
            record = records[band]
            sci = root/'products/uvis'/record['reference']
            ps = next(p for p in record['outputs'] if p['model'] == 'empirical_v2')
            psfpath = psfroot/ps['bundle_path']
            if sha256(sci) != meta['science_sha256'] or sha256(psfpath) != meta['psf_sha256'] or sha256(psfpath) != ps['sha256']:
                raise ValueError(f"Input provenance hash mismatch: {band}")
            with fits.open(psfpath) as ph:
                header = ph[0].header
                if header.get('PSFMODEL') != 'empirical_v2':
                    raise ValueError("Wrong PSF product")
                psf = EmpiricalPSF(ph[0].data, record['pixel_scale_arcsec'],
                                   (header['PSFCENX'], header['PSFCENY']),
                                   provenance=dict(path=str(psfpath), sha256=sha256(psfpath),
                                                   kernel=record['kernel'], pixfrac=record['pixfrac'],
                                                   drizzle_validation=record['validation']))
            h = hd[band+'_DATA'].header.copy()
            wcs = WCS(h).celestial
            scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix)))*3600)
            pivot = float(h['PHOTPLAM']); flam = float(h['PHOTFLAM'])
            matrix = np.array(meta['reference_to_band_matrix'])
            offset = np.array(meta['reference_to_band_offset'])
            shifts = saved['fitted_guess']['registration'][band]
            shift = np.array([shifts['dx']['value'], shifts['dy']['value']])
            n = saved['image_size_pixels']
            xy = np.array([[0, 0], [n-1, 0], [0, n-1], [n-1, n-1], [(n-1)/2, (n-1)/2]])
            sky = reference.pixel_to_world(xy[:, 0], xy[:, 1])
            bx, by = wcs.world_to_pixel(sky)
            error = np.array([bx, by]).T-(xy@matrix.T+offset)
            affine_max = float(np.max(np.linalg.norm(error, axis=1)))
            qa = psf.qa()
            if abs(qa['sum']-1) > .01:
                report['blockers'].append(band+': PSF normalization differs from unity by >1%')
            if affine_max > .1:
                report['blockers'].append(band+': WCS affine residual exceeds 0.1 pixel')
            if qa['outer_two_pixel_positive_fraction'] > .01:
                report['warnings'].append(band+': more than 1% positive PSF energy in outer border')
            if qa['centroid_offset_pixels'] > .5:
                report['warnings'].append(band+': flux centroid differs from declared PSF center by >0.5 pixel')
            if record['validation']['context_equal_fraction'] < .999 or record['validation']['weight_relative_max'] > 1e-4:
                report['blockers'].append(band+': drizzle replay geometry/weight QA failed')
            with fits.open(sci) as raw:
                available = [x.name for x in raw]
                unit = str(raw['SCI'].header.get('BUNIT', '')).upper()
                if unit not in ['ELECTRONS/S', 'ELECTRON/S']:
                    report['blockers'].append(band+': science units are not electron rate')
            observations.append(Observation(
                band, band, hd[band+'_DATA'].data, hd[band+'_SIGMA'].data,
                psf, scale, flam*pivot**2/2.99792458e18*1e29,
                mask=hd[band+'_FITMASK'].data.astype(bool), matrix=matrix, offset=offset+shift,
                background=float(meta['sky']), header=h, pivot_angstrom=pivot,
                provenance=dict(science=str(sci), science_sha256=sha256(sci),
                                noise_source=str(fitpath.with_suffix('.fits')), noise_ext=band+'_SIGMA',
                                mask_ext=band+'_FITMASK', fitted_registration=shift.tolist())))
            report['filters'].append(dict(band=band, shape=list(observations[-1].data.shape),
                                          n_fit_pixels=int(observations[-1].mask.sum()), raw_extensions=available,
                                          pixel_scale_arcsec=scale, pivot_angstrom=pivot, photflam=flam,
                                          psf=qa, affine_max_error_pixels=affine_max))
    redshift, source = _redshift(root, metadata)
    report.update(redshift=redshift, redshift_source=source,
                  initial_guess=saved.get('fitted_guess'))
    if redshift is None:
        report['warnings'].append('No redshift metadata found; physical SED fits require explicit context')
    return observations, report


def attach_hst_throughputs(observations, obsmodes=None):
    from .photometry import HSTThroughput
    for o in observations:
        o.throughput = HSTThroughput((obsmodes or {}).get(o.band, 'wfc3,uvis2,'+o.band.lower()))
    return observations
