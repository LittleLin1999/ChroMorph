"""Target-independent loading of calibrated images and explicit PSFs."""
from pathlib import Path
import numpy as np
import yaml
from astropy.io import fits
from .observations import Observation, EmpiricalPSF, sha256


def load_observations(manifest, *, throughputs=True):
    """Read a YAML/JSON manifest; mask values of True include pixels.

    File paths are relative to the manifest, never to the working directory.
    Calibrations, uncertainty maps, PSF sampling and affine transforms are
    supplied explicitly. No filter, target, instrument or pipeline is inferred.
    """
    path = Path(manifest).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get('format') != 'chromorph-observations-v1':
        raise ValueError('Expected format: chromorph-observations-v1')
    rows = config.get('observations', [])
    if not rows:
        raise ValueError('Manifest contains no observations')
    hashes = {}
    def array(spec):
        spec = {'path':spec} if isinstance(spec, str) else dict(spec)
        filename = (path.parent/spec['path']).resolve()
        if filename not in hashes:hashes[filename] = sha256(filename)
        if spec.get('sha256') and hashes[filename] != spec['sha256']:
            raise ValueError('Input provenance hash mismatch: '+str(filename))
        with fits.open(filename) as hd:
            hdu = hd[spec.get('ext', 0)]
            data, header = np.array(hdu.data, copy=True), hdu.header.copy()
        return data, header, {'path':str(filename), 'ext':spec.get('ext', 0), 'sha256':hashes[filename]}
    result, diagnostics = [], []
    names = set()
    for row in rows:
        name = row['name']
        if name in names:raise ValueError('Duplicate observation name: '+name)
        names.add(name)
        data, header, source = array(row['data'])
        sigma, _, noise = array(row['sigma'])
        mask = None if row.get('mask') is None else array(row['mask'])[0].astype(bool)
        psf_spec = row['psf']
        kernel, _, kernel_source = array(psf_spec)
        psf = EmpiricalPSF(kernel, float(psf_spec['pixel_scale']),
            None if 'center' not in psf_spec else tuple(psf_spec['center']),
            bool(psf_spec.get('pixel_integrated', True)), provenance=kernel_source)
        throughput = None
        if throughputs and row.get('throughput') is not None:
            spec = row['throughput']
            if isinstance(spec, str) or 'obsmode' in spec:
                from .photometry import HSTThroughput
                throughput = HSTThroughput(spec if isinstance(spec, str) else spec['obsmode'])
            else:
                from .photometry import TabulatedThroughput
                throughput = TabulatedThroughput(spec['wave_angstrom'], spec['response'])
        observation = Observation(name, row['band'], data, sigma, psf,
            float(row['pixel_scale']), float(row['ujy_per_rate']), mask=mask,
            matrix=row.get('matrix', np.eye(2)), offset=row.get('offset', [0.,0.]),
            background=float(row.get('background', 0.)), header=header,
            pivot_angstrom=row.get('pivot_angstrom'), throughput=throughput,
            provenance=dict(source=source, noise=noise, **row.get('provenance', {})))
        result.append(observation)
        diagnostics.append(dict(name=name, band=observation.band, shape=list(data.shape),
            n_fit_pixels=int(observation.mask.sum()), psf=psf.qa(), source=source, noise=noise))
    report = dict(target=config.get('target'), manifest=str(path), observations=diagnostics,
        blockers=[], warnings=['Fixed calibrated uncertainty maps are used; drizzle covariance is not inferred.'])
    return result, report, dict(config.get('context', {}))
