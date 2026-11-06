"""Evaluate a second FSPS library at fixed best-fit parameters, without refitting."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from astropy.table import Table
from .sed import FreeBandSED


def export_postfit_spectra(scene, values, output, reference_fluxes, *,
                           expected_library='c3k_hr', workers=4):
    """Run in a process whose FSPS build uses the requested display library.

    Spatial models, amplitudes and SED parameters remain fixed. The returned
    spectra are not asserted to be exact high-resolution versions of a different
    stellar library. Integrate both through the same bandpasses and report the
    discrepancy. In-memory FSPS caches are isolated per component/process;
    disk cache keys include the compiled backend and full component context.
    """
    import fsps
    probe = fsps.StellarPopulation()
    libraries = [v.decode() if isinstance(v, bytes) else str(v) for v in probe.libraries]
    if libraries[1] != expected_library:
        raise ValueError('Post-fit backend is '+libraries[1]+', expected '+expected_library)
    extension = next(Path(fsps.__file__).parent.glob('_fsps*.so'))
    backend = dict(version=fsps.__version__, libraries=libraries,
                   extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                   native_samples=len(probe.wavelengths))
    resolution = []
    for lo, hi in [(1500., 3000.), (3000., 7000.), (7000., 16000.)]:
        mask = (probe.wavelengths >= lo) & (probe.wavelengths <= hi)
        sigma = probe.resolutions[mask]
        resolution.append(dict(rest_range_angstrom=[lo, hi],
            median_sigma_kms=float(np.median(sigma)),
            approximate_FWHM_resolving_power=float(299792.458/(2.354820045*np.median(abs(sigma)))),
            approximate=bool(np.any(sigma < 0))))
    backend['resolution'] = resolution
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'library_resolution.npz',
                        rest_wave=probe.wavelengths, sigma_kms=probe.resolutions)
    frozen = dict(values)
    stellar = [c for c in scene.components if not isinstance(c.sed, FreeBandSED)]

    def evaluate(component):
        name = component.name
        state = dict(backend=backend, context=asdict(scene.context), sed=component.sed.describe(),
                     parameters={k: v for k, v in frozen.items() if k.startswith(name+'.sed.')},
                     throughputs=[getattr(o.throughput, 'fingerprint', None) for o in scene.observations])
        key = json.dumps(state, sort_keys=True)
        path = output/(name+'.npz')
        cached = False
        if path.exists():
            with np.load(path) as previous:
                if str(previous['key']) == key:
                    wave, flam = previous['wave'].copy(), previous['flam'].copy()
                    cached = True
        if not cached:
            wave, flam = component.sed.spectrum(frozen, name, scene.context)
            np.savez_compressed(path, wave=wave, flam=flam, key=key)
        rows = []
        for observation in scene.observations:
            high = observation.throughput.microjy(wave, flam)
            low = reference_fluxes[name][observation.band]
            rows.append(dict(component=name, band=observation.band,
                low_resolution_fit_microjy=low, high_resolution_postfit_microjy=high,
                fractional_difference=high/low-1 if low > 0 else np.nan))
        table = Table({'observed_wavelength_angstrom': wave,
                       'flam_erg_s_cm2_angstrom': flam})
        table.meta.update(CMPT=name, SPECLIB=libraries[1],
                          REFITTED=False, REDSHIFT=scene.context.redshift)
        table.write(output/(name+'.fits'), overwrite=True)
        return name, (wave, flam), rows

    with ThreadPoolExecutor(max_workers=min(max(int(workers), 1), max(len(stellar), 1))) as pool:
        evaluated = list(pool.map(evaluate, stellar))
    rows = [row for name, spectrum, comparisons in evaluated for row in comparisons]
    Table(rows=rows).write(output/'band_flux_comparison.ecsv', format='ascii.ecsv', overwrite=True)
    finite = [abs(r['fractional_difference']) for r in rows if np.isfinite(r['fractional_difference'])]
    report = dict(backend=backend, parameters_refitted=False, likelihood_backend_changed=False,
        maximum_absolute_band_flux_fractional_difference=max(finite, default=0.),
        interpretation='Post-fit spectral evaluation at fixed low-resolution best-fit parameters; not a high-resolution fit.',
        component_states={c.name: {k: v for k, v in frozen.items() if k.startswith(c.name+'.sed.')} for c in stellar})
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    if values != frozen:raise RuntimeError('Post-fit spectral evaluation changed fit parameters')
    return {name: spectrum for name, spectrum, comparisons in evaluated}, report
