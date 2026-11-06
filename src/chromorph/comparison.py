"""Compare SFH-dependent morphology, formed mass and component photometry."""
from pathlib import Path
import html
import json
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from .result import _json
from .simulation import match_components


def compare_sfh_results(directories, output):
    paths = {name: Path(path) for name, path in directories.items()}
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    results = {name: json.loads((path/'result.json').read_text()) for name, path in paths.items()}
    baseline = next((name for name, result in results.items()
        if all(c['sed']['model'] == 'free' for c in result['component_definitions'].values())), next(iter(results)))
    reference = results[baseline]
    with fits.open(paths[baseline]/'images.fits', checksum=True) as first:
        for name, path in paths.items():
            with fits.open(path/'images.fits', checksum=True) as other:
                assert all(h.verify_checksum() == 1 for h in other)
                for hdu in first[1:]:
                    if hdu.name.endswith(('_DATA', '_SIGMA', '_MASK', '_PSF')):
                        np.testing.assert_array_equal(hdu.data, other[hdu.name].data)
                for observation in results[name]['observations']:
                    summed = sum(other[f'{observation}_COMP_{component}'].data
                                 for component in results[name]['component_definitions'])
                    summed += results[name]['values'][f'observation.{observation}.background']
                    np.testing.assert_allclose(summed, other[observation+'_MODEL'].data, rtol=1e-10, atol=1e-10)
    morph, populations, photometry, matching = [], [], [], {}
    ref_centers = {n: c['reference_xy'] for n, c in reference['centroids'].items()
                   if reference['component_definitions'][n]['spatial'] == 'Sersic'}
    ref_phot = Table.read(paths[baseline]/'photometry.ecsv', format='ascii.ecsv')
    ref_flux = {(str(r['component']), str(r['band'])): float(r['total_microjy']) for r in ref_phot}
    ref_frame = {(str(r['component']), str(r['band'])): float(r['frame_rate']) for r in ref_phot}
    for model, result in results.items():
        values = result['values']
        centers = {n: c['reference_xy'] for n, c in result['centroids'].items()
                   if result['component_definitions'][n]['spatial'] == 'Sersic'}
        matching[model] = match_components(ref_centers, centers)
        for component, definition in result['component_definitions'].items():
            row = dict(model=model, component=component)
            for key in ['x', 'y', 're', 'n', 'q', 'pa']:
                parameter = f'{component}.morphology.{key}'
                if parameter not in values:continue
                value = values[parameter]
                delta = value-reference['values'][parameter]
                if key == 'pa':delta = (delta+90)%180-90
                row[key] = value;row['delta_'+key+'_from_free'] = delta
            morph.append(row)
            if definition['sed']['model'] == 'fsps':
                sfh = definition['sed']['sfh']
                populations.append(dict(model=model, component=component, sfh=sfh,
                    time_interpretation='SF_duration_Gyr' if sfh == 'constant' else 'SSP_age_Gyr' if sfh == 'ssp' else 'time_since_SF_onset_Gyr',
                    time_Gyr=values[f'{component}.sed.age'], log_formed_mass=values[f'{component}.sed.log_mass'],
                    constant_SFR_msun_per_year=(10**values[f'{component}.sed.log_mass']
                        /(values[f'{component}.sed.age']*1e9) if sfh == 'constant' else None),
                    metallicity=values[f'{component}.sed.metallicity'], av=values[f'{component}.sed.av'],
                    gas_logu=values.get(f'{component}.sed.gas_logu'),
                    gas_logu_free=result['parameter_definitions'].get(f'{component}.sed.gas_logu', {}).get('vary', False)))
        table = Table.read(paths[model]/'photometry.ecsv', format='ascii.ecsv')
        for row in table:
            component, band = str(row['component']), str(row['band'])
            flux = float(row['total_microjy']);base = ref_flux[(component, band)]
            photometry.append(dict(model=model, component=component, band=band, observation=str(row['observation']), microjy=flux,
                frame_rate=float(row['frame_rate']),delta_frame_rate_from_free=float(row['frame_rate'])-ref_frame[(component,band)],
                delta_microjy_from_free=flux-base,
                fractional_delta_from_free=None if base <= 0 else flux/base-1))
    # Distinct time definitions are reported separately; do not subtract SSP
    # ages from constant-SFR durations as if they were one measured quantity.
    mass_systematics = []
    constant_models = [name for name, result in results.items()
        if any(c['sed'].get('sfh') == 'constant' for c in result['component_definitions'].values())]
    ssp_models = [name for name, result in results.items()
        if any(c['sed'].get('sfh') == 'ssp' for c in result['component_definitions'].values())]
    for constant_name in constant_models:
        for ssp_name in ssp_models:
            constant, ssp = results[constant_name], results[ssp_name]
            for component, definition in constant['component_definitions'].items():
                if definition['sed'].get('sfh') != 'constant' or ssp['component_definitions'][component]['sed'].get('sfh') != 'ssp':continue
                key = component+'.sed.log_mass'
                delta = constant['values'][key]-ssp['values'][key]
                mass_systematics.append(dict(component=component, constant_model=constant_name, ssp_model=ssp_name,
                    delta_log_formed_mass_constant_minus_ssp=delta, formed_mass_ratio_constant_to_ssp=10**delta))
    morphology_sensitivity = []
    for row in morph:
        if row['model'] == baseline or 're' not in row:continue
        base_re = reference['values'][row['component']+'.morphology.re']
        morphology_sensitivity.append(dict(model=row['model'], component=row['component'],
            centroid_shift_pixels=float(np.hypot(row['delta_x_from_free'], row['delta_y_from_free'])),
            fractional_radius_change=row['re']/base_re-1,
            delta_sersic_n=row['delta_n_from_free'], delta_axis_ratio=row['delta_q_from_free']))
    for filename, rows in [('morphology_comparison.ecsv', morph), ('population_comparison.ecsv', populations),
                           ('photometry_comparison.ecsv', photometry), ('mass_sfh_systematics.ecsv', mass_systematics)]:
        if rows:Table(rows=rows).write(output/filename, format='ascii.ecsv', overwrite=True)
    summary = dict(models={name:{k:r[k] for k in ['success', 'chi2', 'bic_like', 'n_parameters', 'identifiability', 'warnings']}
                           for name, r in results.items()}, identical_inputs_verified=True,
        centroid_label_matching=matching, mass_systematics=mass_systematics,
        morphology_sensitivity=morphology_sensitivity,
        caveats=['BIC-like ignores drizzle covariance and is not Bayesian evidence.',
                 'SFH-dependent best-fit shifts are systematic sensitivities, not significance estimates.',
                 'Limited filter coverage can leave age, metallicity, dust and formed mass weakly constrained.',
                 'Total photometry includes profile extrapolation; compare frame-integrated rates as well.',
                 'Centroid matching is a label-switch diagnostic, not proof of physical component identity.'])
    (output/'sfh_comparison.json').write_text(json.dumps(_json(summary), indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    components = list(reference['component_definitions'])
    bands = sorted({row['band'] for row in photometry})
    fig, axes = plt.subplots(len(components), 1, figsize=(9, 2.5*len(components)), squeeze=False, layout='constrained')
    for component, ax in zip(components, axes[:, 0]):
        for index, model in enumerate(results):
            lookup = {r['band']: r['microjy'] for r in photometry if r['model'] == model and r['component'] == component}
            ax.plot(np.arange(len(bands))+(index-1)*.08, [lookup[b] for b in bands],
                    linestyle='none', marker=['o', 's', '^', 'D'][index % 4], label=model)
        ax.set(title=component, xticks=range(len(bands)), xticklabels=bands, ylabel='Flux (µJy)')
        ax.legend(fontsize=8)
    fig.savefig(output/'component_photometry_sfh.pdf');fig.savefig(output/'component_photometry_sfh.png', dpi=160);plt.close(fig)
    body = '<h1>Component SED and morphology comparison</h1><p>Identical data, masks, uncertainty maps and empirical PSFs verified. Nuclear band fluxes remain independent in every model.</p>'
    body += '<table><tr><th>Model</th><th>Converged</th><th>Image χ²</th><th>Parameters</th><th>BIC-like</th><th>Local rank</th></tr>'
    for name, result in results.items():
        relative = os.path.relpath(paths[name].resolve(), output.resolve())
        body += f'<tr><td><a href="{html.escape(str(relative))}/index.html">{html.escape(name)}</a></td><td>{result["success"]}</td><td>{result["chi2"]:.4f}</td><td>{result["n_parameters"]}</td><td>{result["bic_like"]:.4f}</td><td>{result["identifiability"]["data_rank"]}</td></tr>'
    body += '</table><ul>'+''.join('<li>'+html.escape(c)+'</li>' for c in summary['caveats'])+'</ul>'
    body += '<h2>Morphology sensitivity relative to free-band</h2><p>Offsets and radii are on the reference image pixel grid. These are best-fit shifts, not significance estimates.</p>'
    body += '<table><tr><th>Model</th><th>Component</th><th>Centroid shift (pixel)</th><th>Radius change</th><th>Δn</th><th>Δq</th></tr>'
    for row in morphology_sensitivity:
        body += (f'<tr><td>{html.escape(row["model"])}</td><td>{html.escape(row["component"])}</td>'
                 f'<td>{row["centroid_shift_pixels"]:.3f}</td><td>{100*row["fractional_radius_change"]:+.1f}%</td>'
                 f'<td>{row["delta_sersic_n"]:+.3f}</td><td>{row["delta_axis_ratio"]:+.3f}</td></tr>')
    body += '</table>'
    if mass_systematics:
        body += '<h2>Formed stellar mass sensitivity</h2><p>Constant-SFR minus SSP; these are formed masses, before stellar mass loss. Free-band fitting alone does not provide a stellar mass.</p>'
        body += '<table><tr><th>Component</th><th>Δlog₁₀ formed mass (dex)</th><th>Constant-SFR / SSP mass</th></tr>'
        for row in mass_systematics:
            body += (f'<tr><td>{html.escape(row["component"])}</td>'
                     f'<td>{row["delta_log_formed_mass_constant_minus_ssp"]:+.3f}</td>'
                     f'<td>{row["formed_mass_ratio_constant_to_ssp"]:.3f}</td></tr>')
        body += '</table>'
    body += '<h2>Fit limitations</h2><ul>'
    for name, result in results.items():
        for warning in result['warnings']:
            body += '<li>'+html.escape(name+': '+warning)+'</li>'
    body += '</ul><h2>Component photometry</h2>'
    body += '<img src="component_photometry_sfh.png" style="max-width:100%">'
    body += '<p>'+ ' · '.join(f'<a href="{name}">{name}</a>' for name in ['sfh_comparison.json', 'morphology_comparison.ecsv', 'population_comparison.ecsv', 'mass_sfh_systematics.ecsv', 'photometry_comparison.ecsv'] if (output/name).exists())+'</p>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:auto;padding:24px}td,th{border:1px solid #ccd;padding:8px}table{border-collapse:collapse}</style>'+body)
    return summary
