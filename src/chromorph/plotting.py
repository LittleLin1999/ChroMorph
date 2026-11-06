"""Component SED figures with calibrated photometry and aligned RGB diagnostics.

Context spectra/photometry are display-only and never enter the likelihood.
The plotted band points are total model fluxes, not monochromatic spectral
samples. Free-band points, including every nuclear band, are never connected.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from .sed import FreeBandSED
from .spatial import PointSource

C_ANGSTROM_S = 2.99792458e18
COMPONENT_STYLES = [('#d18024', 'v'), ('#3179a1', '^'),
                    ('#7956a4', 'P'), ('#af548a', 'X'), ('#009e73', 'h')]
BAND_COLORS = ['#425cba', '#2b83ba', '#4d9a38', '#c43c39']


def microjy_to_scaled_flam(flux, wavelength_angstrom):
    """Pivot-equivalent f_lambda in 1e-17 erg/s/cm2/Angstrom."""
    return np.asarray(flux)*1e-29*C_ANGSTROM_S/np.asarray(wavelength_angstrom)**2*1e17


def component_rgbs(result, *, crop=None):
    """Align to the reference grid; use one host/data stretch without smoothing.

    crop is (xmin, xmax, ymin, ymax) in reference pixels. Photometric inputs
    and saved images are never modified. RGB intensities use surface brightness
    so observations with different pixel areas can be displayed consistently.
    """
    scene, values = result.scene, result.values
    observations = sorted(scene.observations, key=lambda o: o.pivot_angstrom or 0)
    if any(o.pivot_angstrom is None for o in observations):return None
    # One exposure per filter for display; repeated exposures remain in the fit.
    observations = list({o.band: o for o in observations}.values())
    reference = scene.observations[0]
    yy, xx = np.indices(reference.data.shape, dtype=float)
    points = [c.name for c in scene.components if isinstance(c, PointSource)]
    extended = [c.name for c in scene.components if c.name not in points]
    if not points or not extended:return None
    aligned = {key: [] for key in ['total', 'extended', 'point', 'data_minus_psf']+['component:'+name for name in extended]}
    noise, validity = [], []
    for o in observations:
        bx = o.matrix[0, 0]*xx+o.matrix[0, 1]*yy+o.offset[0]+values[f'observation.{o.name}.dx']
        by = o.matrix[1, 0]*xx+o.matrix[1, 1]*yy+o.offset[1]+values[f'observation.{o.name}.dy']
        point = sum(result.images[o.name][name] for name in points)
        host = sum(result.images[o.name][name] for name in extended)
        data_minus = o.data-values[f'observation.{o.name}.background']-point
        np.testing.assert_allclose(data_minus, host+o.data-result.models[o.name], rtol=1e-9, atol=1e-10)
        factor = o.ujy_per_rate/o.pixel_scale**2
        planes = dict(total=host+point, extended=host, point=point, data_minus_psf=data_minus)
        planes.update({'component:'+name: result.images[o.name][name] for name in extended})
        for key, image in planes.items():
            aligned[key].append(factor*map_coordinates(image, [by, bx], order=1, mode='constant', cval=0.))
        noise.append(factor*float(np.median(o.sigma[o.mask])))
        validity.append(map_coordinates(o.mask.astype(float), [by, bx], order=1, mode='constant', cval=0.) > .999)
    if crop is None:
        centers = np.array([[values[f'{c.name}.morphology.{axis}'] for axis in ['x', 'y']]
                            for c in scene.components])
        margin = max(8, min(reference.data.shape)*.12)
        lower = np.maximum(0, np.floor(centers.min(axis=0)-margin)).astype(int)
        upper = np.minimum(reference.data.shape[::-1], np.ceil(centers.max(axis=0)+margin)).astype(int)
        crop = (lower[0], upper[0], lower[1], upper[1])
    x0, x1, y0, y1 = map(int, crop)
    if not (0 <= x0 < x1 <= xx.shape[1] and 0 <= y0 < y1 <= xx.shape[0]):
        raise ValueError('RGB crop lies outside the reference image')
    groups = np.array_split(np.arange(len(observations)), 3) if len(observations) >= 3 else None
    def channels(arrays):
        if groups is None:
            return np.stack([arrays[-1], np.mean(arrays, axis=0), arrays[0]], axis=-1)
        return np.stack([np.mean([arrays[i] for i in group], axis=0) for group in groups[::-1]], axis=-1)
    rgbs = {key: channels(arrays)[y0:y1, x0:x1] for key, arrays in aligned.items()}
    if groups is None:
        channel_noise = [noise[-1], np.linalg.norm(noise)/len(noise), noise[0]]
    else:
        channel_noise = [np.linalg.norm([noise[i] for i in group])/len(group) for group in groups]
    intensity_noise = np.linalg.norm(channel_noise)/3
    black = 1.5*intensity_noise
    host_white = max(float(np.percentile(rgbs['extended'].mean(axis=-1), 99.5)), black*1.01, 1e-30)
    common_valid = np.logical_and.reduce(validity)[y0:y1, x0:x1]
    for key, rgb in rgbs.items():
        intensity = rgb.mean(axis=-1)
        if key not in {'extended', 'data_minus_psf'}:
            white = max(float(np.percentile(intensity, 99.5)), 1e-30)
            stretch = 6. if key == 'point' else 1.
            display_intensity = np.arcsinh(stretch*np.maximum(intensity, 0)/white)/np.arcsinh(stretch)
        else:
            display_intensity = np.arcsinh(np.maximum(intensity-black, 0)/(host_white-black))/np.arcsinh(1.)
        gain = np.divide(display_intensity, intensity, out=np.zeros_like(intensity), where=intensity > 0)
        rgbs[key] = np.clip(np.maximum(rgb, 0)*gain[..., None], 0, 1)
        rgbs[key][~common_valid] = 0.
    bar_pixels, bar_label = None, None
    if scene.context.redshift > 0:
        from astropy.cosmology import Planck18
        from astropy import units as u
        kpc_arcsec = Planck18.kpc_proper_per_arcmin(scene.context.redshift).to_value(u.kpc/u.arcsec)
        bar_pixels, bar_label = 1/(kpc_arcsec*reference.pixel_scale), '1 kpc'
    else:
        bar_pixels, bar_label = .2/reference.pixel_scale, '0.2 arcsec'
    return dict(images=rgbs, crop=[int(v) for v in crop], bands=[o.band for o in observations],
                bar_pixels=bar_pixels, bar_label=bar_label,
                host_white=host_white, host_black=black, smoothing=False,
                note='Extended/data-minus-PSF share a stretch; other model panels are individually scaled. Masked pixels are black.')


def plot_component_sed(result, directory, *, title='ChroMorph component SEDs',
                       context_spectra=(), context_photometry=None, xlim=None,
                       ylim=None, rgb_crop=None, prefix='sed', measured_fluxes=None,
                       measured_errors=None, photometry_source=None,
                       separate_components=False, spectrum_provenance=None,
                       nuclear_spectra=()):
    """Plot full-range stellar curves and measured versus synthetic band fluxes.

    measured_fluxes/errors are nested component -> band -> microJy dictionaries,
    normally supplied by an independent free-band decomposition of the same data.
    Aggregate errors may be supplied under __total__ and __stellar__; these
    should include cross-component covariance. No physical nuclear curve is drawn.
    Context spectra/photometry are display-only. Filled points are photometry;
    white points with colored edges are fitted-model band predictions.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
    from matplotlib import patheffects as pe
    from matplotlib.lines import Line2D
    from astropy.table import Table
    scene = result.scene
    observed = sorted({o.band: o for o in scene.observations}.values(),
                      key=lambda o: o.pivot_angstrom or scene.bands.index(o.band)+1)
    calibrated = all(o.pivot_angstrom is not None for o in observed)
    pivots = np.array([o.pivot_angstrom or (scene.bands.index(o.band)+1)*1e4 for o in observed])
    waves = pivots/1e4
    limits = xlim or (waves.min()*.82, waves.max()*1.18)
    convert = (lambda f: microjy_to_scaled_flam(f, pivots)) if calibrated else np.asarray
    physical = any(not isinstance(c.sed, FreeBandSED) for c in scene.components)
    measured = measured_fluxes if measured_fluxes is not None else {
        c.name: result.fluxes[c.name] for c in scene.components if isinstance(c.sed, FreeBandSED)}
    errors = measured_errors or {}
    styles = {'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 15,
              'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.minor.visible': True,
              'ytick.minor.visible': True, 'xtick.top': True, 'ytick.right': True,
              'ytick.left': True, 'pdf.fonttype': 42, 'ps.fonttype': 42}
    records = []
    with plt.rc_context(styles):
        rgb = component_rgbs(result, crop=rgb_crop)
        has_point = any(isinstance(c, PointSource) for c in scene.components)
        nuclear_detail = separate_components and has_point
        fig, ax = plt.subplots(figsize=(10.4, 10.5 if nuclear_detail else 9.5 if separate_components else 6.5), layout='constrained')
        for spectrum in context_spectra:
            ax.plot(spectrum['wave_micron'], spectrum['scaled_flam'], lw=.7,
                    color=spectrum.get('color', '#3e6ca8'), alpha=.85,
                    label=spectrum.get('label'), zorder=1, rasterized=True)
        if context_photometry is not None:
            p = context_photometry
            ax.errorbar(p['wave_micron'], p['scaled_flam'], yerr=p['error'], fmt='s', ms=5.5,
                        mfc='#56687a', mec='#56687a', color='#56687a', lw=.7, capsize=2,
                        label=p.get('label', 'Ground photometry'), zorder=3)
        point_names = [c.name for c in scene.components if isinstance(c, PointSource)]
        extended = [c for c in scene.components if c.name not in point_names]

        def draw_points(key, names, color, marker, label, size=7.5):
            model = np.array([sum(result.fluxes[n][o.band] for n in names) for o in observed])
            actual = (np.array([sum(measured[n][o.band] for n in names) for o in observed])
                      if all(n in measured for n in names) else None)
            error_key = names[0] if len(names) == 1 else key
            error = np.array([errors.get(error_key, {}).get(o.band, np.nan) for o in observed], dtype=float)
            if actual is not None:
                line, = ax.plot(waves, np.where(actual > 0, convert(actual), np.nan), ls='none',
                    marker=marker, ms=size, mfc=color, mec='#263238', mew=1.1, label=label, zorder=9)
                line.set_gid('photometry:'+key)
                line.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
                valid = np.isfinite(error) & (error >= 0) & (actual > 0)
                if np.any(valid):
                    ax.errorbar(waves[valid], convert(actual)[valid], yerr=convert(error)[valid],
                                fmt='none', ecolor=color, elinewidth=1., capsize=2.5, zorder=8)
            if physical:
                line, = ax.plot(waves, np.where(model > 0, convert(model), np.nan), ls='none',
                    marker='o', ms=size+4., mfc='white', mec=color, mew=1.5, alpha=.7,
                    label=label if actual is None else None, zorder=6)
                line.set_gid('synthetic:'+key)
            for i, o in enumerate(observed):
                records.append(dict(component=key, band=o.band, wavelength_angstrom=pivots[i],
                    photometry_microjy=np.nan if actual is None else actual[i],
                    photometry_error_microjy=error[i], model_microjy=model[i]))

        draw_points('__total__', [c.name for c in scene.components], '#364d65', 's', 'HST total', 9.)
        if extended:
            draw_points('__stellar__', [c.name for c in extended], '#167e72', 'D',
                        'Total stellar' if physical else 'Extended', 8.)
        spectra = []
        for c in scene.components:
            point = c.name in point_names
            color, marker = ('#a54449', 'o') if point else COMPONENT_STYLES[extended.index(c) % len(COMPONENT_STYLES)]
            draw_points(c.name, [c.name], color, marker, 'PSF' if point and len(point_names) == 1 else c.name)
            if calibrated and not isinstance(c.sed, FreeBandSED) and hasattr(c.sed, 'spectrum'):
                w, flam = c.sed.spectrum(result.values, c.name, scene.context)
                valid = np.isfinite(w) & np.isfinite(flam)
                w, flam = np.asarray(w)[valid], np.asarray(flam)[valid]
                spectra.append((w, flam))
                visible = (w/1e4 >= limits[0]) & (w/1e4 <= limits[1]) & (flam > 0)
                line, = ax.plot(w[visible]/1e4, flam[visible]*1e17, color=color, lw=.85, alpha=.9, zorder=4)
                line.set_gid('stellar-spectrum:'+c.name)
        if spectra:
            # Sum on the union grid within common coverage; no spectral extrapolation.
            low = max(limits[0]*1e4, max(w[0] for w, f in spectra))
            high = min(limits[1]*1e4, min(w[-1] for w, f in spectra))
            grid = np.unique(np.concatenate([w[(w >= low) & (w <= high)] for w, f in spectra]))
            total_stellar = sum(np.interp(grid, w, f) for w, f in spectra)
            line, = ax.plot(grid/1e4, np.where(total_stellar > 0, total_stellar*1e17, np.nan),
                           color='#167e72', lw=1.35, zorder=5)
            line.set_gid('stellar-spectrum:total')
        ax.set(xscale='log', yscale='log', xlim=limits,
               xlabel='Observed wavelength (µm)' if calibrated else 'Band index',
               ylabel=(r'Observed $f_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)'
                       if calibrated else 'Flux density (µJy)'), title=title)
        if ylim is not None:ax.set_ylim(ylim)
        ticks = ([t for t in [.2, .3, .4, .6, .8, 1., 1.5, 2., 2.5, 3.] if limits[0] <= t <= limits[1]]
                 if xlim is not None else list(waves))
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        for index, o in enumerate(observed):
            throughput = getattr(o.throughput, 'bandpass', None)
            if throughput is None:continue
            from astropy import units as u
            w = throughput.waveset.to_value(u.AA)/1e4
            response = np.asarray(throughput(throughput.waveset).value)
            response = response/max(float(response.max()), 1e-30)
            color = BAND_COLORS[index % len(BAND_COLORS)]
            ax.fill_between(w, 0, .10*response, color=color, alpha=.20, transform=ax.get_xaxis_transform(), zorder=0)
            ax.plot(w, .10*response, color=color, lw=.9, transform=ax.get_xaxis_transform(), zorder=0)
            label_wave = waves[index]
            if index+1 < len(waves) and waves[index+1]/waves[index] < 1.15:label_wave *= .965
            if index > 0 and waves[index]/waves[index-1] < 1.15:label_wave *= 1.035
            ax.text(label_wave, .115, o.band, color=color, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        if physical:
            handles.append(Line2D([], [], ls='none', marker='o', ms=10., mfc='white', mec='#333333', mew=1.5, alpha=.7))
            labels.append('Synthetic')
        legend = ax.legend(handles, labels, loc='upper right', ncol=2, fontsize=9, frameon=False,
                           labelspacing=.35, columnspacing=1.2, handletextpad=.45)
        for label in legend.get_texts():
            label.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
        if rgb is not None:
            fig.set_layout_engine(None)
            fig.subplots_adjust(left=.105, right=.98, top=.95 if nuclear_detail else .94 if separate_components else .91,
                                bottom=.66 if nuclear_detail else .60 if separate_components else .38)
            panels = [('total', 'Total'), ('extended', 'Extended'), ('point', 'PSF'), ('data_minus_psf', 'Data - PSF')]
            panels += [('component:'+c.name, c.name) for c in extended]
            width = .875/len(panels)
            for i, (key, label) in enumerate(panels):
                inset = fig.add_axes([.105+i*width, .025 if nuclear_detail else .035, width*.92,
                                      .145 if nuclear_detail else .20 if separate_components else .22])
                inset.imshow(rgb['images'][key], origin='lower', interpolation='nearest')
                bar = rgb['bar_pixels'];h, image_width = rgb['images'][key].shape[:2]
                if i == 0 and bar < image_width*.8:
                    inset.plot([2, 2+bar], [2, 2], color='white', lw=2, solid_capstyle='butt')
                    text = inset.text(2, 3.5, rgb['bar_label'], color='white', fontsize=7, va='bottom')
                    text.set_path_effects([pe.withStroke(linewidth=1.5, foreground='black')])
                inset.set(xticks=[], yticks=[], title=label)
                inset.title.set_fontsize(10)
                for spine in inset.spines.values():spine.set(color='#46505a', linewidth=.8)
        panel_ylimits = {}
        if nuclear_detail:
            panel = fig.add_axes([.105, .45, .875/3*1.88, .13])
            nuclear_values = [];segment_bounds = []
            for spectrum in nuclear_spectra:
                line = np.asarray(spectrum['scaled_flam'])
                panel.plot(spectrum['wave_micron'], line, color=spectrum.get('color', '#3e6ca8'), lw=.65)
                nuclear_values.extend(line[np.isfinite(line)].tolist())
                if np.isfinite(line).any():segment_bounds.append(np.percentile(line[np.isfinite(line)], [3, 92]))
            for name in point_names:
                for source in ax.lines:
                    if source.get_gid() not in {'photometry:'+name, 'synthetic:'+name}:continue
                    synthetic = source.get_gid().startswith('synthetic:')
                    copied, = panel.plot(source.get_xdata(), source.get_ydata(), ls='none', marker=source.get_marker(),
                               ms=11.5 if synthetic else 7.5, mfc=source.get_markerfacecolor(),
                               mec=source.get_markeredgecolor(), mew=1.4 if synthetic else 1.1,
                               zorder=source.get_zorder(), alpha=.7 if synthetic else 1.)
                    if not synthetic:copied.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
                    nuclear_values.extend(np.asarray(source.get_ydata())[np.isfinite(source.get_ydata())].tolist())
                rows = [r for r in records if r['component'] == name]
                actual = np.array([r['photometry_microjy'] for r in rows])
                error = np.array([r['photometry_error_microjy'] for r in rows])
                valid = np.isfinite(actual) & np.isfinite(error)
                if valid.any():
                    panel.errorbar(waves[valid], convert(actual)[valid], yerr=convert(error)[valid],
                                   fmt='none', ecolor='#a54449', elinewidth=1.1, capsize=3, zorder=8)
            finite = np.asarray(nuclear_values)
            low, high = np.percentile(finite, [3, 92]) if len(finite) else (0., 1.)
            if segment_bounds:low=min(v[0] for v in segment_bounds);high=max(v[1] for v in segment_bounds)
            band_values = [convert([result.fluxes[n][o.band] for o in observed]) for n in point_names]
            low = min(low, 0., np.min(band_values));high = max(high, np.max(band_values))
            pad = max((high-low)*.15, 1e-6)
            panel.set(xscale='log', xlim=limits, ylim=(low-pad, high+pad))
            panel.set_title('PSF: total - extended' if nuclear_spectra else 'PSF', fontsize=11, pad=4)
            panel.xaxis.set_major_locator(FixedLocator([v for v in [.3, .6, 1., 1.5, 2., 2.5] if limits[0] <= v <= limits[1]]))
            panel.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
            panel.xaxis.set_minor_formatter(NullFormatter())
            panel.tick_params(labelsize=9);panel.set_xlabel('µm', fontsize=10, labelpad=1)
            panel_ylimits['nucleus'] = list(panel.get_ylim())
        if separate_components and extended:
            # Each continuum zoom has its own limits; narrow peaks remain in overview.
            count = len(extended)
            for i, c in enumerate(extended):
                if nuclear_detail:
                    row, column = (0, 2) if i == 0 else (1+(i-1)//3, (i-1)%3)
                    panel = fig.add_axes([.105+column*.875/3, .45-row*.20, .875/3*.88, .13])
                else:panel = fig.add_axes([.105+i*.875/count, .33, .875/count*.88, .19])
                color = COMPONENT_STYLES[i % len(COMPONENT_STYLES)][0]
                continuum = [];photometric = []
                for line in ax.lines:
                    if line.get_gid() not in {'stellar-spectrum:'+c.name, 'photometry:'+c.name, 'synthetic:'+c.name}:continue
                    synthetic = line.get_gid().startswith('synthetic:')
                    actual = line.get_gid().startswith('photometry:')
                    copied, = panel.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                               ls=line.get_linestyle(), lw=.65, marker=line.get_marker(),
                               ms=11.5 if synthetic else 7.5, mfc=line.get_markerfacecolor(),
                               mec=line.get_markeredgecolor(), mew=1.4 if synthetic else 1.1,
                               zorder=line.get_zorder(), alpha=.7 if synthetic else 1.)
                    if actual:copied.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
                    y = np.asarray(line.get_ydata());y = y[np.isfinite(y) & (y > 0)]
                    if line.get_gid().startswith('stellar-spectrum:'):continuum.extend(y.tolist())
                    else:photometric.extend(y.tolist())
                rows = [r for r in records if r['component'] == c.name]
                error = np.array([r['photometry_error_microjy'] for r in rows])
                actual = np.array([r['photometry_microjy'] for r in rows])
                good = np.isfinite(error) & np.isfinite(actual) & (actual > 0)
                if good.any():panel.errorbar(waves[good], convert(actual)[good], yerr=convert(error)[good],
                                            fmt='none', ecolor=color, elinewidth=1.1, capsize=3, zorder=8)
                base = np.asarray(continuum or photometric or [1.])
                low, high = np.percentile(base, [3, 90])
                if photometric:low=min(low,min(photometric));high=max(high,max(photometric))
                if good.any():
                    low_errors = convert(actual-error)[good]
                    low = min(low, np.min(low_errors[low_errors > 0])) if np.any(low_errors > 0) else low
                    high = max(high, np.max(convert(actual+error)[good]))
                panel.set(xscale='log', yscale='log', xlim=limits, ylim=(max(low*.65, 1e-30), high*1.4))
                panel.set_title(c.name, color=color, fontsize=11, pad=4)
                panel.xaxis.set_major_locator(FixedLocator([v for v in [.3, .6, 1., 2.] if limits[0] <= v <= limits[1]]))
                panel.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
                panel.xaxis.set_minor_formatter(NullFormatter())
                panel.tick_params(labelsize=9)
                ylo, yhi = panel.get_ylim()
                candidates = np.array([m*10.**exponent for exponent in range(int(np.floor(np.log10(ylo))), int(np.ceil(np.log10(yhi)))+1)
                                       for m in [1., 2., 5.] if ylo <= m*10.**exponent <= yhi])
                if len(candidates) > 4:candidates = candidates[np.unique(np.round(np.linspace(0, len(candidates)-1, 4)).astype(int))]
                panel.yaxis.set_major_locator(FixedLocator(candidates))
                panel.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:g}'))
                panel.yaxis.set_minor_formatter(NullFormatter())
                panel.set_xlabel('µm', fontsize=10, labelpad=1)
                panel_ylimits[c.name] = list(panel.get_ylim())
        directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
        note = ('Filled: measured free-band photometry; white/colored edge: model band predictions. '
                'Stellar curves cover the full displayed range; context spectra are not fitted. '
                'Nuclear predictions are independent band amplitudes, not an FSPS spectrum.')
        fig.savefig(directory/(prefix+'.pdf'), dpi=300, metadata={'Subject': note})
        fig.savefig(directory/(prefix+'.png'), dpi=220)
        Table(rows=records).write(directory/(prefix+'_photometry.ecsv'), format='ascii.ecsv', overwrite=True)
        import json
        settings = dict(title=title, units='pivot_equivalent_scaled_flam' if calibrated else 'microjy',
                        flux_definition='total_microjy', context_used_in_likelihood=False,
                        photometry_source=photometry_source, measured_error_type='local linear covariance',
                        synthetic_style='white face, colored edge', measured_style='filled color',
                        photometry_emphasis='Opaque filled foreground markers with dark edges/white halo and stronger errors; translucent hollow synthetic markers behind.',
                        nuclear_curves=False, total_stellar_curve=bool(spectra),
                        separate_components=separate_components, spectrum_provenance=spectrum_provenance,
                        panel_ylimits=panel_ylimits, nuclear_subtracted_spectra=bool(nuclear_spectra),
                        detail_limit_policy='Continuum/photometry zoom; narrow line peaks may exceed detail limits. Nuclear residual shown on a linear axis, including negative values.',
                        xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
        if rgb is not None:settings['rgb'] = {k: v for k, v in rgb.items() if k != 'images'}
        (directory/(prefix+'_display.json')).write_text(json.dumps(settings, indent=2)+'\n')
        return fig
