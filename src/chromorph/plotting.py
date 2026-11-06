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
    aligned = {key: [] for key in ['extended', 'point', 'data_minus_psf']}
    noise, validity = [], []
    for o in observations:
        bx = o.matrix[0, 0]*xx+o.matrix[0, 1]*yy+o.offset[0]+values[f'observation.{o.name}.dx']
        by = o.matrix[1, 0]*xx+o.matrix[1, 1]*yy+o.offset[1]+values[f'observation.{o.name}.dy']
        point = sum(result.images[o.name][name] for name in points)
        host = sum(result.images[o.name][name] for name in extended)
        data_minus = o.data-values[f'observation.{o.name}.background']-point
        np.testing.assert_allclose(data_minus, host+o.data-result.models[o.name], rtol=1e-9, atol=1e-10)
        factor = o.ujy_per_rate/o.pixel_scale**2
        for key, image in zip(aligned, [host, point, data_minus]):
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
        if key == 'point':
            white = max(float(np.percentile(intensity, 99.5)), 1e-30)
            display_intensity = np.arcsinh(6*np.maximum(intensity, 0)/white)/np.arcsinh(6)
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
                note='Extended/data-minus-PSF share a stretch; PSF uses its own stretch. Masked pixels are black.')


def plot_component_sed(result, directory, *, title='ChroMorph component SEDs',
                       context_spectra=(), context_photometry=None, xlim=None,
                       ylim=None, rgb_crop=None, prefix='sed'):
    """Draw the fitted components plus optional display-only observational context.

    Context curves contain wave_micron, scaled_flam, label and optional color.
    Context photometry additionally contains error in the same f_lambda units.
    Stellar curves are limited to the observed-filter region. No curve is
    invented for a free-band SED. Return the figure for inspection/customizing.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
    from matplotlib import patheffects as pe
    scene = result.scene
    observed = sorted({o.band: o for o in scene.observations}.values(),
                      key=lambda o: o.pivot_angstrom or scene.bands.index(o.band)+1)
    calibrated_wavelengths = all(o.pivot_angstrom is not None for o in observed)
    pivots = np.array([o.pivot_angstrom or (scene.bands.index(o.band)+1)*1e4 for o in observed])
    waves = pivots/1e4
    convert = (lambda f: microjy_to_scaled_flam(f, pivots)) if calibrated_wavelengths else np.asarray
    styles = {'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 15,
              'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.minor.visible': True,
              'ytick.minor.visible': True, 'xtick.top': True, 'ytick.right': True,
              'ytick.left': True, 'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(styles):
        rgb = component_rgbs(result, crop=rgb_crop)
        fig, ax = plt.subplots(figsize=(10.4, 6.6), layout='constrained')
        for spectrum in context_spectra:
            ax.plot(spectrum['wave_micron'], spectrum['scaled_flam'], lw=.65,
                    color=spectrum.get('color', '#3e6ca8'), alpha=.85,
                    label=spectrum.get('label'), zorder=1, rasterized=True)
        if context_photometry is not None:
            p = context_photometry
            ax.errorbar(p['wave_micron'], p['scaled_flam'], yerr=p['error'], fmt='s', ms=6.5,
                        mfc='white', mec='#56687a', color='#56687a', lw=.8, capsize=2,
                        label=p.get('label', 'Aperture photometry (context)'), zorder=3)
        point_names = [c.name for c in scene.components if isinstance(c, PointSource)]
        extended = [c for c in scene.components if c.name not in point_names]
        total = convert([sum(result.fluxes[c.name][o.band] for c in scene.components) for o in observed])
        ax.plot(waves, np.where(total > 0, total, np.nan), ls='none', marker='s', ms=9.5,
                mfc='white', mec='#364d65', mew=1.5, label='HST total model', zorder=6)
        if extended:
            host = convert([sum(result.fluxes[c.name][o.band] for c in extended) for o in observed])
            ax.plot(waves, np.where(host > 0, host, np.nan), ls='none', marker='D', ms=7.8,
                    mfc='#167e72', mec='#263238', mew=.9, label='Sum of extended components', zorder=7)
        zero_points = []
        for index, c in enumerate(scene.components):
            flux = np.array([result.fluxes[c.name][o.band] for o in observed])
            point = c.name in point_names
            if point:
                color, marker = '#a54449', 'o'
                label = c.name+' (free band fluxes)'
            else:
                color, marker = COMPONENT_STYLES[extended.index(c) % len(COMPONENT_STYLES)]
                label = c.name+(' (free)' if isinstance(c.sed, FreeBandSED) else ' (stellar)')
            points = ax.plot(waves, np.where(flux > 0, convert(flux), np.nan), ls='none',
                    marker=marker, ms=7.5, mfc=color if point else 'white', mec=color,
                    mew=1.3, label=label, zorder=8)[0]
            points.set_gid('photometry:'+c.name)
            zero_points += [c.name+': '+o.band+' = 0' for o, f in zip(observed, flux) if f <= 0]
            if calibrated_wavelengths and not isinstance(c.sed, FreeBandSED) and hasattr(c.sed, 'spectrum'):
                w, flam = c.sed.spectrum(result.values, c.name, scene.context)
                visible = ((w >= pivots.min()*.85) & (w <= pivots.max()*1.15)
                           & np.isfinite(flam) & (flam > 0))
                line, = ax.plot(w[visible]/1e4, flam[visible]*1e17, color=color, lw=.85, alpha=.85, zorder=4)
                line.set_gid('stellar-spectrum:'+c.name)
        if zero_points:
            ax.text(.02, .97, '\n'.join(zero_points), transform=ax.transAxes, va='top', fontsize=8)
        ax.set(xscale='log', yscale='log', xlim=xlim or (waves.min()*.82, waves.max()*1.18),
               xlabel='Observed wavelength (µm)' if calibrated_wavelengths else 'Band index',
               ylabel=(r'Observed $f_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)'
                       if calibrated_wavelengths else 'Flux density (µJy)'), title=title)
        if ylim is not None:ax.set_ylim(ylim)
        if xlim is not None:
            ticks = [t for t in [.2, .3, .4, .6, .8, 1., 1.5, 2., 2.5, 3.] if xlim[0] <= t <= xlim[1]]
        else:ticks = list(waves)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        # The inset strip occupies a dedicated footer, never covering spectra.
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
        legend = ax.legend(loc='upper right', ncol=2, fontsize=9, frameon=False,
                  facecolor='white', labelspacing=.35, columnspacing=1.2, handletextpad=.45)
        for label in legend.get_texts():
            label.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
        if rgb is not None:
            # A separate lower row keeps all SED curves visible at every wavelength.
            fig.set_layout_engine(None)
            fig.subplots_adjust(left=.105, right=.98, top=.91, bottom=.40)
            labels = [('extended', 'Extended model RGB'), ('point', 'PSF source RGB'),
                      ('data_minus_psf', 'Data - fitted PSF RGB')]
            for i, (key, label) in enumerate(labels):
                inset = fig.add_axes([.38+i*.19, .075, .17, .215])
                inset.imshow(rgb['images'][key], origin='lower', interpolation='nearest')
                h, width = rgb['images'][key].shape[:2]
                bar = rgb['bar_pixels']
                if bar < width*.8:
                    inset.plot([2, 2+bar], [2, 2], color='white', lw=2, solid_capstyle='butt')
                    text = inset.text(2, 3.5, rgb['bar_label'], color='white', fontsize=8, va='bottom')
                    text.set_path_effects([pe.withStroke(linewidth=1.5, foreground='black')])
                inset.set(xticks=[], yticks=[], title=label)
                inset.title.set_fontsize(10)
                for spine in inset.spines.values():spine.set(color='#46505a', linewidth=.8)
            blue, green, red = np.array_split(rgb['bands'], 3) if len(rgb['bands']) >= 3 else (rgb['bands'][:1], rgb['bands'], rgb['bands'][-1:])
            channel_note = ('RGB channels\nR = '+ '/'.join(red)+'\nG = '+ '/'.join(green)
                            +'\nB = '+ ('mean('+ ', '.join(blue)+')' if len(blue) > 1 else '/'.join(blue)))
            fig.text(.105, .265, channel_note, fontsize=9, va='top', linespacing=1.3)
            fig.text(.105, .13, 'Extended / data - PSF:\nsame stretch, no smoothing.\nPSF: separate stretch.', fontsize=9, linespacing=1.3, va='top')
        note = 'Points: band-integrated model fluxes; curves: stellar models over HST coverage.'
        if context_spectra or context_photometry is not None:note += '\nObserved spectra / aperture photometry are display context only.'
        fig.text(.105, .005, note, fontsize=8, color='#515b65', va='bottom')
        directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory/(prefix+'.pdf'), dpi=300, metadata={'Subject': note})
        fig.savefig(directory/(prefix+'.png'), dpi=220)
        # Record display settings independently of scientific fit products.
        import json
        settings = dict(title=title, units='pivot_equivalent_scaled_flam' if calibrated_wavelengths else 'microjy',
                        flux_definition='total_model_microjy', context_used_in_likelihood=False,
                        nuclear_curves=False, xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
        if rgb is not None:settings['rgb'] = {k: v for k, v in rgb.items() if k != 'images'}
        (directory/(prefix+'_display.json')).write_text(json.dumps(settings, indent=2)+'\n')
        return fig
