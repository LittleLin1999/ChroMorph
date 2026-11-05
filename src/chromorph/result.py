"""Local identifiability diagnostics and scientifically explicit result products."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import html
import importlib.metadata
import json
import numpy as np
from astropy.io import fits
from astropy.table import Table
from .sed import FreeBandSED


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


class FitResult:
    def __init__(self, scene, values, runs, projected):
        self.scene = scene
        self.values = dict(values)
        self.runs = runs
        self.projected = list(projected)
        self.best_optimizer = min(runs, key=lambda r: r['objective'])
        self.success = bool(self.best_optimizer['success'])
        self.models, self.images, self.fluxes = scene.predict(values)
        r = scene.residual(values, include_priors=False)
        self.chi2 = float(r@r)
        self.n_pixels = len(r)
        self.n_parameters = len(scene.registry.names())
        self.bic_like = self.chi2 + self.n_parameters*np.log(self.n_pixels)
        self.warnings = ["Diagonal fixed-noise likelihood; drizzle covariance is not included.",
                         "Local linear uncertainties do not capture nonlinearity or label switching."]

        if not self.success:
            self.warnings.append('Best optimizer run did not converge; inspect multistart results before interpretation.')

    def estimate_uncertainty(self):
        scene = self.scene
        names = scene.registry.names()
        self.parameter_names = names
        lo, hi = scene.registry.bounds(names)
        vector = np.array([self.values[n] for n in names])
        central = scene.residual(self.values)
        jac = np.empty((len(central), len(names)))
        for i, name in enumerate(names):
            step = 1e-4*max(abs(vector[i]), 1.)
            left, right = max(lo[i], vector[i]-step), min(hi[i], vector[i]+step)
            def evaluate(value):
                v = dict(self.values)
                v[name] = value
                for alias in scene.registry.ties:
                    v[alias] = v[scene.registry.resolve(alias)]
                return scene.residual(v)
            jac[:, i] = (evaluate(right)-evaluate(left))/(right-left)
        if not names:
            self.covariance = np.empty((0, 0)); self.correlation = self.covariance.copy()
            self.uncertainty = {}; self.identifiability = dict(rank=0, data_rank=0, null_modes=[], near_bounds=[], prior_information_fraction={})
            return
        _, singular, vt = np.linalg.svd(jac, full_matrices=False)
        tolerance = max(singular[0]*1e-7, 1e-10) if len(singular) else 1e-10
        rank = int(np.count_nonzero(singular > tolerance))
        good = vt[:len(singular)][singular > tolerance]
        self.covariance = (good.T/singular[singular > tolerance]**2)@good
        errors = np.sqrt(np.maximum(np.diag(self.covariance), 0))
        self.correlation = np.divide(self.covariance, errors[:, None]*errors[None, :],
                                      out=np.zeros_like(self.covariance), where=(errors[:, None]*errors[None, :]) > 0)
        null = vt[rank:]
        if jac.shape[0] < jac.shape[1]:
            from scipy.linalg import null_space
            null = null_space(jac, rcond=1e-7).T
        ambiguous = np.sum(null**2, axis=0) > 1e-6 if len(null) else np.zeros(len(names), dtype=bool)
        self.uncertainty = {n: None if ambiguous[i] else float(errors[i]) for i, n in enumerate(names)}
        if rank < len(names):
            self.warnings.append("Rank-deficient fit: some component parameters/flux exchanges are not identifiable.")
        data_jac = jac[:self.n_pixels]
        dsv = np.linalg.svd(data_jac, compute_uv=False)
        dr = int(np.count_nonzero(dsv > max(dsv[0]*1e-7, 1e-10))) if len(dsv) else 0
        prior_info = np.sum(jac[self.n_pixels:]**2, axis=0)
        all_info = np.sum(jac**2, axis=0)
        fraction = np.divide(prior_info, all_info, out=np.zeros_like(prior_info), where=all_info > 0)
        near = [n for n, v, low, high in zip(names, vector, lo, hi)
                if min(v-low, high-v) <= 1e-4*max(abs(v), 1.)]
        if near:
            self.warnings.append("Parameters at bounds have unreliable symmetric errors: " + ', '.join(near))
        self.identifiability = dict(rank=rank, data_rank=dr, singular_values=singular,
                                   null_modes=[{n: float(x) for n, x in zip(names, mode) if abs(x) > .05} for mode in null],
                                   near_bounds=near, prior_information_fraction=dict(zip(names, fraction)))

    def sample(self, *, steps=1000, walkers=None, seed=0):
        """Optional emcee posterior, retaining free nuclear band amplitudes."""
        import emcee
        names = self.scene.registry.names()
        lo, hi = self.scene.registry.bounds(names)
        center = np.array([self.values[n] for n in names])
        ndim = len(names)
        if ndim == 0:
            raise ValueError("No varying parameters")
        walkers = max(2*ndim+2, 32) if walkers is None else walkers
        rng = np.random.default_rng(seed)
        start = np.clip(center + rng.normal(size=(walkers, ndim))*1e-4*np.maximum(abs(center), 1), lo+1e-10, hi-1e-10)
        def logp(x):
            if not ((x >= lo).all() and (x <= hi).all()):
                return -np.inf
            try:
                v = self.scene.registry.unpack(x, names)
                r = self.scene.residual(v)
                return -.5*float(r@r)
            except ValueError:
                return -np.inf
        sampler = emcee.EnsembleSampler(walkers, ndim, logp)
        sampler.random_state = np.random.RandomState(seed).get_state()
        sampler.run_mcmc(start, steps, progress=False)
        return sampler

    def write(self, directory, *, configuration=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        scene = self.scene
        versions = {}
        for name in ['chromorph', 'numpy', 'scipy', 'astropy', 'galsim', 'fsps', 'stsynphot']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = 'not installed'
        anchor = next((c for c in scene.components if getattr(c, 'nuclear', False) or c.name == 'nucleus'), None)
        centroids = {}
        for c in scene.components:
            xy = np.array([self.values[f'{c.name}.morphology.{a}'] for a in ['x', 'y']])
            offset = None if anchor is None else xy-np.array([self.values[f'{anchor.name}.morphology.{a}'] for a in ['x', 'y']])
            centroids[c.name] = dict(reference_xy=xy, offset_reference_pixels=offset,
                                     offset_anchor=None if anchor is None else anchor.name)
        data = dict(values=self.values, chi2=self.chi2, bic_like=self.bic_like,
                    success=self.success, best_optimizer=self.best_optimizer, centroids=centroids,
                    n_pixels=self.n_pixels, n_parameters=self.n_parameters, multistart=self.runs,
                    projected_parameters=self.projected, uncertainty=self.uncertainty,
                    identifiability=self.identifiability, warnings=self.warnings,
                    context=asdict(scene.context), parameter_definitions={n: asdict(p) for n, p in scene.registry.parameters.items()},
                    ties=scene.registry.ties,
                    component_definitions={c.name: dict(spatial=type(c).__name__,
                        sed=c.sed.describe() if hasattr(c.sed, 'describe') else dict(model=type(c.sed).__name__))
                        for c in scene.components},
                    component_sed_classes={c.name: type(c.sed).__name__ for c in scene.components},
                    configuration=configuration, versions=versions,
                    observations={o.name: dict(band=o.band, psf=o.psf.qa(), provenance=o.provenance,
                                               affine_matrix=o.matrix, affine_offset=o.offset,
                                               throughput=None if o.throughput is None else dict(
                                                   obsmode=getattr(o.throughput, 'obsmode', None),
                                                   fingerprint=getattr(o.throughput, 'fingerprint', None))) for o in scene.observations})
        (directory/'result.json').write_text(json.dumps(_json(data), indent=2, allow_nan=False)+'\n')
        Table(rows=[dict(parameter=n, value=self.values[n], error=self.uncertainty.get(scene.registry.resolve(n)))
                    for n in scene.registry.parameters]).write(directory/'parameters.ecsv', format='ascii.ecsv', overwrite=True)
        photometry = []
        hdus = [fits.PrimaryHDU()]
        for o in scene.observations:
            planes = dict(DATA=o.data, MODEL=self.models[o.name], RESID=o.data-self.models[o.name],
                          RESIDSIG=(o.data-self.models[o.name])/o.sigma,
                          SIGMA=o.sigma, MASK=o.mask.astype(np.uint8), PSF=o.psf.image)
            for c in scene.components:
                image = self.images[o.name][c.name]
                planes['COMP_'+c.name] = image
                xy = o.matrix@np.array([self.values[f'{c.name}.morphology.{axis}'] for axis in ['x', 'y']])+o.offset
                xy += [self.values[f'observation.{o.name}.{axis}'] for axis in ['dx', 'dy']]
                photometry.append(dict(observation=o.name, band=o.band, component=c.name,
                                       sed_class=type(c.sed).__name__, total_microjy=self.fluxes[c.name][o.band],
                                       frame_rate=float(image.sum()), x_observation=float(xy[0]), y_observation=float(xy[1])))
            for key, array in planes.items():
                header = fits.Header() if o.header is None or key == 'PSF' else o.header.copy()
                header['BUNIT'] = 'DIMENSIONLESS' if key in {'MASK', 'RESIDSIG', 'PSF'} else 'ELECTRONS/S'
                if key == 'PSF':
                    header['PSFCENX'], header['PSFCENY'] = o.psf.center
                    header['PIXSCALE'] = o.psf.pixel_scale
                    header['PIXINTEG'] = o.psf.pixel_integrated
                hdus.append(fits.ImageHDU(np.asarray(array), header=header, name=f'{o.name}_{key}'))
        fits.HDUList(hdus).writeto(directory/'images.fits', overwrite=True, checksum=True)
        Table(rows=photometry).write(directory/'photometry.ecsv', format='ascii.ecsv', overwrite=True)
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(self.covariance, name='COVARIANCE'),
                      fits.ImageHDU(self.correlation, name='CORRELATION'),
                      fits.BinTableHDU(Table({'parameter': self.parameter_names}), name='PARAMETERS')]).writeto(
                          directory/'uncertainty.fits', overwrite=True, checksum=True)
        self._plots(directory)
        body = '<h1>ChroMorph joint fit</h1><p>χ²={:.3f}; BIC-like={:.3f}</p>'.format(self.chi2, self.bic_like)
        body += '<ul>'+''.join('<li>'+html.escape(w)+'</li>' for w in self.warnings)+'</ul>'
        body += '<img src="diagnostic.png" alt="Image model and residuals" style="max-width:100%"><img src="sed.png" alt="Independent component SEDs" style="max-width:100%">'
        body += '<p><a href="result.json">Parameters and provenance</a> · <a href="photometry.ecsv">Component photometry</a> · <a href="images.fits">Images</a> · <a href="uncertainty.fits">Local covariance</a></p>'
        (directory/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>ChroMorph QA</title><body>'+body+'</body></html>')

    def _plots(self, directory):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        scene = self.scene
        fig, axes = plt.subplots(len(scene.observations), 3, figsize=(11, 3*len(scene.observations)), squeeze=False, layout='constrained')
        for row, o in enumerate(scene.observations):
            sky = self.values[f'observation.{o.name}.background']
            scale = np.median(o.sigma[o.mask])
            top = np.percentile(np.arcsinh((self.models[o.name]-sky)/scale), 99.5)
            for ax, image, title in zip(axes[row], [o.data-sky, self.models[o.name]-sky, (o.data-self.models[o.name])/o.sigma], ['Data - sky', 'Model - sky', 'Residual / sigma']):
                if title.startswith('Residual'):
                    shown = ax.imshow(np.where(o.mask, image, np.nan), origin='lower', vmin=-5, vmax=5, cmap='RdBu_r')
                else:
                    shown = ax.imshow(np.arcsinh(image/scale), origin='lower', vmin=-1, vmax=top, cmap='magma')
                ax.set_title(o.name+' '+title); fig.colorbar(shown, ax=ax, fraction=.046)
        fig.savefig(directory/'diagnostic.png', dpi=160); fig.savefig(directory/'diagnostic.pdf'); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
        observed = sorted({o.band: o for o in scene.observations}.values(), key=lambda o: o.pivot_angstrom or self.scene.bands.index(o.band)+1)
        waves = np.array([o.pivot_angstrom or (self.scene.bands.index(o.band)+1)*1e4 for o in observed])/1e4
        colors = plt.cm.tab10(np.linspace(0, .9, max(len(scene.components), 2)))
        for index, c in enumerate(scene.components):
            flux = [self.fluxes[c.name][o.band] for o in observed]
            # Free-band points are NEVER joined into an assumed nuclear spectrum.
            ax.plot(waves, flux, linestyle='none', marker=['o', 's', '^', 'v', 'D', 'P'][index % 6], color=colors[index], label=c.name+' ('+type(c.sed).__name__+')')
            if not isinstance(c.sed, FreeBandSED) and hasattr(c.sed, 'spectrum'):
                w, flam = c.sed.spectrum(self.values, c.name, scene.context)
                fnu = flam*w**2/2.99792458e18*1e29
                ax.plot(w/1e4, fnu, color=colors[index], lw=.8)
        ax.plot(waves, [sum(self.fluxes[c.name][o.band] for c in scene.components) for o in observed], 'k*', ms=10, label='Total predicted photometry')
        ax.set(xscale='log', yscale='log', xlabel='Observed wavelength (µm)', ylabel='Flux density (µJy)', xlim=(waves.min()*.85, waves.max()*1.15))
        ax.legend(fontsize=8); fig.savefig(directory/'sed.png', dpi=180); fig.savefig(directory/'sed.pdf'); plt.close(fig)
