"""Prepared, process-local execution state for one declarative scene.

The model owns scientific definitions; this object owns caches, rendering and
spectral resources. Mathematical likelihood kernels do not import samplers.
"""
import numpy as np
from scipy.optimize import nnls
from .cache import ExactLRU
from .sed import FreeBandSED


class PreparedEvaluator:
    def __init__(self, model):
        self.model = model
        self.renderer = model._renderer
        self._basis_cache = ExactLRU(model.cache_bytes)
        self._compressed = None
        self._gaussian_cache = ExactLRU(model.cache_bytes//2)
        self._gaussian_tokens = {}
        self._likelihood_engines = {}
        self._spectral_engine = None
        self.metrics = dict(render_calls=0)
        self.closed = False

    @property
    def observations(self): return self.model.observations
    @property
    def components(self): return self.model.components
    @property
    def registry(self): return self.model.registry
    @property
    def context(self): return self.model.context
    @property
    def bands(self): return self.model.bands
    @property
    def sed_workers(self): return self.model.sed_workers

    def prepare(self): return self

    def _ensure_renderer(self):
        if self.renderer is None:
            from .rendering import GalSimRenderer
            self.renderer = GalSimRenderer()

    def morphology(self, component, values):
        return {key: values[f"{component.name}.morphology.{key}"] for key in component.parameters}

    def spatial_key(self, component, observation, values):
        self._ensure_renderer()
        morphology = self.morphology(component, values)
        shift = tuple(values[f'observation.{observation.name}.{axis}'] for axis in ['dx', 'dy'])
        psf = observation.psf
        renderer_key = getattr(self.renderer, 'cache_token', lambda: id(self.renderer))()
        return (component.name, type(component).__name__, renderer_key,
                tuple(sorted(morphology.items())), shift, tuple(observation.matrix.ravel()),
                tuple(observation.offset), observation.data.shape, psf.cache_token())

    def basis_observation(self, observation, values):
        self._ensure_renderer()
        result = {}
        shift = np.array([values[f'observation.{observation.name}.{axis}'] for axis in ['dx', 'dy']])
        for component in self.components:
            key = self.spatial_key(component, observation, values)
            cache_id = (observation.name, key)
            cached = self._basis_cache.get(cache_id)
            if cached is None:
                image = np.asarray(self.renderer.render(component, self.morphology(component, values), observation, shift), dtype=float)
                self.metrics['render_calls'] += 1
                image.setflags(write=False)
                cached = image
                self._basis_cache[cache_id] = cached
            result[component.name] = cached
        return result

    def basis(self, values):
        return {o.name: self.basis_observation(o, values) for o in self.observations}

    @property
    def compressed(self):
        if self._compressed is None:
            from .likelihood import CompressedGaussianLikelihood
            self._compressed = CompressedGaussianLikelihood(self)
        return self._compressed

    def component_fluxes(self,values):
        from .spectral import SpectralEngine
        if self._spectral_engine is None:self._spectral_engine=SpectralEngine(self,workers=self.sed_workers)
        return self._spectral_engine.fluxes(values)

    def predict(self, values=None, *, basis=None):
        if not self.components:
            raise ValueError("Scene has no components")
        values = self.registry.unpack() if values is None else values
        basis = self.basis(values) if basis is None else basis
        fluxes = self.component_fluxes(values)
        if any(not np.isfinite(v) or v < 0 for bands in fluxes.values() for v in bands.values()):
            raise ValueError("SED must predict finite nonnegative microJy fluxes")
        images, models = {}, {}
        for o in self.observations:
            calibration = np.exp(values[f"observation.{o.name}.log_calibration"])
            images[o.name] = {c.name: basis[o.name][c.name]*fluxes[c.name][o.band]/o.ujy_per_rate*calibration
                              for c in self.components}
            sky = values[f"observation.{o.name}.background"]
            models[o.name] = np.full(o.data.shape, sky) + sum(images[o.name].values())
        return models, images, fluxes

    def residual(self, values, *, basis=None, include_priors=True):
        models, _, _ = self.predict(values, basis=basis)
        from .noise import whitening_for
        data = np.concatenate([whitening_for(o).whiten((o.data-models[o.name])[o.mask])
                               for o in self.observations])
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite likelihood")
        return np.r_[data, self.registry.prior_residuals(values)] if include_priors else data

    def pixel_loglikelihood(self, values):
        from .noise import whitening_for
        residual = self.residual(values, include_priors=False)
        return float(sum(whitening_for(o).log_normalization for o in self.observations)
                     -.5*(residual@residual))

    def clear_caches(self):
        if self._spectral_engine is not None:self._spectral_engine.close();self._spectral_engine=None
        self._basis_cache.clear()
        self._gaussian_cache.clear()
        self._gaussian_tokens.clear()
        self._likelihood_engines.clear()
        self._compressed = None

    def gaussian_data(self, observation):
        """Reuse only parameter-independent whitening and data statistics."""
        from .noise import whitening_for
        from .gaussian import FixedGaussianData
        token = observation.likelihood_token()
        if self._gaussian_tokens.get(observation.name) != token:
            observation.freeze()
            token = observation.likelihood_token()
            self._gaussian_tokens[observation.name] = token
        key = (observation.name, token)
        fixed = self._gaussian_cache.get(key)
        if fixed is None:
            whitening = whitening_for(observation)
            target = whitening.whiten(observation.data[observation.mask])
            target.setflags(write=False)
            fixed = FixedGaussianData(token, whitening, target)
            self._gaussian_cache[key] = fixed
        return fixed

    def loglikelihood(self, values=None, *, mode='auto'):
        """Normalized Gaussian likelihood, without profiling nuisance parameters."""
        from .direct import FullPixelLikelihood, AutoGaussianLikelihood
        values = self.registry.unpack() if values is None else values
        if mode == 'compressed':return self.compressed.loglikelihood(values)
        if mode not in {'auto','full'}:raise ValueError('Use auto, full or compressed')
        if mode not in self._likelihood_engines:
            self._likelihood_engines[mode] = (FullPixelLikelihood(self) if mode=='full' else AutoGaussianLikelihood(self))
        return self._likelihood_engines[mode].loglikelihood(values)

    def _eligible_linear(self):
        linked = set(self.registry.ties) | set(self.registry.ties.values())
        return [n for c in self.components if isinstance(c.sed, FreeBandSED)
                for band in self.bands for n in [f"{c.name}.flux.{band}"]
                if n not in linked and self.registry.parameters[n].vary
                and self.registry.parameters[n].prior is None
                and self.registry.parameters[n].lower == 0
                and not np.isfinite(self.registry.parameters[n].upper)]

    def project(self, values, eligible, basis):
        """One NNLS over all images; only eligible linear fluxes are eliminated."""
        if not eligible:
            return values
        base = dict(values)
        for n in eligible:
            base[n] = 0.
        baseline, _, _ = self.predict(base, basis=basis)
        from .noise import whitening_for
        whiteners = {o.name: whitening_for(o) for o in self.observations}
        target = np.concatenate([whiteners[o.name].whiten((o.data-baseline[o.name])[o.mask])
                                 for o in self.observations])
        design = np.zeros((len(target), len(eligible)))
        row = 0
        for o in self.observations:
            count = o.mask.sum()
            for col, name in enumerate(eligible):
                component, _, band = name.split('.')
                if band == o.band:
                    rate = np.exp(values[f"observation.{o.name}.log_calibration"])/o.ujy_per_rate
                    design[row:row+count, col] = whiteners[o.name].whiten((basis[o.name][component]*rate)[o.mask])
            row += count
        fluxes, _ = nnls(design, target, maxiter=max(100, 10*len(eligible)))
        updated = dict(values)
        updated.update(zip(eligible, map(float, fluxes)))
        return updated

    def close(self):
        if self.closed: return
        self.clear_caches()
        for component in self.components:
            if hasattr(component.sed, 'close'): component.sed.close()
        self.closed = True

    def __enter__(self):
        if self.closed: raise RuntimeError('Evaluator is closed; prepare a new workspace')
        return self

    def __exit__(self, *args): self.close()
