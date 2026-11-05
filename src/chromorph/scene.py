"""A single image likelihood over every observation and spatial/SED component."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares, nnls
from astropy import units as u
from astropy.cosmology import Planck18
from .parameters import Registry, Parameter, identifier
from .sed import FreeBandSED, FSPSStellarSED, SEDContext
from .spatial import PointSource


class Scene:
    def __init__(self, observations, *, context=None, renderer=None,
                 fit_background=False, fit_registration=False, fit_calibration=False,
                 cache_bytes=256*1024**2):
        self.observations = tuple(observations)
        if not self.observations or len({o.name for o in self.observations}) != len(self.observations):
            raise ValueError("Need observations with distinct names")
        self.context = context or SEDContext()
        self.renderer = renderer
        self.components = []
        from .cache import ExactLRU
        self._basis_cache = ExactLRU(cache_bytes)
        self._compressed = None
        self.metrics = dict(render_calls=0)
        self.registry = Registry()
        for index, o in enumerate(self.observations):
            identifier(o.name)
            identifier(o.band)
            self.registry.add(f"observation.{o.name}.background", Parameter(o.background, vary=fit_background))
            # First observation anchors registration and calibration gauges.
            for axis in ['dx', 'dy']:
                self.registry.add(f"observation.{o.name}.{axis}",
                                  Parameter(0., -3, 3, vary=fit_registration and index > 0,
                                            prior=(0., .75) if fit_registration and index > 0 else None))
            self.registry.add(f"observation.{o.name}.log_calibration",
                              Parameter(0., -.5, .5, vary=fit_calibration and index > 0,
                                        prior=(0., .02) if fit_calibration and index > 0 else None))

    @property
    def bands(self):
        return list(dict.fromkeys(o.band for o in self.observations))

    def add_component(self, component):
        if any(c.name == component.name for c in self.components):
            raise ValueError("Component names must be unique")
        if any(c.sed is component.sed for c in self.components):
            raise ValueError("Each component must own a distinct SED model instance")
        nuclear = isinstance(component, PointSource) and (component.nuclear or component.name in {'nucleus', 'nuclear'})
        if nuclear:
            if type(component.sed) is not FreeBandSED:
                raise ValueError("The nucleus requires a completely free per-band SED")
            for p in component.sed.parameters.values():
                if p.prior is not None or not p.vary or p.lower != 0 or np.isfinite(p.upper):
                    raise ValueError("Nuclear fluxes cannot have stellar/color priors or fixed/upper-bound constraints")
        # Register atomically so invalid configuration cannot leave a partial scene.
        staged = Registry()
        for key, p in component.parameters.items():
            staged.add(f"{component.name}.morphology.{key}", p)
        component.sed.register(staged, component.name, self.bands)
        if isinstance(component.sed, FSPSStellarSED) and self.context.redshift > 0:
            name = f"{component.name}.sed.age"
            p = staged.parameters[name]
            age_limit = Planck18.age(self.context.redshift).to_value(u.Gyr)
            if p.value > age_limit or p.lower >= age_limit:
                raise ValueError("Age initialization/bounds exceed cosmic age")
            staged.parameters[name] = replace(p, upper=min(p.upper, age_limit))
        if set(staged.parameters) & set(self.registry.parameters):
            raise ValueError("Parameter namespace collision")
        self.registry.parameters.update(staged.parameters)
        self.components.append(component)
        return component

    def tie(self, name, target):
        for c in self.components:
            nuclear = isinstance(c, PointSource) and (c.nuclear or c.name in {'nucleus', 'nuclear'})
            if nuclear and any(n.startswith(f"{c.name}.flux.") for n in [name, target]):
                raise ValueError("Nuclear fluxes cannot be tied")
        self.registry.tie(name, target)

    def tie_centroids(self, names, anchor=None):
        if len(names) < 2:
            raise ValueError("A centroid group requires at least two components")
        anchor = names[0] if anchor is None else anchor
        for name in names:
            if name != anchor:
                for axis in ['x', 'y']:
                    self.tie(f"{name}.morphology.{axis}", f"{anchor}.morphology.{axis}")

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

    def predict(self, values=None, *, basis=None):
        if not self.components:
            raise ValueError("Scene has no components")
        values = self.registry.unpack() if values is None else values
        basis = self.basis(values) if basis is None else basis
        fluxes = {c.name: c.sed.fluxes(values, c.name, self.observations, self.context)
                  for c in self.components}
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
        self._basis_cache.clear()
        self._compressed = None

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

    def fit(self, *, multistart=3, seed=0, max_nfev=300, variable_projection=True,
            workers=1, likelihood='compressed', optimizer='trf', project_stellar_mass=True,
            estimate_uncertainty=True, checkpoint_directory=None, tolerance=1e-7):
        if likelihood == 'pixel':
            if optimizer != 'trf':
                raise ValueError('Pixel reference fitting supports trf; use compressed for L-BFGS-B')
            return self._fit_reference(multistart=multistart, seed=seed, max_nfev=max_nfev,
                                       variable_projection=variable_projection, workers=workers)
        if likelihood != 'compressed':
            raise ValueError('likelihood must be compressed or pixel')
        if workers != 1:
            raise ValueError('Compressed fitting uses one dependency-aware scene; parallelize independent models or use likelihood="pixel" for parallel rendering derivatives')
        from .optimize import fit_scene
        return fit_scene(self, optimizer=optimizer, multistart=multistart, seed=seed,
            max_nfev=max_nfev, variable_projection=variable_projection,
            project_stellar_mass=project_stellar_mass, estimate_uncertainty=estimate_uncertainty,
            checkpoint_directory=checkpoint_directory, tolerance=tolerance)

    def _fit_reference(self, *, multistart=3, seed=0, max_nfev=300, variable_projection=True, workers=1):
        from .result import FitResult
        if not self.components:
            raise ValueError("Scene has no components")
        multistart = (3 if multistart else 1) if isinstance(multistart, bool) else multistart
        if int(multistart) < 1:
            raise ValueError("multistart must be positive")
        registry = self.registry
        eligible = self._eligible_linear() if variable_projection else []
        names = registry.names(exclude=eligible)
        lo, hi = registry.bounds(names)
        initial = registry.initial(names)
        def decode(x):
            v = registry.unpack(x, names)
            basis = self.basis(v)
            return self.project(v, eligible, basis), basis
        def objective(x):
            v, basis = decode(x)
            return self.residual(v, basis=basis)
        objective(initial)  # dependency/configuration errors are never swallowed
        worker_map = None
        if workers != 1:
            if not isinstance(workers, int) or workers < 1:
                raise ValueError('workers must be a positive integer')
            from .parallel import JacobianWorkers
            worker_map = JacobianWorkers(self, names, eligible, workers)
        rng = np.random.default_rng(seed)
        runs = []
        try:
            for index in range(int(multistart)):
                start = initial.copy()
                if index:
                    width = np.where(np.isfinite(lo) & np.isfinite(hi), hi-lo, np.maximum(abs(initial), 1.))
                    start += rng.normal(0, .15, len(start))*width
                    start = np.maximum(np.minimum(start, hi-1e-9), lo+1e-9)
                if names:
                    extra = {} if worker_map is None else dict(workers=worker_map)
                    fit = least_squares(objective, start, bounds=(lo, hi), x_scale='jac',
                                        diff_step=1e-4, ftol=1e-7, xtol=1e-7, gtol=1e-7,
                                        max_nfev=max_nfev, **extra)
                    vector, success, message, nfev = fit.x, bool(fit.success), fit.message, fit.nfev
                else:
                    vector, success, message, nfev = start, True, 'Linear solve only', 1
                values, _ = decode(vector)
                residual = self.residual(values)
                runs.append(dict(values=values, objective=float(residual@residual),
                                 success=success, message=str(message), nfev=int(nfev)))
        finally:
            if worker_map is not None:
                worker_map.close()
        best = min(runs, key=lambda r: r['objective'])
        result = FitResult(self, best['values'], runs, eligible)
        result.estimate_uncertainty()
        return result

    def close(self):
        for c in self.components:
            if hasattr(c.sed, 'close'):
                c.sed.close()

    def sample_polychord(self, directory, **options):
        """Sample the joint posterior with analytically marginalized nuclear fluxes."""
        from .inference import PolyChordPosterior
        return PolyChordPosterior(self).run(directory, **options)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
