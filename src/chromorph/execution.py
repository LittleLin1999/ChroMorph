"""Prepared, process-local execution state for one declarative scene.

The model owns scientific definitions; this object owns caches, rendering and
spectral resources. Mathematical likelihood kernels do not import samplers.
"""
import numpy as np
import time
from .cache import ExactLRU
from .sed import FreeBandSED


class PreparedEvaluator:
    def __init__(self, model):
        self.model = model
        self.renderer = model._renderer
        self._basis_cache = ExactLRU(model.cache_bytes//2)
        self._whitened_cache = ExactLRU(model.cache_bytes//2)
        self._compressed = None
        # One required preparation per observation, independent of optional
        # spatial-cache budget. Obsolete versions are replaced, never retained.
        self._fixed_data = {}
        self._gaussian_cache = self._fixed_data
        self._gaussian_tokens = {}
        self._likelihood_engines = {}
        self._policy = None
        self._spectral_engine = None
        self.metrics = dict(render_calls=0, noise_preparations=0, noise_preparation_seconds=0.,
                            whitening_calls=0, whitening_seconds=0., residual_calls=0,
                            prepared_input_bytes=0,covariance_factorizations=0)
        self.closed = False
        self._workers = []
        self._plan_revision = None
        self._morphology_names = {}
        self._observation_names = {}

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

    def prepare(self):
        if self.closed: raise RuntimeError('Evaluator is closed; prepare a new workspace')
        return self

    def _ensure_renderer(self):
        self.prepare()
        if self.renderer is None:
            from .rendering import GalSimRenderer
            self.renderer = GalSimRenderer()

    def _plans(self):
        token = (self.registry.revision, len(self.components), len(self.observations))
        if token != self._plan_revision:
            self._components_by_name = {c.name:c for c in self.components}
            self._morphology_names = {c.name: tuple((key,c.name+'.morphology.'+key) for key in c.parameters)
                                      for c in self.components}
            self._observation_names = {o.name: tuple('observation.'+o.name+'.'+key
                for key in ['dx','dy','log_calibration','background']) for o in self.observations}
            self._plan_revision = token

    def component_named(self, name):
        self._plans()
        return self._components_by_name[name]

    def whiten(self, observation, value):
        fixed=self.gaussian_data(observation);started=time.perf_counter()
        result=fixed.whitening.whiten(value)
        self.metrics['whitening_calls']+=1
        self.metrics['whitening_seconds']+=time.perf_counter()-started
        return result

    def whitened_template(self, component, observation, values):
        fixed=self.gaussian_data(observation)
        spatial='sky' if component is None else self.spatial_key(component,observation,values)
        key=(observation.name,fixed.noise_token,spatial)
        column=self._whitened_cache.get(key)
        if column is None:
            raw=(np.ones(len(fixed.included)) if component is None else
                 self.unit_template(component,observation,values).ravel()[fixed.included])
            column=self.whiten(observation,raw);column.setflags(write=False);self._whitened_cache[key]=column
        return column

    def normalized_values(self, values):
        aliases=self.registry.layout.aliases
        if any(values[name]!=values[root] for name,root in aliases):
            values=dict(values)
            for name,root in aliases:values[name]=values[root]
        return values

    def morphology(self, component, values):
        self._plans();roots=self.registry.layout.roots
        return {key: values[roots[name]] for key,name in self._morphology_names[component.name]}

    def registration(self, observation, values):
        self._plans();dx,dy,_,_=self._observation_names[observation.name]
        return np.array([values[dx],values[dy]])

    def image_coefficients(self, observation, values, fluxes):
        self._plans();_,_,calibration,background=self._observation_names[observation.name]
        scale=np.exp(values[calibration])/observation.ujy_per_rate
        return np.array([fluxes[c.name][observation.band]*scale for c in self.components]),float(values[background]),scale

    def store_template(self, component, observation, values, image):
        image.setflags(write=False)
        self._basis_cache[(observation.name,self.spatial_key(component,observation,values))]=image

    def unit_template(self, component, observation, values):
        self._ensure_renderer();key=(observation.name,self.spatial_key(component,observation,values))
        image=self._basis_cache.get(key)
        if image is None:
            image=np.asarray(self.renderer.render(component,self.morphology(component,values),observation,self.registration(observation,values)),dtype=float)
            self.metrics['render_calls']+=1;image.setflags(write=False);self._basis_cache[key]=image
        return image

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
        return {c.name:self.unit_template(c,observation,values) for c in self.components}

    def basis(self, values):
        return {o.name: self.basis_observation(o, values) for o in self.observations}

    @property
    def compressed(self):
        if self._compressed is None:
            from .likelihood import CompressedGaussianLikelihood
            self._compressed = CompressedGaussianLikelihood(self)
        return self._compressed

    def component_fluxes(self,values):
        self.prepare()
        from .spectral import SpectralEngine
        if self._spectral_engine is None:self._spectral_engine=SpectralEngine(self,workers=self.sed_workers)
        return self._spectral_engine.fluxes(self.normalized_values(values))

    def _assemble(self, values, *, basis=None, component_images=False):
        if not self.components: raise ValueError('Scene has no components')
        basis = self.basis(values) if basis is None else basis
        fluxes = self.component_fluxes(values)
        if any(not np.isfinite(v) or v<0 for bands in fluxes.values() for v in bands.values()):
            raise ValueError('SED must predict finite nonnegative microJy fluxes')
        models={};images={}
        for o in self.observations:
            calibration=np.exp(values['observation.'+o.name+'.log_calibration'])/o.ujy_per_rate
            model=np.full(o.data.shape,values['observation.'+o.name+'.background'],dtype=float)
            if component_images: images[o.name]={}
            for c in self.components:
                contribution=basis[o.name][c.name]*(fluxes[c.name][o.band]*calibration)
                model+=contribution
                if component_images:images[o.name][c.name]=contribution
            models[o.name]=model
        return models,images,fluxes

    def predict(self, values=None, *, basis=None):
        values=self.registry.unpack() if values is None else values
        return self._assemble(values,basis=basis,component_images=True)

    def whitened_residual(self, observation, model):
        fixed=self.gaussian_data(observation)
        return self.whiten(observation,fixed.masked_data-model.ravel()[fixed.included])

    def residual(self, values, *, basis=None, include_priors=True):
        models, _, _ = self._assemble(values, basis=basis)
        data = np.concatenate([self.whitened_residual(o, models[o.name]) for o in self.observations])
        if not np.isfinite(data).all(): raise ValueError('Nonfinite likelihood')
        self.metrics['residual_calls'] += 1
        return np.r_[data, self.registry.prior_residuals(values)] if include_priors else data

    def pixel_loglikelihood(self, values):
        residual = self.residual(values, include_priors=False)
        return float(sum(self.gaussian_data(o).whitening.log_normalization for o in self.observations)
                     -.5*(residual@residual))

    def clear_caches(self):
        if self._spectral_engine is not None:self._spectral_engine.close();self._spectral_engine=None
        self._basis_cache.clear()
        self._whitened_cache.clear()
        self._gaussian_cache.clear()
        self._gaussian_tokens.clear()
        self.metrics['prepared_input_bytes'] = 0
        self._likelihood_engines.clear()
        if self._policy is not None:self._policy.clear()
        self._compressed = None

    def gaussian_data(self, observation):
        """Reuse fixed data, mask and factorization across every execution role."""
        self.prepare()
        from .noise import whitening_for
        from .gaussian import FixedGaussianData
        observation.ensure_frozen()
        token = observation.likelihood_token()
        fixed = self._fixed_data.get(observation.name)
        if fixed is None or fixed.token != token:
            started = time.perf_counter()
            noise_token = observation.noise_token()
            same_noise = fixed is not None and fixed.noise_token == noise_token
            whitening = fixed.whitening if same_noise else whitening_for(observation)
            included = fixed.included if same_noise else np.flatnonzero(observation.mask.ravel())
            masked = observation.data.ravel()[included]
            whitening_started=time.perf_counter()
            target = whitening.whiten(masked)
            self.metrics['whitening_calls']+=1
            self.metrics['whitening_seconds']+=time.perf_counter()-whitening_started
            for array in [included, masked, target]: array.setflags(write=False)
            fixed = FixedGaussianData(token, whitening, target, masked, included, noise_token)
            self._fixed_data[observation.name] = fixed
            self._gaussian_tokens[observation.name] = token
            self.metrics['noise_preparations'] += int(not same_noise)
            from .noise import CholeskyWhitening
            self.metrics['covariance_factorizations'] += int(not same_noise and isinstance(whitening,CholeskyWhitening))
            self.metrics['noise_preparation_seconds'] += time.perf_counter()-started
            self.metrics['prepared_input_bytes'] = sum(x.nbytes for x in self._fixed_data.values())
        return fixed

    @property
    def policy(self):
        if self._policy is None:
            from .policy import ExecutionPolicy
            self._policy=ExecutionPolicy(self,**self.model.policy_options)
        return self._policy

    def nuclear_plan(self, linear):
        result={name.split('.')[-1]:name for name in linear}
        if len(result)!=len(linear):raise ValueError('Sample multiple overlapping nuclear amplitudes explicitly with proper priors')
        return result

    def compressed_evaluation(self, values, linear=()):
        from .gaussian import GaussianAccumulator
        values=self.normalized_values(values)
        if linear:
            values=dict(values)
            for name in linear:values[name]=0.
        statistics=self.compressed.statistics(values);coefficients=self.compressed.coefficients(values)
        plans=self.nuclear_plan(linear);indices={c.name:i for i,c in enumerate(self.components)}
        accumulator=GaussianAccumulator()
        for o in self.observations:
            stat=statistics[o.name];residual=stat.z-stat.R@coefficients[o.name]
            parameter=plans.get(o.band);column=None
            if parameter is not None:
                self._plans()
                calibration=np.exp(values[self._observation_names[o.name][2]])/o.ujy_per_rate
                column=stat.R[:,indices[parameter.split('.')[0]]]*calibration
            accumulator.add(residual,stat.log_normalization,perpendicular_chi2=stat.perpendicular_chi2,
                            parameter=parameter,column=column)
        return accumulator.finish()

    def evaluate_gaussian(self, values, *, mode='auto', linear=()):
        from .direct import FullPixelLikelihood
        if mode=='auto':return self.policy.evaluate(values,linear)
        if mode=='compressed':return self.compressed_evaluation(values,linear)
        if mode not in {'full','aggregate'}:raise ValueError('Use auto, full, aggregate or compressed')
        if mode not in self._likelihood_engines:
            self._likelihood_engines[mode]=FullPixelLikelihood(self,rendering='aggregate' if mode=='aggregate' else 'components')
        return self._likelihood_engines[mode].evaluate(values,linear)

    def loglikelihood(self, values=None, *, mode='auto'):
        values=self.registry.unpack() if values is None else values
        return self.evaluate_gaussian(values,mode=mode).loglikelihood

    def marginalized_loglikelihood(self, values, linear, *, mode='auto'):
        answer=self.evaluate_gaussian(values,mode=mode,linear=linear)
        return answer.loglikelihood,answer.conditionals

    def _eligible_linear(self):
        from .linear import LinearProjection
        plans=LinearProjection(self.model,stellar_mass=False,background=False).amplitudes
        return [a.name for a in plans if self.registry.parameters[a.name].lower==0
                and not np.isfinite(self.registry.parameters[a.name].upper)]

    def project(self, values, eligible, basis=None):
        from .linear import LinearProjection
        return LinearProjection(self.model,stellar_mass=False,background=False,names=eligible).solve(values)

    def release_workers(self, resource):
        resource.close()
        if resource in self._workers:self._workers.remove(resource)

    def derivative_workers(self, names, eligible, workers, *, amplitudes=None):
        from .workers import JacobianWorkers
        resource=JacobianWorkers(self.model,names,eligible,workers,amplitudes=amplitudes)
        self._workers.append(resource)
        return resource

    def close(self):
        if self.closed: return
        for worker in self._workers: worker.close()
        self._workers.clear()
        self.clear_caches()
        for component in self.components:
            if hasattr(component.sed, 'close'): component.sed.close()
        self.closed = True

    def __enter__(self):
        if self.closed: raise RuntimeError('Evaluator is closed; prepare a new workspace')
        return self

    def __exit__(self, *args): self.close()
