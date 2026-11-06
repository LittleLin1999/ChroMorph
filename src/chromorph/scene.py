"""Declarative joint scene with thin fitting/evaluation conveniences."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18
from .parameters import Registry, Parameter, identifier
from .sed import FreeBandSED, FSPSStellarSED, SEDContext
from .spatial import PointSource


class Scene:
    def __init__(self, observations, *, context=None, renderer=None,
                 fit_background=False, fit_registration=False, fit_calibration=False,
                 cache_bytes=256*1024**2, allow_nuclear_flux_priors=False, sed_workers=1):
        self.observations = tuple(observations)
        if not self.observations or len({o.name for o in self.observations}) != len(self.observations):
            raise ValueError("Need observations with distinct names")
        self.context = context or SEDContext()
        # Compatibility argument only: independent statistical flux priors are
        # always allowed and never specify a nuclear spectral shape.
        self.allow_nuclear_flux_priors = bool(allow_nuclear_flux_priors)
        self.sed_workers=int(sed_workers)
        self._renderer = renderer
        self.cache_bytes = int(cache_bytes)
        self._evaluator = None
        self.components = []
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
                if not p.vary or p.lower < 0:
                    raise ValueError("Nuclear fluxes must remain independent, nonnegative and varying")
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

    def nuclear_flux_names(self):
        """One independent amplitude per observed band and nuclear component."""
        from .grouping import component_groups
        return [f'{name}.flux.{band}'
                for name in component_groups(self.components)['nucleus']
                for band in self.bands]

    def validate_nuclear_seds(self):
        """Guard the nuclear invariants after explicit registry edits, too."""
        from .grouping import component_groups
        nuclear = set(component_groups(self.components)['nucleus'])
        for component in self.components:
            if component.name in nuclear and type(component.sed) is not FreeBandSED:
                raise ValueError('The nucleus requires a completely free per-band SED')
        names = set(self.nuclear_flux_names())
        for name in names:
            p = self.registry.parameters[name]
            if not p.vary or p.lower < 0:
                raise ValueError('Nuclear fluxes must remain independent, nonnegative and varying')
        if names.intersection(self.registry.ties) or names.intersection(self.registry.ties.values()):
            raise ValueError('Nuclear fluxes cannot be tied')

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

    def prepare(self):
        """Return the process-local evaluator; no model approximation or JIT."""
        from .execution import PreparedEvaluator
        if self._evaluator is None or self._evaluator.closed:
            self._evaluator = PreparedEvaluator(self)
        return self._evaluator

    @property
    def renderer(self):
        return self._renderer if self._evaluator is None else self._evaluator.renderer

    @renderer.setter
    def renderer(self, value):
        self._renderer = value
        if self._evaluator is not None:
            self._evaluator.renderer = value
            self._evaluator._basis_cache.clear()
            self._evaluator._compressed = None
            self._evaluator._likelihood_engines.clear()

    @property
    def metrics(self): return self.prepare().metrics

    @property
    def compressed(self): return self.prepare().compressed

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_evaluator'] = None
        return state

    def _ensure_renderer(self): return self.prepare()._ensure_renderer()
    def morphology(self, component, values): return self.prepare().morphology(component, values)
    def spatial_key(self, component, observation, values): return self.prepare().spatial_key(component, observation, values)
    def basis_observation(self, observation, values): return self.prepare().basis_observation(observation, values)
    def basis(self, values): return self.prepare().basis(values)
    def component_fluxes(self, values): return self.prepare().component_fluxes(values)
    def predict(self, values=None, *, basis=None): return self.prepare().predict(values, basis=basis)
    def residual(self, values, *, basis=None, include_priors=True):
        return self.prepare().residual(values, basis=basis, include_priors=include_priors)
    def pixel_loglikelihood(self, values): return self.prepare().pixel_loglikelihood(values)
    def gaussian_data(self, observation): return self.prepare().gaussian_data(observation)
    def loglikelihood(self, values=None, *, mode='auto'): return self.prepare().loglikelihood(values, mode=mode)
    def _eligible_linear(self): return self.prepare()._eligible_linear()
    def project(self, values, eligible, basis): return self.prepare().project(values, eligible, basis)
    def clear_caches(self):
        if self._evaluator is not None: self._evaluator.clear_caches()

    def fit(self, *, objective='map', **options):
        from .optimize import fit_model
        return fit_model(self, objective=objective, **options)

    def sample_polychord(self, directory, *, nuclear_mode='sample', likelihood='auto', **options):
        """Sample all nuisance parameters; improper-flat marginalization is opt-in."""
        from .inference import PolyChordPosterior, PolyChordSampler
        if nuclear_mode=='sample':sampler=PolyChordSampler(self,likelihood=likelihood)
        elif nuclear_mode=='flat_marginal':sampler=PolyChordPosterior(self,likelihood=likelihood)
        else:raise ValueError('Use sample or flat_marginal nuclear mode')
        return sampler.run(directory, **options)

    def close(self):
        if self._evaluator is not None:
            self._evaluator.close()
            self._evaluator = None
        else:
            for component in self.components:
                if hasattr(component.sed, 'close'): component.sed.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def _workspace_property(name):
    """Temporary private compatibility views, never a second execution state."""
    return property(lambda model: getattr(model.prepare(), name),
                    lambda model, value: setattr(model.prepare(), name, value))


for _name in ['_basis_cache', '_compressed', '_gaussian_cache', '_gaussian_tokens',
              '_spectral_engine', '_likelihood_engines']:
    setattr(Scene, _name, _workspace_property(_name))
