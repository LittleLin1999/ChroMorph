"""Stable sufficient statistics for fixed Gaussian multi-image likelihoods."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import qr, qr_update
from .noise import whitening_for


@dataclass
class GaussianStatistics:
    R: np.ndarray
    z: np.ndarray
    perpendicular_chi2: float
    log_normalization: float
    n_pixels: int
    singular_values: np.ndarray
    rank: int
    gram: object = None
    rhs: object = None
    constant: object = None

    @classmethod
    def build(cls, templates, data, whitening):
        design, target = whitening.whiten(templates), whitening.whiten(data)
        q, r = qr(design, mode='economic', check_finite=False)
        z = q.T@target
        perpendicular = target-q@z
        singular = np.linalg.svd(r, compute_uv=False)
        cutoff = max(singular[0]*max(design.shape)*np.finfo(float).eps, 1e-15) if len(singular) else 1e-15
        return cls(r, z, float(perpendicular@perpendicular), whitening.log_normalization,
                   len(target), singular, int(np.count_nonzero(singular > cutoff)))

    @property
    def G(self):
        return self.R.T@self.R if self.gram is None else self.gram

    @property
    def h(self):
        return self.R.T@self.z if self.rhs is None else self.rhs

    @property
    def c(self):
        return float(self.z@self.z+self.perpendicular_chi2) if self.constant is None else self.constant

    @property
    def nbytes(self):
        return (self.R.nbytes+self.z.nbytes+self.singular_values.nbytes+128
                + (0 if self.gram is None else self.gram.nbytes+self.rhs.nbytes))

    @property
    def condition_number(self):
        if self.rank < self.R.shape[1] or not len(self.singular_values):
            return np.inf
        return float(self.singular_values[0]/self.singular_values[-1])

    def residual(self, amplitudes):
        return self.R@amplitudes-self.z

    def chi2(self, amplitudes):
        # Algebraically c-2*a.h+a.G.a, but stable even for bright sources.
        residual = self.residual(amplitudes)
        return float(self.perpendicular_chi2+residual@residual)


@dataclass
class SpatialStatisticsState:
    data_key: tuple
    spatial_keys: tuple
    design: np.ndarray
    q: np.ndarray
    statistics: GaussianStatistics

    @property
    def nbytes(self):
        return self.design.nbytes+self.q.nbytes+self.statistics.nbytes+256


class CompressedGaussianLikelihood:
    def __init__(self, scene, *, incremental=True, state_cache_bytes=64*1024**2):
        from .cache import ExactLRU
        self.scene = scene
        self.incremental = bool(incremental)
        self._statistics = ExactLRU(8*1024**2)
        self._states = ExactLRU(state_cache_bytes)
        self.metrics = dict(statistics_builds=0, likelihood_calls=0, statistics_seconds=0.,
            full_factorizations=0, rank_updates=0, gram_columns_updated=0, whitened_columns=0)

    def statistics(self, values):
        scene = self.scene
        scene._ensure_renderer()
        output = {}
        for observation in scene.observations:
            fixed = scene.gaussian_data(observation)
            data_key = fixed.token
            spatial = tuple(scene.spatial_key(c, observation, values) for c in scene.components)
            key = (observation.name, data_key, spatial)
            stats = self._statistics.get(key)
            if stats is None:
                start = time.perf_counter()
                state = self._states.get(observation.name)
                changed = ([] if state is None or state.data_key != data_key or len(state.spatial_keys)!=len(spatial)
                           else [j for j, (a,b) in enumerate(zip(state.spatial_keys,spatial)) if a!=b])
                can_update = (self.incremental and state is not None and state.data_key==data_key
                              and len(state.spatial_keys)==len(spatial) and changed
                              and len(changed)<=max(1,len(spatial)//2)
                              and state.statistics.R.shape[0]==state.statistics.R.shape[1])
                basis = scene.basis_observation(observation, values)
                if can_update:
                    design = state.design.copy();q = state.q; r = state.statistics.R
                    gram = state.statistics.G.copy();rhs = state.statistics.h.copy()
                    for j in changed:
                        column = fixed.whitening.whiten(basis[scene.components[j].name][observation.mask])
                        if not np.isfinite(column).all():raise ValueError('Nonfinite spatial template')
                        direction = np.zeros(design.shape[1]);direction[j]=1.
                        q,r = qr_update(q,r,column-design[:,j],direction,check_finite=False)
                        design[:,j] = column
                        rhs[j] = column@fixed.target
                    for j in changed:
                        gram[:,j] = design.T@design[:,j];gram[j,:] = gram[:,j]
                    self.metrics['rank_updates'] += len(changed)
                    self.metrics['gram_columns_updated'] += len(changed)
                    self.metrics['whitened_columns'] += len(changed)
                else:
                    templates = np.stack([basis[c.name][observation.mask] for c in scene.components]
                                         +[np.ones(observation.mask.sum())],axis=1)
                    design = fixed.whitening.whiten(templates)
                    if not np.isfinite(design).all():raise ValueError('Nonfinite spatial templates')
                    q,r = qr(design,mode='economic',check_finite=False)
                    gram = design.T@design;rhs = design.T@fixed.target
                    self.metrics['full_factorizations'] += 1
                    self.metrics['whitened_columns'] += design.shape[1]
                z = q.T@fixed.target
                perpendicular = fixed.target-q@z
                singular = np.linalg.svd(r,compute_uv=False)
                cutoff = max(singular[0]*max(design.shape)*np.finfo(float).eps,1e-15) if len(singular) else 1e-15
                stats = GaussianStatistics(r,z,float(perpendicular@perpendicular),fixed.whitening.log_normalization,
                    len(fixed.target),singular,int(np.count_nonzero(singular>cutoff)),gram,rhs,float(fixed.target@fixed.target))
                self._states[observation.name] = SpatialStatisticsState(data_key,spatial,design,q,stats)
                self._statistics[key] = stats
                self.metrics['statistics_builds'] += 1
                self.metrics['statistics_seconds'] += time.perf_counter()-start
            output[observation.name] = stats
        return output

    def coefficients(self, values):
        scene = self.scene
        fluxes = scene.component_fluxes(values)
        result = {}
        for o in scene.observations:
            calibration = np.exp(values[f'observation.{o.name}.log_calibration'])/o.ujy_per_rate
            amplitude = np.array([fluxes[c.name][o.band]*calibration for c in scene.components]
                                 +[values[f'observation.{o.name}.background']])
            if not np.isfinite(amplitude).all() or np.any(amplitude[:-1] < 0):
                raise ValueError('Nonfinite or negative component flux')
            result[o.name] = amplitude
        return result

    def chi2(self, values, *, include_priors=False):
        statistics = self.statistics(values)
        coefficients = self.coefficients(values)
        result = sum(statistics[name].chi2(coefficients[name]) for name in statistics)
        if include_priors:
            prior = self.scene.registry.prior_residuals(values)
            result += float(prior@prior)
        self.metrics['likelihood_calls'] += 1
        return float(result)

    def loglikelihood(self, values):
        statistics = self.statistics(values)
        coefficients = self.coefficients(values)
        self.metrics['likelihood_calls'] += 1
        return float(sum(s.log_normalization-.5*s.chi2(coefficients[name]) for name,s in statistics.items()))

    def residual(self, values, *, include_priors=True):
        statistics = self.statistics(values)
        coefficients = self.coefficients(values)
        residuals = [s.residual(coefficients[name]) for name, s in statistics.items()]
        # Keep the constant orthogonal component so residual@residual is the
        # full objective even though it contributes no amplitude gradient.
        residuals.extend(np.array([np.sqrt(s.perpendicular_chi2)]) for s in statistics.values())
        if include_priors:
            residuals.append(self.scene.registry.prior_residuals(values))
        return np.concatenate(residuals)
