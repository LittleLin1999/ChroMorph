"""Stable sufficient statistics for fixed Gaussian multi-image likelihoods."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import qr
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
        return self.R.T@self.R

    @property
    def h(self):
        return self.R.T@self.z

    @property
    def c(self):
        return float(self.z@self.z+self.perpendicular_chi2)

    @property
    def nbytes(self):
        return self.R.nbytes+self.z.nbytes+self.singular_values.nbytes+128

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


class CompressedGaussianLikelihood:
    def __init__(self, scene):
        from .cache import ExactLRU
        self.scene = scene
        self._observations = {}
        self._statistics = ExactLRU(8*1024**2)
        self.metrics = dict(statistics_builds=0, likelihood_calls=0, statistics_seconds=0.)

    def _data_token(self, observation):
        token = observation.likelihood_token()
        if self._observations.get(observation.name) != token:
            observation.freeze()
            token = observation.likelihood_token()
            self._observations[observation.name] = token
        return token

    def statistics(self, values):
        scene = self.scene
        scene._ensure_renderer()
        output = {}
        for observation in scene.observations:
            data_key = self._data_token(observation)
            key = (observation.name, data_key, tuple(scene.spatial_key(c, observation, values) for c in scene.components))
            stats = self._statistics.get(key)
            if stats is None:
                start = time.perf_counter()
                basis = scene.basis_observation(observation, values)
                templates = np.stack([basis[c.name][observation.mask] for c in scene.components]
                                     +[np.ones(observation.mask.sum())], axis=1)
                stats = GaussianStatistics.build(templates, observation.data[observation.mask],
                                                 whitening_for(observation))
                self._statistics[key] = stats
                self.metrics['statistics_builds'] += 1
                self.metrics['statistics_seconds'] += time.perf_counter()-start
            output[observation.name] = stats
        return output

    def coefficients(self, values):
        scene = self.scene
        fluxes = {c.name: c.sed.fluxes(values, c.name, scene.observations, scene.context)
                  for c in scene.components}
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
        return float(sum(s.log_normalization for s in statistics.values())-.5*self.chi2(values))

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
