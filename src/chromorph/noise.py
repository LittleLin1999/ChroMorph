"""Fixed Gaussian whitening and normalization, without covariance inversion."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky, solve_triangular


class DiagonalWhitening:
    def __init__(self, sigma):
        self.sigma = np.array(sigma, dtype=float, copy=True)
        if self.sigma.ndim != 1 or not np.isfinite(self.sigma).all() or np.any(self.sigma <= 0):
            raise ValueError('Fixed uncertainties must be positive and finite')
        self.n = len(self.sigma)
        self.logdet_covariance = float(2*np.log(self.sigma).sum())

    def whiten(self, value):
        value = np.asarray(value)
        return value/self.sigma if value.ndim == 1 else value/self.sigma[:, None]

    @property
    def log_normalization(self):
        return -.5*(self.n*np.log(2*np.pi)+self.logdet_covariance)


class CholeskyWhitening:
    def __init__(self, covariance):
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.isfinite(covariance).all():
            raise ValueError('Covariance must be a finite square matrix')
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
            raise ValueError('Covariance must be symmetric')
        self.factor = cholesky(covariance, lower=True, check_finite=False)
        self.n = len(self.factor)
        self.logdet_covariance = float(2*np.log(np.diag(self.factor)).sum())

    def whiten(self, value):
        return solve_triangular(self.factor, value, lower=True, check_finite=False)

    @property
    def log_normalization(self):
        return -.5*(self.n*np.log(2*np.pi)+self.logdet_covariance)


class DenseCovariance:
    """Covariance in full flattened image order; masks select its submatrix."""
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float, copy=True)
        self.matrix.setflags(write=False)

    def for_mask(self, mask):
        if self.matrix.shape != (mask.size, mask.size):
            raise ValueError('Dense covariance must use the full flattened image order')
        included = np.flatnonzero(mask.ravel())
        return CholeskyWhitening(self.matrix[np.ix_(included, included)])


@dataclass(frozen=True)
class FixedWhiteningOperator:
    """User-supplied W satisfying W.T W=C^-1 on a declared fixed mask.

    `operator` is a scipy LinearOperator (or compatible matvec/matmat object).
    Its covariance log determinant must be supplied independently. No stochastic
    determinant estimate is silently substituted into an evidence calculation.
    """
    operator: object
    logdet_covariance: float
    mask_indices: tuple

    def for_mask(self, mask):
        if tuple(np.flatnonzero(mask.ravel())) != tuple(self.mask_indices):
            raise ValueError('Whitening operator mask changed; supply a matching operator')
        if self.operator.shape != (len(self.mask_indices), len(self.mask_indices)):
            raise ValueError('Whitening operator shape mismatch')
        if not np.isfinite(self.logdet_covariance):
            raise ValueError('Fixed covariance requires a finite log determinant')
        return self

    @property
    def n(self):
        return len(self.mask_indices)

    def whiten(self, value):
        return self.operator.matvec(value) if value.ndim == 1 else self.operator.matmat(value)

    @property
    def log_normalization(self):
        return -.5*(self.n*np.log(2*np.pi)+self.logdet_covariance)


def whitening_for(observation):
    covariance = observation.covariance
    if covariance is None:
        return DiagonalWhitening(observation.sigma[observation.mask])
    if isinstance(covariance, np.ndarray):
        covariance = DenseCovariance(covariance)
    if not hasattr(covariance, 'for_mask'):
        raise TypeError('Use DenseCovariance or FixedWhiteningOperator for fixed covariance')
    return covariance.for_mask(observation.mask)
