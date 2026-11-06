"""Sampler-independent Gaussian data and exact amplitude integration."""
from dataclasses import dataclass
import numpy as np
from scipy.special import log_ndtr


@dataclass
class FixedGaussianData:
    token: tuple
    whitening: object
    target: np.ndarray

    @property
    def nbytes(self):
        factor = getattr(self.whitening, 'factor', np.empty(0))
        sigma = getattr(self.whitening, 'sigma', np.empty(0))
        return self.target.nbytes+factor.nbytes+sigma.nbytes+256


def marginal_linear_flux(residual, column):
    """Log integral over F>=0 and the underlying Gaussian mean/sigma.

    residual and column are already divided by the observational sigma.
    The reference integration measure is dF with F expressed in microJy.
    """
    precision = float(column @ column)
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError('Unobserved nuclear flux: flat-prior posterior is improper')
    sigma = precision**-.5
    mean = float(column @ residual)/precision
    # Subtract the unconstrained optimum explicitly to avoid catastrophic
    # cancellation between two large quadratic forms.
    remaining = residual-column*mean
    log_integral = (-.5*float(remaining @ remaining) + .5*np.log(2*np.pi)
                    + np.log(sigma) + log_ndtr(mean/sigma))
    return float(log_integral), mean, sigma
