"""Sampler-independent Gaussian data and exact amplitude integration."""
from dataclasses import dataclass
import numpy as np
from scipy.special import log_ndtr


@dataclass
class FixedGaussianData:
    token: tuple
    whitening: object
    target: np.ndarray
    masked_data: object = None
    included: object = None
    noise_token: object = None

    @property
    def nbytes(self):
        factor = getattr(self.whitening, 'factor', np.empty(0))
        sigma = getattr(self.whitening, 'sigma', np.empty(0))
        return (self.target.nbytes+factor.nbytes+sigma.nbytes+256
                +(0 if self.masked_data is None else self.masked_data.nbytes)
                +(0 if self.included is None else self.included.nbytes))


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


@dataclass(frozen=True)
class GaussianEvaluation:
    loglikelihood: float
    conditionals: dict


class GaussianAccumulator:
    """Block-independent observations, with shared per-filter nuclear fluxes."""
    def __init__(self):self.total=0.;self.bands={}

    def add(self,residual,normalization,*,perpendicular_chi2=0.,parameter=None,column=None):
        self.total+=normalization-.5*perpendicular_chi2
        if parameter is None:self.total-=.5*float(residual@residual)
        else:
            residuals,columns=self.bands.setdefault(parameter,([],[]))
            residuals.append(residual);columns.append(column)

    def finish(self):
        conditional={}
        for name,(residuals,columns) in self.bands.items():
            value,mean,sigma=marginal_linear_flux(residuals[0] if len(residuals)==1 else np.concatenate(residuals),
                                                   columns[0] if len(columns)==1 else np.concatenate(columns))
            self.total+=value;conditional[name]=(mean,sigma)
        if not np.isfinite(self.total):raise ValueError('Nonfinite Gaussian likelihood')
        return GaussianEvaluation(float(self.total),conditional)
