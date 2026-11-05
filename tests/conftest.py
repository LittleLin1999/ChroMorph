"""Analytic test fixtures; explicitly not a substitute for GalSim validation."""
from __future__ import annotations
import numpy as np
import pytest
from scipy.special import erf
from chromorph import Observation, EmpiricalPSF, Sersic, PointSource, Parameter, Scene, FreeBandSED


class GaussianTestRenderer:
    """Exactly integrated circular n=0.5 Gaussian; optimizer unit-test seam."""
    def render(self, component, morphology, observation, shift=(0, 0)):
        xy = observation.matrix @ np.array([morphology['x'], morphology['y']])+observation.offset+shift
        sigma = float(observation.psf.provenance.get('gaussian_sigma', 1.))
        if isinstance(component, Sersic):
            if morphology['n'] != .5 or morphology['q'] != 1:
                raise ValueError('Test renderer only supports circular n=0.5 profiles')
            sigma = np.hypot(sigma, morphology['re']/np.sqrt(2*np.log(2)))
        yy, xx = np.indices(observation.data.shape)
        def integral(grid, center):
            return (erf((grid-center+.5)/(np.sqrt(2)*sigma))-erf((grid-center-.5)/(np.sqrt(2)*sigma)))/2
        return integral(xx, xy[0])*integral(yy, xy[1])


def observations(bands=('blue', 'red'), size=25, noise=.002):
    yy, xx = np.indices((15, 15)); stamp = np.exp(-((xx-7)**2+(yy-7)**2)/2)
    stamp /= stamp.sum()
    return [Observation(b, b, np.zeros((size, size)), np.full((size, size), noise),
                        EmpiricalPSF(stamp, .04, provenance={'gaussian_sigma': 1.}), .04, 1.,
                        pivot_angstrom=4000+3000*i) for i, b in enumerate(bands)]


def clump(name, fluxes=None, x=8., y=12., re=1.5, moving=False, sed=None):
    return Sersic(name, sed or FreeBandSED(fluxes), parameters={
        'x': Parameter(x, x-3, x+3, vary=moving),
        'y': Parameter(y, y-3, y+3, vary=moving),
        're': Parameter(re, .3, 4., vary=moving),
        'n': Parameter(.5, .3, 1., vary=False),
        'q': Parameter(1., .5, 1., vary=False),
        'pa': Parameter(0., -180, 180, vary=False)})


@pytest.fixture
def renderer():
    return GaussianTestRenderer()
