"""GalSim renderer for empirical PSFs that already contain detector pixels."""
import numpy as np
from .spatial import PointSource


class GalSimRenderer:
    def __init__(self, interpolation_order=5, maximum_fft_size=16384):
        try:
            import galsim
        except ImportError as exc:
            raise ImportError("GalSim is required for spatial rendering; install chromorph dependencies") from exc
        self.gs = galsim
        self.interpolation_order = interpolation_order
        self.gsparams = galsim.GSParams(maximum_fft_size=maximum_fft_size,
                                       folding_threshold=1e-4, maxk_threshold=1e-5)
        self._psfs = {}

    def _psf(self, observation):
        # Observation/PSF instances are kept alive by Scene; identity is stable.
        key = id(observation.psf)
        if key not in self._psfs:
            psf = observation.psf
            image = self.gs.ImageD(np.array(psf.image, copy=True), scale=1.0)
            obj = self.gs.InterpolatedImage(
                image, normalization="flux", flux=1.0,
                x_interpolant=self.gs.Lanczos(self.interpolation_order),
                gsparams=self.gsparams)
            geometric = (np.array(psf.image.shape[::-1])-1)/2
            dx, dy = geometric-np.array(psf.center)
            self._psfs[key] = obj.shift(float(dx), float(dy))
        return self._psfs[key]

    def render(self, component, morphology, observation, shift=(0, 0)):
        gs = self.gs
        psf = self._psf(observation)
        xy = observation.matrix @ np.array([morphology['x'], morphology['y']]) + observation.offset + shift
        if isinstance(component, PointSource):
            obj = psf
        else:
            # GalSim shear preserves circularized radius; API re is semimajor.
            intrinsic = gs.Sersic(n=morphology['n'],
                                  half_light_radius=morphology['re']*np.sqrt(morphology['q']),
                                  flux=1.0, gsparams=self.gsparams)
            intrinsic = intrinsic.shear(q=morphology['q'], beta=morphology['pa']*gs.degrees)
            a = observation.matrix
            intrinsic = intrinsic.transform(*a.ravel()).withFlux(1.0)
            obj = gs.Convolve([intrinsic, psf], gsparams=self.gsparams)
        ny, nx = observation.data.shape
        offset = gs.PositionD(float(xy[0]-(nx-1)/2), float(xy[1]-(ny-1)/2))
        # The drizzled empirical kernel ALREADY contains pixel integration.
        # Adding the usual drawImage pixel convolution would broaden it twice.
        method = "no_pixel" if observation.psf.pixel_integrated else "auto"
        return np.array(obj.drawImage(nx=nx, ny=ny, scale=1.0, offset=offset,
                                      method=method, use_true_center=True, dtype=np.float64).array, dtype=float)

    def validate_psf_sampling(self, observation):
        """Same-grid round trip and half-pixel sample diagnostics, no PSF rebuilding."""
        from .spatial import PointSource
        from .sed import FreeBandSED
        from dataclasses import replace
        psf = observation.psf
        obs = replace(observation, data=np.zeros(psf.image.shape), sigma=np.ones(psf.image.shape),
                      mask=None, matrix=np.eye(2), offset=np.zeros(2))
        point = PointSource("probe", FreeBandSED())
        pars = dict(x=psf.center[0], y=psf.center[1])
        rendered = self.render(point, pars, obs)
        expected = psf.image/psf.image.sum()
        relative = float(np.linalg.norm(rendered-expected)/np.linalg.norm(expected))
        shifted = self.render(point, pars, obs, shift=(.5, .5))
        return dict(same_grid_relative_l2=relative, same_grid_sum=float(rendered.sum()),
                    half_pixel_sum=float(shifted.sum()), method="no_pixel" if psf.pixel_integrated else "auto",
                    pass_same_grid=relative < .01)
