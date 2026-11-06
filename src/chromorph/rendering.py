"""GalSim spatial rendering, with independent component and aggregate draws."""
from dataclasses import dataclass
import time
import numpy as np
from .spatial import PointSource
from .cache import ExactLRU


@dataclass(frozen=True)
class RenderedTotal:
    image: np.ndarray
    columns: dict
    draw_count: int


class GalSimRenderer:
    def __init__(self, interpolation_order=5, maximum_fft_size=16384,
                 folding_threshold=1e-4, maxk_threshold=1e-5):
        try:
            import galsim
        except ImportError as exc:
            raise ImportError('GalSim is required for spatial rendering; install chromorph dependencies') from exc
        self.gs = galsim
        self.interpolation_order = interpolation_order
        self.configuration = dict(interpolation_order=interpolation_order,
            maximum_fft_size=maximum_fft_size, folding_threshold=folding_threshold, maxk_threshold=maxk_threshold)
        self.gsparams = galsim.GSParams(maximum_fft_size=maximum_fft_size,
            folding_threshold=folding_threshold, maxk_threshold=maxk_threshold)
        self._psfs = ExactLRU(64*1024**2, max_entries=32)
        self.metrics = dict(intrinsic_constructions=0, intrinsic_seconds=0., transform_seconds=0.,
            psf_preparations=0, psf_preparation_seconds=0., position_calls=0, position_seconds=0.,component_draws=0, total_draws=0,
            draw_attempts=0, draw_seconds=0., convolution_builds=0, convolution_setup_seconds=0.,
            aggregate_fallbacks=0, component_images_materialized=0)

    def __reduce__(self): return (_rebuild_renderer, (self.configuration,))

    def cache_token(self):
        return ('galsim', repr(self.gsparams), self.interpolation_order)

    def _psf(self, observation):
        p = observation.psf
        key = (p.cache_token(), observation.pixel_scale, self.cache_token())
        obj = self._psfs.get(key)
        if obj is None:
            if not np.isclose(p.pixel_scale, observation.pixel_scale, rtol=.005):
                raise ValueError('Science and empirical PSF pixel scales disagree')
            started = time.perf_counter()
            image = self.gs.ImageD(np.array(p.image, copy=True), scale=1.)
            obj = self.gs.InterpolatedImage(image, normalization='flux', flux=1.,
                x_interpolant=self.gs.Lanczos(self.interpolation_order), gsparams=self.gsparams)
            geometric = (np.array(p.image.shape[::-1])-1)/2
            dx,dy = geometric-np.array(p.center)
            obj = obj.shift(float(dx), float(dy))
            self._psfs[key] = obj
            self.metrics['psf_preparations'] += 1
            self.metrics['psf_preparation_seconds'] += time.perf_counter()-started
        return obj

    def _intrinsic(self, morphology, observation):
        started = time.perf_counter()
        # Preserve the existing validated Gaussian-index limiting workaround.
        index = .5 if abs(morphology['n']-.5)<1e-10 else morphology['n']
        intrinsic = self.gs.Sersic(n=index,
            half_light_radius=morphology['re']*np.sqrt(morphology['q']), flux=1., gsparams=self.gsparams)
        self.metrics['intrinsic_constructions'] += 1
        self.metrics['intrinsic_seconds'] += time.perf_counter()-started
        started = time.perf_counter()
        intrinsic = intrinsic.shear(q=morphology['q'], beta=morphology['pa']*self.gs.degrees)
        intrinsic = intrinsic.transform(*observation.matrix.ravel()).withFlux(1.)
        self.metrics['transform_seconds'] += time.perf_counter()-started
        return intrinsic

    def _position(self, morphology, observation, shift):
        started=time.perf_counter()
        xy=observation.matrix@np.array([morphology['x'],morphology['y']])+observation.offset+shift
        answer=xy-(np.array(observation.data.shape[::-1])-1)/2
        self.metrics['position_calls']+=1;self.metrics['position_seconds']+=time.perf_counter()-started
        return answer

    def _convolve(self, profile, psf):
        started=time.perf_counter()
        obj=self.gs.Convolve([profile,psf],gsparams=self.gsparams)
        self.metrics['convolution_builds']+=1
        self.metrics['convolution_setup_seconds']+=time.perf_counter()-started
        return obj

    def _draw(self, obj, observation, offset=(0.,0.)):
        ny,nx=observation.data.shape
        started=time.perf_counter();self.metrics['draw_attempts']+=1
        try:
            # The empirical PSF already contains the detector response.
            image=obj.drawImage(nx=nx,ny=ny,scale=1.,
                offset=self.gs.PositionD(float(offset[0]),float(offset[1])),
                method='no_pixel' if observation.psf.pixel_integrated else 'auto',
                use_true_center=True,dtype=np.float64)
            return np.asarray(image.array,dtype=float)
        finally:self.metrics['draw_seconds']+=time.perf_counter()-started

    def render(self, component, morphology, observation, shift=(0,0)):
        psf=self._psf(observation)
        obj=psf if isinstance(component,PointSource) else self._convolve(self._intrinsic(morphology,observation),psf)
        image=self._draw(obj,observation,self._position(morphology,observation,shift))
        self.metrics['component_draws']+=1
        self.metrics['component_images_materialized']+=1
        return image

    def _component_total(self, components, morphologies, amplitudes, observation, shift, retain):
        image=np.zeros(observation.data.shape);columns={};draws=0
        for c,m,a in zip(components,morphologies,amplitudes):
            if a or c.name in retain:
                unit=self.render(c,m,observation,shift);draws+=1;image+=a*unit
                if c.name in retain:columns[c.name]=unit
        return RenderedTotal(image,columns,draws)

    def render_total(self, components, morphologies, amplitudes, observation, shift=(0,0), retain=()):
        """Convolve the sum of intrinsic extended profiles with the common PSF.

        Native shifted point responses are drawn separately: forcing a
        real-space empirical interpolant into a joint FFT changes its subpixel
        response. Finite-stamp flux losses and the supplied PSF are preserved.
        """
        amplitudes=np.asarray(amplitudes,dtype=float)
        if len(components)!=len(morphologies) or amplitudes.shape!=(len(components),):
            raise ValueError('Component, morphology and amplitude lengths differ')
        if not np.isfinite(amplitudes).all() or np.any(amplitudes<0):raise ValueError('Invalid component amplitude')
        if not set(retain)<=set(c.name for c in components):raise ValueError('Unknown retained component')
        if not amplitudes.any() and not retain:return RenderedTotal(np.zeros(observation.data.shape),{},0)
        center=(np.array(observation.data.shape[::-1])-1)/2
        for c,m,a in zip(components,morphologies,amplitudes):
            if getattr(c,'psf',None) is not None and c.psf is not observation.psf:
                raise ValueError('Aggregate rendering requires the declared common observation PSF')
            xy=self._position(m,observation,shift)+center
            if a and not isinstance(c,PointSource) and (np.any(xy<0) or np.any(xy>2*center)):
                # Shifted off-field profiles can alias on a joint Fourier grid.
                # Use the original draw convention, without moving/truncating them.
                self.metrics['aggregate_fallbacks']+=1
                return self._component_total(components,morphologies,amplitudes,observation,shift,retain)
        psf=self._psf(observation);extended=[]
        for c,m,a in zip(components,morphologies,amplitudes):
            if a and not isinstance(c,PointSource):
                dx,dy=map(float,self._position(m,observation,shift))
                extended.append(self._intrinsic(m,observation).withFlux(float(a)).shift(dx,dy))
        columns={};draws=0
        if extended:
            intrinsic=extended[0] if len(extended)==1 else self.gs.Add(extended,gsparams=self.gsparams)
            try:
                image=self._draw(self._convolve(intrinsic,psf),observation)
                draws+=1;self.metrics['total_draws']+=1
            except self.gs.GalSimFFTSizeError:
                self.metrics['aggregate_fallbacks']+=1
                return self._component_total(components,morphologies,amplitudes,observation,shift,retain)
        else:image=np.zeros(observation.data.shape)
        for c,m,a in zip(components,morphologies,amplitudes):
            if isinstance(c,PointSource) and (a or c.name in retain):
                unit=self.render(c,m,observation,shift);image+=a*unit;draws+=1
                if c.name in retain:columns[c.name]=unit
            elif c.name in retain:
                columns[c.name]=self.render(c,m,observation,shift);draws+=1
        return RenderedTotal(image,columns,draws)

    def validate_psf_sampling(self, observation):
        from .sed import FreeBandSED
        from dataclasses import replace
        psf=observation.psf
        obs=replace(observation,data=np.zeros(psf.image.shape),sigma=np.ones(psf.image.shape),
                    mask=None,matrix=np.eye(2),offset=np.zeros(2))
        point=PointSource('probe',FreeBandSED());pars=dict(x=psf.center[0],y=psf.center[1])
        rendered=self.render(point,pars,obs);expected=psf.image/psf.image.sum()
        relative=float(np.linalg.norm(rendered-expected)/np.linalg.norm(expected))
        shifted=self.render(point,pars,obs,shift=(.5,.5))
        return dict(same_grid_relative_l2=relative,same_grid_sum=float(rendered.sum()),
            half_pixel_sum=float(shifted.sum()),method='no_pixel' if psf.pixel_integrated else 'auto',pass_same_grid=relative<.01)


def _rebuild_renderer(configuration): return GalSimRenderer(**configuration)
