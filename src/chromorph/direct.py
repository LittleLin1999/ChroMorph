"""Exact full-pixel Gaussian evaluation, independent of spatial compression."""
from dataclasses import dataclass
import time
import numpy as np


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


class FullPixelLikelihood:
    """Render a complete scene at each call; only fixed data/PSFs are reused.

    `retain_templates` optionally seeds the bounded component cache as a
    byproduct, for a later flux/SED-only proposal. It never skips rendering.
    No Gram matrix or spatial-template cache is required for this path.
    """
    def __init__(self, scene, *, retain_templates=False):
        self.scene = scene
        self.retain_templates = bool(retain_templates)
        self.metrics = dict(likelihood_calls=0, render_calls=0, seconds=0.)

    def _render_observation(self,o,values,fluxes,retain=()):
        scene=self.scene;scene._ensure_renderer()
        shift=np.array([values[f'observation.{o.name}.{axis}'] for axis in ['dx','dy']])
        calibration=np.exp(values[f'observation.{o.name}.log_calibration'])/o.ujy_per_rate
        model=np.full(o.data.shape,values[f'observation.{o.name}.background'],dtype=float)
        morphologies=[scene.morphology(c,values) for c in scene.components]
        batched=getattr(scene.renderer,'render_batch',None)
        templates=(batched(scene.components,morphologies,o,shift) if batched else
            (scene.renderer.render(c,m,o,shift) for c,m in zip(scene.components,morphologies)))
        columns={}
        for c,image in zip(scene.components,templates):
            flux=fluxes[c.name][o.band]
            if not np.isfinite(flux) or flux<0:raise ValueError('Invalid component flux')
            image=np.asarray(image,dtype=float);model+=calibration*flux*image
            self.metrics['render_calls']+=1;scene.metrics['render_calls']+=1
            if c.name in retain:columns[c.name]=image*calibration
            if self.retain_templates:
                image.setflags(write=False)
                scene._basis_cache[(o.name,scene.spatial_key(c,o,values))]=image
        return model,columns

    def loglikelihood(self, values):
        start=time.perf_counter();scene=self.scene
        if not scene.components:raise ValueError('Scene has no components')
        fluxes=scene.component_fluxes(values)
        total=0.
        for o in scene.observations:
            fixed=scene.gaussian_data(o);model,_=self._render_observation(o,values,fluxes)
            residual=fixed.whitening.whiten((o.data-model)[o.mask])
            total+=fixed.whitening.log_normalization-.5*float(residual@residual)
        self.metrics['likelihood_calls']+=1;self.metrics['seconds']+=time.perf_counter()-start
        if not np.isfinite(total):raise ValueError('Nonfinite Gaussian likelihood')
        return float(total)

    def marginalized_loglikelihood(self,values,linear):
        """Exact independent F>=0 integration directly in pixel space.

        This path never constructs U^T C^-1 U. One independent nuclear
        amplitude per band is supported; all other parameters remain explicit.
        The flat unbounded nuclear measure does not define absolute evidence.
        """
        from .inference import marginal_linear_flux
        start=time.perf_counter();scene=self.scene;values=dict(values)
        for name in linear:values[name]=0.
        by_band={n.split('.')[-1]:n for n in linear}
        if len(by_band)!=len(linear):raise ValueError('Sample multiple overlapping nuclear amplitudes explicitly with proper priors')
        fluxes=scene.component_fluxes(values)
        bands={};total=0.;conditional={}
        for o in scene.observations:
            fixed=scene.gaussian_data(o);parameter=by_band.get(o.band)
            component=None if parameter is None else parameter.split('.')[0]
            model,columns=self._render_observation(o,values,fluxes,() if component is None else (component,))
            residual=fixed.whitening.whiten((o.data-model)[o.mask])
            total+=fixed.whitening.log_normalization
            if parameter is None:total-=.5*float(residual@residual)
            else:
                residuals,templates=bands.setdefault(o.band,([],[]))
                residuals.append(residual);templates.append(fixed.whitening.whiten(columns[component][o.mask]))
        for band,(residuals,templates) in bands.items():
            score,mean,sigma=marginal_linear_flux(np.concatenate(residuals),np.concatenate(templates))
            total+=score;conditional[by_band[band]]=(mean,sigma)
        self.metrics['likelihood_calls']+=1;self.metrics['seconds']+=time.perf_counter()-start
        return float(total),conditional


class AutoGaussianLikelihood:
    """Exact dispatch by changed spatial state, with explicit measured policy.

    No proposal sparsity is assumed. A proposal may change every component.
    `morphology_path` should be selected by an end-to-end workload benchmark;
    full rendering is the conservative default. Fast proposals reuse statistics.
    """
    def __init__(self,scene,*,morphology_path='pixel'):
        if morphology_path not in {'pixel','compressed'}:raise ValueError('Unknown morphology path')
        self.scene=scene;self.morphology_path=morphology_path;self._state=None
        self.full=FullPixelLikelihood(scene,retain_templates=True)
        self.metrics=dict(pixel_calls=0,compressed_calls=0,all_components_changed=0)

    def loglikelihood(self,values):
        state=tuple((o.likelihood_token(),tuple(self.scene.spatial_key(c,o,values) for c in self.scene.components))
                    for o in self.scene.observations)
        changed=state!=self._state
        path=self.morphology_path if changed else 'compressed'
        if self._state is not None and changed:
            if all(any(state[b][1][j]!=self._state[b][1][j] for b in range(len(state))) for j in range(len(self.scene.components))):
                self.metrics['all_components_changed']+=1
        score=(self.full.loglikelihood(values) if path=='pixel' else self.scene.compressed.loglikelihood(values))
        # Freezing data can update tokens on the first call. Record final state.
        self._state=tuple((o.likelihood_token(),tuple(self.scene.spatial_key(c,o,values) for c in self.scene.components))
                          for o in self.scene.observations)
        self.metrics[path+'_calls']+=1
        return score
