"""Independent full component/aggregate pixel reference implementation."""
import time
import numpy as np
from .gaussian import FixedGaussianData,GaussianAccumulator


class FullPixelLikelihood:
    def __init__(self,scene,*,retain_templates=False,rendering='components'):
        self.scene=scene.prepare();self.retain_templates=bool(retain_templates)
        if rendering not in {'components','aggregate'}:raise ValueError('Unknown full rendering strategy')
        self.rendering=rendering;self.metrics=dict(likelihood_calls=0,render_calls=0,seconds=0.)

    def _render_observation(self,o,values,fluxes,retain=()):
        e=self.scene;e._ensure_renderer();amplitudes,sky,calibration=e.image_coefficients(o,values,fluxes)
        shift=e.registration(o,values);morphologies=[e.morphology(c,values) for c in e.components]
        model=np.full(o.data.shape,sky,dtype=float)
        aggregate=getattr(e.renderer,'render_total',None)
        if self.rendering=='aggregate' and aggregate is not None:
            rendered=aggregate(e.components,morphologies,amplitudes,o,shift,retain)
            model+=rendered.image;self.metrics['render_calls']+=rendered.draw_count;e.metrics['render_calls']+=rendered.draw_count
            return model,{n:col*calibration for n,col in rendered.columns.items()}
        batched=getattr(e.renderer,'render_batch',None)
        templates=(batched(e.components,morphologies,o,shift) if batched else
                   (e.renderer.render(c,m,o,shift) for c,m in zip(e.components,morphologies)))
        columns={}
        for c,image,amplitude in zip(e.components,templates,amplitudes):
            image=np.asarray(image,dtype=float);model+=amplitude*image
            self.metrics['render_calls']+=1;e.metrics['render_calls']+=1
            if c.name in retain:columns[c.name]=image*calibration
            if self.retain_templates:e.store_template(c,o,values,image)
        return model,columns

    def evaluate(self,values,linear=()):
        started=time.perf_counter();e=self.scene
        if not e.components:raise ValueError('Scene has no components')
        values=e.normalized_values(values)
        if linear:
            values=dict(values)
            for name in linear:values[name]=0.
        plans=e.nuclear_plan(linear);fluxes=e.component_fluxes(values);accumulator=GaussianAccumulator()
        for o in e.observations:
            parameter=plans.get(o.band);retain=() if parameter is None else (parameter.split('.')[0],)
            model,columns=self._render_observation(o,values,fluxes,retain)
            fixed=e.gaussian_data(o);residual=e.whitened_residual(o,model)
            column=None if parameter is None else e.whiten(o,columns[retain[0]][o.mask])
            accumulator.add(residual,fixed.whitening.log_normalization,parameter=parameter,column=column)
        self.metrics['likelihood_calls']+=1;self.metrics['seconds']+=time.perf_counter()-started
        return accumulator.finish()

    def loglikelihood(self,values):return self.evaluate(values).loglikelihood
    def marginalized_loglikelihood(self,values,linear):
        result=self.evaluate(values,linear)
        return result.loglikelihood,result.conditionals


# Useful legacy import; there is only one policy implementation.
from .policy import ExecutionPolicy as AutoGaussianLikelihood
