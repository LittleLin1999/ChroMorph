"""One lazy execution policy over the same forward model and Gaussian kernels."""
from dataclasses import dataclass
import numpy as np
import time
from .gaussian import GaussianAccumulator


@dataclass
class ImageAnchor:
    geometry: tuple
    amplitudes: np.ndarray
    sky: float
    image: np.ndarray
    excluded: tuple
    residual: np.ndarray
    data_token: tuple


class ExecutionPolicy:
    def __init__(self,scene,*,rendering='components',compression_after=None,morphology_path='pixel'):
        self.scene=scene.prepare()
        if rendering not in {'components','aggregate'}:raise ValueError('Unknown total rendering strategy')
        if morphology_path not in {'pixel','compressed'}:raise ValueError('Unknown morphology path')
        if compression_after is not None and (not isinstance(compression_after,int) or compression_after<1):
            raise ValueError('compression_after must be a positive measured break-even count, or None')
        self.rendering=rendering;self.compression_after=compression_after;self.morphology_path=morphology_path
        self.frames={};self._state=None;self._fixed_run=0
        self.metrics=dict(pixel_calls=0,compressed_calls=0,all_components_changed=0,anchor_updates=0,
            aggregate_observations=0,component_observations=0,anchor_bytes=0,
            residual_copies=0,residual_copy_bytes=0,residual_copy_seconds=0.)

    def clear(self):self.frames.clear();self._state=None;self._fixed_run=0;self.metrics['anchor_bytes']=0

    def _geometry(self,values):
        return tuple(tuple(self.scene.spatial_key(c,o,values) for c in self.scene.components)
                     for o in self.scene.observations)

    def _pixel_residual(self,o,values,fluxes,geometry,excluded,retain):
        e=self.scene;amplitudes,sky,calibration=e.image_coefficients(o,values,fluxes)
        fixed=e.gaussian_data(o);anchor=self.frames.get(o.name)
        same=anchor is not None and anchor.geometry==geometry and anchor.excluded==excluded
        if same:
            if anchor.data_token!=fixed.token:
                anchor.residual=e.whitened_residual(o,anchor.image)
                anchor.residual.setflags(write=False);anchor.data_token=fixed.token
            # Correct a fixed anchor; do not accumulate accepted/rejected-step
            # roundoff or assemble full images for amplitude-only requests.
            copy_started=time.perf_counter();residual=anchor.residual.copy()
            self.metrics['residual_copies']+=1;self.metrics['residual_copy_bytes']+=residual.nbytes
            self.metrics['residual_copy_seconds']+=time.perf_counter()-copy_started
            if sky!=anchor.sky:residual-=(sky-anchor.sky)*e.whitened_template(None,o,values)
            for i,delta in enumerate(amplitudes-anchor.amplitudes):
                if delta:residual-=delta*e.whitened_template(e.components[i],o,values)
            self.metrics['anchor_updates']+=1
        else:
            morphologies=[e.morphology(c,values) for c in e.components]
            aggregate=getattr(e.renderer,'render_total',None)
            if self.rendering=='aggregate' and aggregate is not None:
                total=aggregate(e.components,morphologies,amplitudes,o,e.registration(o,values),retain)
                model=total.image+sky;e.metrics['render_calls']+=total.draw_count
                for name,col in total.columns.items():e.store_template(e.component_named(name),o,values,col)
                self.metrics['aggregate_observations']+=1
            else:
                model=np.full(o.data.shape,sky,dtype=float)
                for c,a in zip(e.components,amplitudes):
                    if a or c.name in retain:
                        unit=e.unit_template(c,o,values)
                        if a:model+=a*unit
                self.metrics['component_observations']+=1
            model.setflags(write=False);residual=e.whitened_residual(o,model);residual.setflags(write=False)
            self.frames[o.name]=ImageAnchor(geometry,amplitudes.copy(),sky,model,excluded,residual,fixed.token)
            self.metrics['anchor_bytes']=sum(x.image.nbytes+x.amplitudes.nbytes+x.residual.nbytes for x in self.frames.values())
        columns={name:e.whitened_template(e.component_named(name),o,values)*calibration for name in retain}
        return residual,columns

    def evaluate(self,values,linear=()):
        e=self.scene;e._ensure_renderer();values=e.normalized_values(values);excluded=tuple(linear)
        if excluded:
            values=dict(values)
            for name in excluded:values[name]=0.
        geometry=self._geometry(values)
        same=self._state==geometry
        if self._state is not None and not same:
            if len(self._state)==len(geometry) and all(any(geometry[b][j]!=self._state[b][j] for b in range(len(geometry)))
                for j in range(len(e.components))):self.metrics['all_components_changed']+=1
        self._fixed_run=self._fixed_run+1 if same else 0
        available=e._compressed is not None and e._compressed.available(values)
        build=self.compression_after is not None and self._fixed_run>=self.compression_after
        if self.morphology_path=='compressed' or available or (same and build):
            answer=e.compressed_evaluation(values,excluded)
            self.metrics['compressed_calls']+=1
        else:
            fluxes=e.component_fluxes(values);plans=e.nuclear_plan(excluded);accumulator=GaussianAccumulator()
            for i,o in enumerate(e.observations):
                parameter=plans.get(o.band);retain=() if parameter is None else (parameter.split('.')[0],)
                residual,columns=self._pixel_residual(o,values,fluxes,geometry[i],excluded,retain)
                fixed=e.gaussian_data(o)
                column=None if parameter is None else columns[retain[0]]
                accumulator.add(residual,fixed.whitening.log_normalization,parameter=parameter,column=column)
            answer=accumulator.finish();self.metrics['pixel_calls']+=1
        self._state=geometry
        return answer

    def loglikelihood(self,values):return self.evaluate(values).loglikelihood
