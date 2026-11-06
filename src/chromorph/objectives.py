"""Prior-aware ML/MAP profiling, separate from ordinary image likelihoods."""
from dataclasses import dataclass
import time
import numpy as np
from .linear import LinearProjection


def parameter_scales(names,initial):
    scales=[]
    for name,value in zip(names,initial):
        scales.append(dict(x=1.,y=1.,dx=.3,dy=.3,n=1.,q=.2,pa=30.,metallicity=.5,av=.5,
            gas_logu=1.,log_mass=1.,log_calibration=.02).get(name.split('.')[-1],max(abs(value),.1)))
    return np.array(scales)


class FitBudgetExceeded(RuntimeError): pass


@dataclass
class ProfiledState:
    vector: np.ndarray
    token: tuple
    values: dict
    cost: float
    residual: object = None


class ProfiledObjective:
    """Global amplitude re-solve at every independent finite-difference point."""
    def __init__(self, scene, *, projection=True, stellar_mass=True, projection_names=None,
                 workers=1, max_wall_seconds=None, likelihood='compressed'):
        self.scene=scene;self.evaluator=scene.prepare();self.registry=scene.registry
        self.projection=LinearProjection(scene,stellar_mass=stellar_mass,names=projection_names,
            amplitudes=None if projection else [])
        self.names=self.registry.names(exclude=self.projection.names)
        self.lower,self.upper=self.registry.bounds(self.names)
        self.revision=self.registry.revision;self._state=None;self.best=None
        self.max_wall_seconds=max_wall_seconds;self.started=time.perf_counter();self.likelihood=likelihood
        self.metrics=dict(evaluations=0,jacobians=0,base_cache_hits=0,derivative_evaluations=0)
        self.workers=None
        if workers!=1:
            self.workers=self.evaluator.derivative_workers(self.names,self.projection.names,workers,
                amplitudes=self.projection.amplitudes)

    def _token(self):
        if self.registry.revision!=self.revision:raise RuntimeError('Model definitions changed during optimization')
        return (self.scene.context,tuple((o.likelihood_token(),o.psf.cache_token(),
            tuple(o.matrix.ravel()),tuple(o.offset),o.ujy_per_rate,
            getattr(o.throughput,'fingerprint',None)) for o in self.scene.observations))

    def evaluate(self,vector,*,residual=False):
        vector=np.asarray(vector,dtype=float);token=self._token()
        state=self._state
        if state is not None and state.token==token and np.array_equal(state.vector,vector):
            self.metrics['base_cache_hits']+=1
        else:
            if self.max_wall_seconds is not None and time.perf_counter()-self.started>self.max_wall_seconds:
                raise FitBudgetExceeded('Explicit optimization wall-time budget exhausted')
            values=self.projection.solve(self.registry.unpack(vector,self.names))
            if self.likelihood=='pixel' or residual:
                r=self.evaluator.residual(values);cost=float(r@r)
            else:r=None;cost=self.scene.compressed.chi2(values,include_priors=True)
            state=ProfiledState(vector.copy(),self._token(),values,cost,r)
            self._state=state;self.metrics['evaluations']+=1
            if self.best is None or state.cost<self.best.cost:self.best=state
        if residual and state.residual is None:state.residual=self.evaluator.residual(state.values)
        return state

    def residual(self,vector):return self.evaluate(vector,residual=True).residual
    def scalar(self,vector):return self.evaluate(vector).cost

    def difference_points(self,vector,relative_step=1e-4):
        vector=np.asarray(vector,dtype=float);scales=parameter_scales(self.names,vector)
        points=[];steps=[]
        for i,value in enumerate(vector):
            step=relative_step*max(abs(value),scales[i],1e-4)
            if value+step>self.upper[i]:step=-step
            candidate=np.clip(value+step,self.lower[i],self.upper[i]);step=candidate-value
            if step==0:raise ValueError('No finite-difference interval for '+self.names[i])
            trial=vector.copy();trial[i]=candidate;points.append(trial);steps.append(step)
        return points,np.array(steps)

    def jacobian(self,vector):
        base=self.evaluate(vector,residual=True);points,steps=self.difference_points(vector)
        results=(map(self.residual,points) if self.workers is None else self.workers(self.residual,points))
        columns=[(residual-base.residual)/step for residual,step in zip(results,steps)]
        self._state=base
        self.metrics['jacobians']+=1;self.metrics['derivative_evaluations']+=len(points)
        return np.column_stack(columns) if columns else np.empty((len(base.residual),0))

    def close(self):
        if self.workers is not None:self.evaluator.release_workers(self.workers);self.workers=None
