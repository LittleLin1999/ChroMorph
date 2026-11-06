"""Process-local derivative workspaces; no nested FSPS pools or per-point scenes."""
import copy,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

_MODEL=None
_NAMES=None
_PROJECTION=None


class PrecomputedSED:
    """Read-only spectral answers, with exact independent mass normalization."""
    def __init__(self, mass=False):self.answer=None;self.mass=mass
    def fluxes(self,values,name,observations,context):
        if self.answer is None:raise RuntimeError('Missing precomputed spectral answers')
        scale=10**values[name+'.sed.log_mass'] if self.mass else 1.
        return {b:scale*f for b,f in self.answer.items()}
    def close(self):pass


def _initialize(model,names,amplitudes):
    global _MODEL,_NAMES,_PROJECTION
    from multiprocessing.util import Finalize
    from .linear import LinearProjection
    _MODEL,_NAMES=model,names
    _PROJECTION=LinearProjection(model,amplitudes=amplitudes)
    Finalize(None,model.close,exitpriority=10)


def _evaluate(task):
    vector,answers=task
    for c in _MODEL.components:
        if isinstance(c.sed,PrecomputedSED):c.sed.answer=answers[c.name]
    evaluator=_MODEL.prepare();before=evaluator.metrics['render_calls'];started=time.process_time()
    values=_PROJECTION.solve(_MODEL.registry.unpack(vector,_NAMES))
    residual=evaluator.residual(values)
    return residual,dict(render_calls=evaluator.metrics['render_calls']-before,
        cpu_seconds=time.process_time()-started,linear_solves=1 if _PROJECTION.names else 0)


class JacobianWorkers:
    def __init__(self,scene,names,eligible,workers,*,amplitudes=None):
        from .sed import FreeBandSED,FSPSStellarSED
        from .linear import LinearProjection
        if not isinstance(workers,int) or workers<1:raise ValueError('workers must be a positive integer')
        self.scene=scene;self.names=list(names);self.validated=False;self.closed=False
        self.metrics=dict(render_calls=0,cpu_seconds=0.,linear_solves=0,tasks=0,ipc_wall_seconds=0.)
        plans=(LinearProjection(scene,stellar_mass=False,background=False,names=eligible).amplitudes
               if amplitudes is None else amplitudes)
        snapshot=copy.copy(scene);snapshot.components=[copy.copy(c) for c in scene.components]
        for c in snapshot.components:
            if not isinstance(c.sed,FreeBandSED):c.sed=PrecomputedSED(isinstance(c.sed,FSPSStellarSED))
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize,initargs=(snapshot,self.names,plans))

    def __call__(self,function,vectors):
        from .sed import FreeBandSED,FSPSStellarSED
        vectors=list(vectors);tasks=[]
        for vector in vectors:
            values=self.scene.registry.unpack(vector,self.names)
            fluxes=self.scene.component_fluxes(values);answers={}
            for c in self.scene.components:
                if isinstance(c.sed,FreeBandSED):continue
                scale=10**values[c.name+'.sed.log_mass'] if isinstance(c.sed,FSPSStellarSED) else 1.
                answers[c.name]={b:f/scale for b,f in fluxes[c.name].items()}
            tasks.append((vector,answers))
        started=time.perf_counter();results=self.pool.map(_evaluate,tasks)
        checks={0} if not self.validated and function is not None else set()
        def collected():
            try:
                for i,(residual,metrics) in enumerate(results):
                    for key,value in metrics.items():self.metrics[key]+=value
                    self.metrics['tasks']+=1
                    if i in checks:np.testing.assert_allclose(residual,function(vectors[i]),rtol=1e-10,atol=1e-10,
                        err_msg='Derivative worker differs from the full global profiled residual')
                    yield residual
                if function is not None:self.validated=True
            finally:self.metrics['ipc_wall_seconds']+=time.perf_counter()-started
        return collected()

    def close(self):
        if not self.closed:
            pool,self.pool=self.pool,None
            pool.shutdown(wait=True);self.closed=True
