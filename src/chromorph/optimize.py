"""One SciPy deterministic engine over the common model and evaluator."""
from dataclasses import asdict,replace
from pathlib import Path
import inspect,json,time
import numpy as np
from scipy.optimize import least_squares,minimize
from .objectives import ProfiledObjective,FitBudgetExceeded,parameter_scales


def fit_model(scene,*,objective='map',**options):
    scene.validate_nuclear_seds()
    if objective not in {'map','ml'}:raise ValueError('Use map or ml objective')
    configured=scene.registry.parameters
    if objective=='ml':scene.registry.parameters={n:replace(p,prior=None) for n,p in configured.items()}
    try:
        result=fit_scene(scene,**options);result.objective_mode=objective
        result.parameter_definitions_snapshot={n:asdict(p) for n,p in scene.registry.parameters.items()}
    finally:
        if objective=='ml':scene.registry.parameters=configured
    return result


def fit_scene(scene,*,optimizer='trf',multistart=3,seed=0,max_nfev=300,variable_projection=True,
              project_stellar_mass=True,estimate_uncertainty=True,checkpoint_directory=None,
              tolerance=1e-7,likelihood='compressed',workers=1,jacobian='dependency',max_wall_seconds=None):
    from .result import FitResult,_json
    if likelihood not in {'compressed','pixel'}:raise ValueError('likelihood must be compressed or pixel')
    if likelihood=='pixel' and optimizer!='trf':raise ValueError('Pixel-reference fitting supports trf')
    if jacobian not in {'dependency','scipy'}:raise ValueError('Use dependency or scipy Jacobians')
    multistart=(3 if multistart else 1) if isinstance(multistart,bool) else int(multistart)
    if multistart<1 or optimizer not in {'trf','L-BFGS-B'}:raise ValueError('Use positive multistart and trf or L-BFGS-B')
    if not scene.components:raise ValueError('Scene has no components')
    if max_wall_seconds is not None and max_wall_seconds<=0:raise ValueError('A wall-time budget must be positive')
    # Preserve the old pixel configuration's amplitude eligibility. Both modes
    # now use this same optimizer, global linear solver and full residuals.
    selection=scene._eligible_linear() if likelihood=='pixel' else None
    objective=ProfiledObjective(scene,projection=variable_projection,stellar_mass=project_stellar_mass,
        projection_names=selection,workers=workers,max_wall_seconds=max_wall_seconds,likelihood=likelihood)
    names=objective.names;initial=scene.registry.initial(names);lo,hi=scene.registry.bounds(names)
    scale=parameter_scales(names,initial);rng=np.random.default_rng(seed);started=time.perf_counter()
    checkpoint=None if checkpoint_directory is None else Path(checkpoint_directory)
    if checkpoint is not None:checkpoint.mkdir(parents=True,exist_ok=True)
    def save(path,values,cost,**extra):
        if checkpoint is not None:
            data=dict(values=values,objective=cost,optimizer=optimizer,likelihood_calls=objective.metrics['evaluations'],
                elapsed_seconds=time.perf_counter()-started,**extra)
            temporary=checkpoint/(path+'.tmp');temporary.write_text(json.dumps(_json(data),indent=2,allow_nan=False)+'\n')
            temporary.replace(checkpoint/path)
    runs=[]
    try:
        for index in range(multistart):
            start=initial.copy()
            if index:start=np.clip(start+rng.normal(size=len(start))*.3*scale,lo+1e-10,hi-1e-10)
            first=dict(zip(names,map(float,start)))
            def callback(intermediate_result):
                vector=intermediate_result.x if optimizer=='trf' else initial+scale*intermediate_result.x
                state=objective.evaluate(vector)
                save(f'start{index+1}_checkpoint.json',state.values,state.cost,start=index+1,final=False)
            try:
                if not len(names):
                    vector=start;success=True;message='Linear solve only';nfev=1
                elif optimizer=='trf':
                    extra={}
                    if checkpoint is not None and 'callback' in inspect.signature(least_squares).parameters:extra['callback']=callback
                    if jacobian=='dependency':extra['jac']=objective.jacobian
                    elif objective.workers is not None and 'workers' in inspect.signature(least_squares).parameters:extra['workers']=objective.workers
                    fit=least_squares(objective.residual,start,bounds=(lo,hi),x_scale='jac',diff_step=1e-4,
                        ftol=tolerance,xtol=tolerance,gtol=tolerance,max_nfev=max_nfev,**extra)
                    vector,success,message,nfev=fit.x,bool(fit.success),str(fit.message),int(fit.nfev)
                else:
                    fit=minimize(lambda z:objective.scalar(initial+scale*z),(start-initial)/scale,
                        method='L-BFGS-B',bounds=list(zip((lo-initial)/scale,(hi-initial)/scale)),
                        callback=callback if checkpoint is not None else None,
                        options=dict(maxiter=max_nfev,maxfun=max_nfev*max(len(names)+1,2),ftol=tolerance,gtol=tolerance,eps=1e-4))
                    vector,success,message,nfev=initial+scale*fit.x,bool(fit.success),str(fit.message),int(fit.nfev)
                state=objective.evaluate(vector,residual=True)
            except FitBudgetExceeded as error:
                if objective.best is None:raise
                state=objective.best;success=False;message=str(error);nfev=objective.metrics['evaluations']
                if state.residual is None:state.residual=scene.residual(state.values)
            np.testing.assert_allclose(state.cost,state.residual@state.residual,rtol=1e-10,atol=1e-7,
                err_msg='Compressed and full-pixel objectives disagree')
            runs.append(dict(values=dict(state.values),objective=state.cost,success=success,message=message,nfev=nfev,
                initial_values=first,optimizer=optimizer,linear_diagnostics=objective.projection.last_diagnostics,
                budget_exhausted=isinstance(message,str) and 'budget exhausted' in message.lower()))
            save(f'start{index+1}_result.json',state.values,state.cost,start=index+1,final=True,success=success,message=message,nfev=nfev)
            if runs[-1]['budget_exhausted']:break
        best=min(runs,key=lambda run:run['objective'])
        result=FitResult(scene,best['values'],runs,objective.projection.names)
        result.performance=dict(likelihood_calls=objective.metrics['evaluations'],elapsed_seconds=time.perf_counter()-started,
            spatial=dict(scene.metrics),compressed=dict(scene.compressed.metrics),jacobian=dict(objective.metrics),
            linear=dict(objective.projection.metrics),workers=None if objective.workers is None else dict(objective.workers.metrics))
        if estimate_uncertainty:result.estimate_uncertainty()
        else:
            result.parameter_names=scene.registry.names();result.uncertainty={n:None for n in result.parameter_names}
            result.covariance=np.full((len(result.parameter_names),len(result.parameter_names)),np.nan);result.correlation=result.covariance.copy()
            result.identifiability=dict(rank=None,data_rank=None,null_modes=[],near_bounds=[],prior_information_fraction={})
            result.warnings.append('Uncertainties were not estimated in this discovery fit.')
        return result
    finally:objective.close()
