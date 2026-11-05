"""Deterministic joint fitting with stable amplitude variable projection."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares, minimize
from .linear import LinearProjection


def parameter_scales(names, initial):
    scales = []
    for name, value in zip(names, initial):
        key = name.split('.')[-1]
        scale = dict(x=1., y=1., dx=.3, dy=.3, n=1., q=.2, pa=30.,
                     metallicity=.5, av=.5, gas_logu=1., log_mass=1.,
                     log_calibration=.02).get(key, max(abs(value), .1))
        scales.append(scale)
    return np.array(scales)


def fit_scene(scene, *, optimizer='trf', multistart=3, seed=0, max_nfev=300,
              variable_projection=True, project_stellar_mass=True,
              estimate_uncertainty=True, checkpoint_directory=None,
              tolerance=1e-7):
    from .result import FitResult, _json
    multistart = (3 if multistart else 1) if isinstance(multistart, bool) else int(multistart)
    if multistart < 1 or optimizer not in {'trf', 'L-BFGS-B'}:
        raise ValueError('Use a positive multistart count and trf or L-BFGS-B')
    if not scene.components:
        raise ValueError('Scene has no components')
    projection = LinearProjection(scene, stellar_mass=project_stellar_mass)
    projected = projection.names if variable_projection else []
    names = scene.registry.names(exclude=projected)
    initial = scene.registry.initial(names)
    lo, hi = scene.registry.bounds(names)
    scale = parameter_scales(names, initial)
    rng = np.random.default_rng(seed)
    checkpoint = None if checkpoint_directory is None else Path(checkpoint_directory)
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    evaluations = [0]

    def decode(vector):
        values = scene.registry.unpack(vector, names)
        return projection.solve(values) if variable_projection else values

    def residual(vector):
        evaluations[0] += 1
        # A geometry-dependent QR rotation preserves chi-square but changes
        # the Gauss-Newton approximation. Keep the full pixel residual for
        # TRF morphology steps; L-BFGS-B uses the exact compressed scalar.
        return scene.residual(decode(vector))

    def objective(vector):
        evaluations[0] += 1
        return scene.compressed.chi2(decode(vector), include_priors=True)

    def save(path, values, cost, **extra):
        if checkpoint is not None:
            data = dict(values=values, objective=cost, optimizer=optimizer,
                        likelihood_calls=evaluations[0], elapsed_seconds=time.perf_counter()-started, **extra)
            temporary = checkpoint/(path+'.tmp')
            temporary.write_text(json.dumps(_json(data), indent=2, allow_nan=False)+'\n')
            temporary.replace(checkpoint/path)

    runs = []
    for index in range(multistart):
        start = initial.copy()
        if index:
            start += rng.normal(size=len(start))*.3*scale
            start = np.clip(start, lo+1e-10, hi-1e-10)
        first = dict(zip(names, map(float, start)))
        last_best = [np.inf]
        def callback(intermediate_result):
            vector = intermediate_result.x if optimizer == 'trf' else initial+scale*intermediate_result.x
            values = decode(vector)
            cost = scene.compressed.chi2(values, include_priors=True)
            if cost < last_best[0]:
                save(f'start{index+1}_checkpoint.json', values, cost, start=index+1, final=False)
                last_best[0] = cost
        if not len(names):
            vector, success, message, nfev = start, True, 'Linear solve only', 1
        elif optimizer == 'trf':
            # Older SciPy remains usable without its optional iteration callback.
            import inspect
            extra = {'callback': callback} if checkpoint is not None and 'callback' in inspect.signature(least_squares).parameters else {}
            fit = least_squares(residual, start, bounds=(lo, hi), x_scale='jac',
                diff_step=1e-4, ftol=tolerance, xtol=tolerance, gtol=tolerance,
                max_nfev=max_nfev, **extra)
            vector, success, message, nfev = fit.x, fit.success, fit.message, fit.nfev
        else:
            fit = minimize(lambda z: objective(initial+scale*z), (start-initial)/scale,
                method='L-BFGS-B', bounds=list(zip((lo-initial)/scale, (hi-initial)/scale)),
                callback=callback if checkpoint is not None else None,
                options=dict(maxiter=max_nfev, maxfun=max_nfev*max(len(names)+1, 2),
                             ftol=tolerance, gtol=tolerance, eps=1e-4), jac=None)
            vector, success, message, nfev = initial+scale*fit.x, fit.success, fit.message, fit.nfev
        values = decode(vector)
        cost = scene.compressed.chi2(values, include_priors=True)
        direct = scene.residual(values)
        np.testing.assert_allclose(cost, direct@direct, rtol=1e-10, atol=1e-7,
                                   err_msg='Final compressed/pixel likelihood mismatch')
        run = dict(values=values, objective=cost, success=bool(success), message=str(message),
                   nfev=int(nfev), initial_values=first, optimizer=optimizer,
                   linear_diagnostics=projection.last_diagnostics if variable_projection else {})
        runs.append(run)
        save(f'start{index+1}_result.json', values, cost, start=index+1, final=True,
             success=bool(success), message=str(message), nfev=int(nfev))
    best = min(runs, key=lambda r: r['objective'])
    result = FitResult(scene, best['values'], runs, projected)
    result.performance = dict(likelihood_calls=evaluations[0], elapsed_seconds=time.perf_counter()-started,
                              spatial=dict(scene.metrics), compressed=dict(scene.compressed.metrics))
    if estimate_uncertainty:
        result.estimate_uncertainty()
    else:
        result.parameter_names = scene.registry.names()
        result.uncertainty = {n: None for n in result.parameter_names}
        result.covariance = np.full((len(result.parameter_names), len(result.parameter_names)), np.nan)
        result.correlation = result.covariance.copy()
        result.identifiability = dict(rank=None, data_rank=None, null_modes=[], near_bounds=[],
                                      prior_information_fraction={})
        result.warnings.append('Uncertainties were not estimated in this discovery fit.')
    return result
