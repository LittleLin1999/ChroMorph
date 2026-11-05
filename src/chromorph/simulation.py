"""Injection/recovery on actual Observation grids, masks and empirical PSFs."""
from dataclasses import replace
from itertools import permutations
import numpy as np
from scipy.optimize import linear_sum_assignment


def simulate(scene, values=None, *, seed=0, noise_scale=1., renderer=None):
    values = scene.registry.unpack() if values is None else dict(values)
    old = scene.renderer
    if renderer is not None:
        scene.renderer = renderer
    try:
        models, _, _ = scene.predict(values)
    finally:
        scene.renderer = old
    rng = np.random.default_rng(seed)
    return [replace(o, data=models[o.name]+rng.normal(size=o.data.shape)*o.sigma*noise_scale)
            for o in scene.observations]


def match_components(truth, fitted):
    """Report optimal centroid label permutation and cost, never silently relabel."""
    true_names, fit_names = list(truth), list(fitted)
    if len(true_names) != len(fit_names):
        raise ValueError("Matching requires equal component counts")
    a = np.array([truth[n] for n in true_names]); b = np.array([fitted[n] for n in fit_names])
    cost = np.linalg.norm(a[:, None, :]-b[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    return dict(mapping={true_names[i]: fit_names[j] for i, j in zip(rows, cols)},
                total_centroid_distance=float(cost[rows, cols].sum()),
                label_switch=any(true_names[i] != fit_names[j] for i, j in zip(rows, cols)))


def recovery_experiment(injection_scene, fit_factory, *, repeats=5, seed=0, fit_options=None):
    """Factory must construct a new independent Scene for each realization.

    Use a different injection Scene/renderer or PSF to test SED/PSF mismatch;
    impose registration perturbations in injection values for astrometry tests.
    """
    truth = injection_scene.registry.unpack()
    runs = []
    for index in range(repeats):
        observations = simulate(injection_scene, truth, seed=seed+index)
        with fit_factory(observations) as scene:
            result = scene.fit(**(fit_options or {}))
            errors = {n: result.values[n]-v for n, v in truth.items() if n in result.values}
            coverage = {n: abs(errors[n]) <= err for n, err in result.uncertainty.items()
                        if err is not None and n in errors}
            runs.append(dict(seed=seed+index, errors=errors, local_1sigma_coverage=coverage,
                             chi2=result.chi2, uncertainty=result.uncertainty,
                             identifiability=result.identifiability,
                             convergence=any(r['success'] for r in result.runs)))
    names = sorted(set().union(*(r['errors'] for r in runs)))
    return dict(repeats=repeats, runs=runs,
                bias={n: float(np.mean([r['errors'][n] for r in runs if n in r['errors']])) for n in names},
                rmse={n: float(np.sqrt(np.mean([r['errors'][n]**2 for r in runs if n in r['errors']]))) for n in names},
                note='Finite-realization frequentist recovery; local covariance coverage is not full posterior coverage.')
