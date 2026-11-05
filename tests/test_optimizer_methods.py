import numpy as np
from chromorph import Scene
from chromorph.simulation import simulate
from conftest import observations, clump


def test_bounded_methods_match_joint_pixel_objective(renderer):
    truth = Scene(observations(noise=.001), renderer=renderer)
    truth.add_component(clump('one', {'blue': 2., 'red': .4}, x=8.))
    truth.add_component(clump('two', {'blue': .3, 'red': 3.}, x=16.))
    data = simulate(truth, seed=29)
    fitted = []
    for method in ['trf', 'L-BFGS-B']:
        scene = Scene(data, renderer=renderer)
        scene.add_component(clump('one', x=8.2, moving=True))
        scene.add_component(clump('two', x=15.8, moving=True))
        fit = scene.fit(multistart=1, optimizer=method, max_nfev=150, estimate_uncertainty=False)
        direct = scene.residual(fit.values, include_priors=False)
        np.testing.assert_allclose(fit.chi2, direct@direct, rtol=1e-12)
        assert fit.success
        fitted.append(fit)
    assert abs(fitted[0].chi2-fitted[1].chi2) < .05
