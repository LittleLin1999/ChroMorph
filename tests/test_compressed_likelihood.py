import numpy as np
import pytest
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import aslinearoperator
from chromorph import Scene, PointSource, FreeBandSED, Parameter, EmpiricalPSF
from chromorph.noise import DenseCovariance, FixedWhiteningOperator
from chromorph.likelihood import GaussianStatistics
from chromorph.cache import ExactLRU
from conftest import observations, clump
from test_independence import ToyPopulation


def scene_for(renderer):
    scene = Scene(observations(('b', 'v', 'r', 'i'), size=17), renderer=renderer,
                  fit_background=True, fit_registration=True)
    scene.add_component(PointSource('nucleus', FreeBandSED(), parameters={
        'x': Parameter(8., 5., 11.), 'y': Parameter(8., 5., 11.)}))
    scene.add_component(clump('one', x=6, y=9, sed=ToyPopulation(), moving=True))
    scene.add_component(clump('two', x=12, y=8, sed=ToyPopulation(1.5, .4, 2.), moving=True))
    model, _, _ = scene.predict()
    rng = np.random.default_rng(73)
    for o in scene.observations:
        o.data[:] = model[o.name]+rng.normal(size=o.data.shape)*o.sigma
        o.mask[0, :] = False
    return scene


def assert_equivalent(scene, values):
    residual = scene.residual(values, include_priors=False)
    np.testing.assert_allclose(scene.compressed.chi2(values), residual@residual, rtol=2e-12, atol=1e-9)
    np.testing.assert_allclose(scene.compressed.loglikelihood(values), scene.pixel_loglikelihood(values),
                               rtol=2e-12, atol=1e-9)


def test_complete_model_and_compression_agree_and_avoid_unrelated_work(renderer):
    scene = scene_for(renderer)
    values = scene.registry.unpack()
    assert_equivalent(scene, values)
    rendered = scene.metrics['render_calls']
    built = scene.compressed.metrics['statistics_builds']
    values['nucleus.flux.b'] *= 1.4
    assert_equivalent(scene, values)
    assert scene.metrics['render_calls'] == rendered
    assert scene.compressed.metrics['statistics_builds'] == built
    values['one.sed.age'] *= 1.1
    assert_equivalent(scene, values)
    assert scene.metrics['render_calls'] == rendered
    assert scene.compressed.metrics['statistics_builds'] == built
    values['two.morphology.x'] += .3
    assert_equivalent(scene, values)
    assert scene.metrics['render_calls'] == rendered+4
    assert scene.compressed.metrics['statistics_builds'] == built+4


def test_mask_noise_and_psf_changes_invalidate_correct_statistics(renderer):
    scene = scene_for(renderer);values = scene.registry.unpack()
    assert_equivalent(scene, values)
    count = scene.metrics['render_calls'];built = scene.compressed.metrics['statistics_builds']
    o = scene.observations[0]
    with pytest.raises(ValueError):
        o.sigma[2, 2] = 3  # fixed inputs cannot silently mutate under a cache
    mask = o.mask.copy();mask[2, 2] = False
    o.update(mask=mask, sigma=o.sigma*1.2)
    assert_equivalent(scene, values)
    assert scene.metrics['render_calls'] == count
    assert scene.compressed.metrics['statistics_builds'] == built+1
    psf = o.psf
    o.psf = EmpiricalPSF(psf.image.copy(), psf.pixel_scale, (psf.center[0]+.1, psf.center[1]))
    assert_equivalent(scene, values)
    assert scene.metrics['render_calls'] == count+3


@pytest.mark.parametrize('operator', [False, True])
def test_fixed_nondiagonal_covariance_and_normalization(renderer, operator):
    obs = observations(('b',), size=8)
    o = obs[0];o.mask[0, 0] = False
    size = o.data.size
    c = .000004*(.8*np.eye(size)+.2*np.ones((size, size)))
    if operator:
        indices = np.flatnonzero(o.mask.ravel())
        l = cholesky(c[np.ix_(indices, indices)], lower=True)
        w = solve_triangular(l, np.eye(len(indices)), lower=True)
        o.covariance = FixedWhiteningOperator(aslinearoperator(w), float(2*np.log(np.diag(l)).sum()), tuple(indices))
    else:
        o.covariance = DenseCovariance(c)
    scene = Scene(obs, renderer=renderer)
    scene.add_component(clump('one', {'b': 2.}, x=3, y=4))
    o.data[:] = scene.predict()[0]['b']+np.random.default_rng(4).normal(0, .001, o.data.shape)
    assert_equivalent(scene, scene.registry.unpack())
    from chromorph.linear import LinearProjection
    values = scene.registry.unpack()
    full = scene.project(values, scene._eligible_linear(), scene.basis(values))
    compressed = LinearProjection(scene).solve(values)
    np.testing.assert_allclose(full['one.flux.b'], compressed['one.flux.b'], rtol=1e-11)


def test_degenerate_templates_report_rank_and_stable_objective(renderer):
    scene = Scene(observations(), renderer=renderer)
    scene.add_component(clump('one'));scene.add_component(clump('two'))
    stats = scene.compressed.statistics(scene.registry.unpack())
    assert all(s.rank < 3 and np.isinf(s.condition_number) for s in stats.values())
    assert_equivalent(scene, scene.registry.unpack())


def test_exact_cache_has_an_explicit_memory_bound():
    cache = ExactLRU(200)
    cache['one'] = np.ones(20);cache['two'] = np.zeros(20)
    assert cache.get('one') is None and cache.get('two') is not None
    assert cache.nbytes <= 200 and cache.evictions == 1
