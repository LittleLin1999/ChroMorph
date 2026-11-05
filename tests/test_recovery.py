import numpy as np
import pytest
from chromorph import Scene, FreeBandSED, PointSource, Parameter
from chromorph.simulation import simulate, match_components
from conftest import observations, clump
from test_independence import ToyPopulation


@pytest.mark.parametrize('separation', [8., 2.5])
def test_free_band_separated_and_overlapping_recovery(renderer, separation):
    obs = observations(noise=.0005)
    truth = Scene(obs, renderer=renderer)
    truth.add_component(clump('one', {'blue': 2., 'red': .3}, x=9))
    truth.add_component(clump('two', {'blue': .4, 'red': 3.}, x=9+separation))
    fitted = Scene(simulate(truth, seed=47), renderer=renderer)
    fitted.add_component(clump('one', {'blue': 1., 'red': 1.}, x=9, moving=True))
    fitted.add_component(clump('two', {'blue': 1., 'red': 1.}, x=9+separation, moving=True))
    result = fitted.fit(multistart=2, max_nfev=100)
    for component, expected in [('one', [2., .3]), ('two', [.4, 3.])]:
        np.testing.assert_allclose(list(result.fluxes[component].values()), expected, atol=.09)
    assert min(result.runs, key=lambda r: r['objective'])['success']


@pytest.mark.parametrize('separation', [8., 3.])
def test_recover_distinct_population_age_dust_mass_when_constrained(renderer, separation):
    obs = observations(('b', 'v', 'r', 'i'), noise=.0001)
    truth = Scene(obs, renderer=renderer)
    truth.add_component(clump('one', x=8, sed=ToyPopulation(.3, .1, 2)))
    truth.add_component(clump('two', x=8+separation, sed=ToyPopulation(2., .6, 3)))
    scene = Scene(simulate(truth, seed=12), renderer=renderer)
    scene.add_component(clump('one', x=8, sed=ToyPopulation(.5, .2, 1.5)))
    scene.add_component(clump('two', x=8+separation, sed=ToyPopulation(1.5, .3, 2.5)))
    result = scene.fit(multistart=2, max_nfev=100)
    for name, expected in [('one', [.3, .1, 2]), ('two', [2., .6, 3])]:
        actual = [result.values[name+'.sed.'+k] for k in ['age', 'dust', 'mass']]
        np.testing.assert_allclose(actual, expected, atol=.08)
    assert result.identifiability['data_rank'] == 6


def test_insufficient_filters_reports_unconstrained_population(renderer):
    truth = Scene(observations(noise=.0005), renderer=renderer)
    truth.add_component(clump('one', sed=ToyPopulation(.4, .2, 2)))
    scene = Scene(simulate(truth, seed=2), renderer=renderer)
    scene.add_component(clump('one', sed=ToyPopulation(.5, .3, 1.5)))
    result = scene.fit(multistart=1)
    assert result.identifiability['data_rank'] < 3
    assert any(v is None for v in result.uncertainty.values())


def test_identical_morphology_flux_exchange_is_degenerate(renderer):
    scene = Scene(observations(), renderer=renderer)
    scene.add_component(clump('one', {'blue': 1, 'red': 3}))
    scene.add_component(clump('two', {'blue': 3, 'red': 1}))
    v = scene.registry.unpack(); before = scene.predict(v)[0]
    v['one.flux.blue'] += .5; v['two.flux.blue'] -= .5
    after = scene.predict(v)[0]
    np.testing.assert_allclose(before['blue'], after['blue'])
    assert scene.predict()[2]['one'] != scene.predict()[2]['two']
    result = scene.fit(multistart=1)
    assert result.identifiability['rank'] < result.n_parameters
    assert result.identifiability['null_modes']


def test_nucleus_non_smooth_flux_recovery(renderer):
    obs = observations(('b', 'v', 'r', 'i'), noise=.0005)
    truth = Scene(obs, renderer=renderer)
    pars = {'x': Parameter(12., vary=False), 'y': Parameter(12., vary=False)}
    truth.add_component(PointSource('nucleus', FreeBandSED(dict(zip(['b', 'v', 'r', 'i'], [3, .1, 5, .03]))), parameters=pars))
    scene = Scene(simulate(truth, seed=77), renderer=renderer)
    scene.add_component(PointSource('nucleus', FreeBandSED(), parameters=pars))
    result = scene.fit(multistart=1)
    np.testing.assert_allclose(list(result.fluxes['nucleus'].values()), [3, .1, 5, .03], atol=.006)


def test_variable_projection_matches_full_global_fit(renderer):
    truth = Scene(observations(noise=.0005), renderer=renderer)
    truth.add_component(clump('a', {'blue': 2, 'red': 3}))
    data = simulate(truth, seed=81)
    results = []
    for projection in [False, True]:
        scene = Scene(data, renderer=renderer)
        scene.add_component(clump('a', moving=True))
        results.append(scene.fit(multistart=1, variable_projection=projection))
    assert abs(results[0].chi2-results[1].chi2) < .01
    np.testing.assert_allclose(list(results[0].fluxes['a'].values()), list(results[1].fluxes['a'].values()), atol=1e-4)


def test_label_switch_matching_is_explicit():
    result = match_components({'one': [1, 1], 'two': [8, 8]}, {'one': [8, 8], 'two': [1, 1]})
    assert result['label_switch'] and result['mapping']['one'] == 'two'


def test_wrong_sed_family_leaves_detectable_residuals(renderer):
    bands = ('b', 'v', 'r', 'i')
    truth = Scene(observations(bands, noise=.0005), renderer=renderer)
    truth.add_component(clump('a', dict(zip(bands, [2, .1, 4, .2]))))
    scene = Scene(simulate(truth, seed=83), renderer=renderer)
    scene.add_component(clump('a', sed=ToyPopulation()))
    result = scene.fit(multistart=2)
    assert result.chi2/result.n_pixels > 3  # this smooth family cannot fit alternating colors
