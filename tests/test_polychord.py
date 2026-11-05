"""Exact nuclear marginalization, prior transforms and full joint dependence."""
import numpy as np
import pytest
from scipy.integrate import quad
from chromorph import Scene, PointSource, FreeBandSED, Parameter
from chromorph.inference import marginal_linear_flux, PolyChordPosterior
from conftest import observations, clump


@pytest.mark.parametrize('mean', [-3., .3, 4.])
def test_marginalized_flux_matches_numerical_integral(mean):
    column = np.array([1., 2., .2])
    residual = mean*column+np.array([.2, -.1, 0.])
    expected = quad(lambda flux: np.exp(-.5*np.sum((residual-column*flux)**2)), 0, np.inf,
                    epsabs=1e-24, epsrel=1e-11)[0]
    actual, fitted_mean, sigma = marginal_linear_flux(residual, column)
    np.testing.assert_allclose(np.exp(actual), expected, rtol=1e-9)
    np.testing.assert_allclose(fitted_mean, mean)
    assert sigma > 0


def test_posterior_preserves_free_nuclear_bands_and_joint_model(renderer):
    scene = Scene(observations(), renderer=renderer, fit_registration=True)
    scene.add_component(PointSource('nucleus', FreeBandSED(), parameters={
        'x': Parameter(12., vary=False), 'y': Parameter(12., vary=False)}))
    # Explicitly bounded non-nuclear free fluxes have proper uniform priors.
    scene.add_component(clump('clump1', sed=FreeBandSED(parameters={
        b: Parameter(1., 0., 5.) for b in scene.bands})))
    scene.add_component(clump('clump2', x=17., sed=FreeBandSED(parameters={
        b: Parameter(2., 0., 5.) for b in scene.bands})))
    posterior = PolyChordPosterior(scene)
    assert posterior.linear == ['nucleus.flux.blue', 'nucleus.flux.red']
    assert len(posterior.names) == 6
    x = scene.registry.initial(posterior.names)
    score, conditional = posterior.evaluate(x)
    assert np.isfinite(score) and set(conditional) == set(posterior.linear)
    changed = x.copy(); changed[posterior.names.index('clump1.flux.blue')] += .5
    assert posterior.evaluate(changed)[0] != score
    names, draws = posterior.restore_fluxes(np.array([x, changed]))
    assert draws.shape == (2, 8)
    assert (draws[:, -2:] >= 0).all()
    transformed = posterior.prior(np.full(6, .5))
    assert np.isfinite(transformed).all()
    assert transformed[posterior.names.index('observation.red.dx')] == pytest.approx(0.)


def test_unobserved_nuclear_flux_is_rejected():
    with pytest.raises(ValueError, match='improper'):
        marginal_linear_flux(np.zeros(3), np.zeros(3))


@pytest.mark.parametrize('correlated', [False, True])
def test_compressed_marginal_matches_full_pixel_integral(renderer, correlated):
    from chromorph.noise import DenseCovariance, whitening_for
    obs = observations(('blue', 'red'), size=17)
    if correlated:
        for o in obs:
            o.covariance = DenseCovariance(.002**2*(.95*np.eye(o.data.size)+.05*np.ones((o.data.size,o.data.size))))
    scene = Scene(obs, renderer=renderer)
    scene.add_component(PointSource('nucleus', FreeBandSED(), parameters={
        'x': Parameter(8., vary=False), 'y': Parameter(8., vary=False)}))
    scene.add_component(clump('one', x=6., y=8., sed=FreeBandSED(parameters={
        b:Parameter(1.,0.,5.) for b in scene.bands})))
    model,_,_=scene.predict()
    for o in obs:o.data[:]=model[o.name]
    posterior=PolyChordPosterior(scene)
    vector=scene.registry.initial(posterior.names)
    actual,_=posterior.evaluate(vector)
    values=scene.registry.unpack(vector,posterior.names)
    for name in posterior.linear:values[name]=0.
    models,_,_=scene.predict(values);basis=scene.basis(values)
    expected=0.
    for o in obs:
        whitening=whitening_for(o)
        residual=whitening.whiten((o.data-models[o.name])[o.mask])
        column=whitening.whiten((basis[o.name]['nucleus']/o.ujy_per_rate)[o.mask])
        expected+=marginal_linear_flux(residual,column)[0]+whitening.log_normalization
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-9)
    count=scene.metrics['render_calls']
    posterior.evaluate(vector+.1)
    assert scene.metrics['render_calls']==count
