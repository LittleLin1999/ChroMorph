"""A free nuclear SED is distinct from its independent statistical priors."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from chromorph import Scene, PointSource, FreeBandSED, Parameter, Prior
from chromorph.config import build_scene
from chromorph.inference import PolyChordSampler, PolyChordPosterior, nuclear_prior_metadata
from conftest import observations


def nucleus_scene(renderer, priors=None, bands=('blue', 'red')):
    scene = Scene(observations(bands, size=13, noise=.03), renderer=renderer)
    scene.add_component(PointSource('central', FreeBandSED(parameters={
        b: Parameter(1., 0., np.inf, prior=(priors or {}).get(b)) for b in scene.bands
    }), nuclear=True, parameters={k: Parameter(6., vary=False) for k in ['x', 'y']}))
    return scene


def test_n_independent_fluxes_and_diagonal_prior_transform(renderer):
    bands = ('blue', 'green', 'red', 'infrared')
    priors = {b: Prior('uniform', lower=0, upper=5.+i*7.) for i, b in enumerate(bands)}
    with nucleus_scene(renderer, priors, bands) as scene:
        sampler = PolyChordSampler(scene)
        assert sampler.evidence_defined and len(sampler.names) == len(bands)
        center = np.full(len(bands), .4)
        values = scene.registry.unpack(sampler.prior(center), sampler.names)
        expected = -sum(np.log(priors[b].upper) for b in bands)
        assert scene.registry.logprior(values) == pytest.approx(expected)
        base_fluxes = scene.component_fluxes(values)['central']
        for j, name in enumerate(sampler.names):
            cube = center.copy();cube[j] = .7
            transformed = sampler.prior(cube)
            changed = scene.registry.unpack(transformed, sampler.names)
            fluxes = scene.component_fluxes(changed)['central']
            for b in bands:
                assert (fluxes[b] != base_fluxes[b]) == (name == f'central.flux.{b}')
            np.testing.assert_array_equal(np.delete(transformed,j), np.delete(sampler.prior(center),j))
        metadata = nuclear_prior_metadata(scene)
        assert metadata['spectral_model'] is None
        assert metadata['cross_filter_prior_correlation'] is False
        assert all(row['proper'] for row in metadata['fluxes'].values())


def test_unbounded_but_proper_half_normal_fluxes_allow_evidence(renderer):
    # Infinite support does not imply an improper prior.
    priors = {'blue': Prior('normal', mean=0., sigma=20.),
              'red': Prior('lognormal', mean=1., sigma=2.)}
    with nucleus_scene(renderer, priors) as scene:
        sampler = PolyChordSampler(scene)
        assert sampler.evidence_defined
        assert np.isinf(sampler.upper).all()
        for distribution in sampler._prior_transform.priors:
            integral = quad(lambda x: np.exp(distribution.logpdf(x)), 0, np.inf)[0]
            assert integral == pytest.approx(1., abs=1e-8)
        assert np.isfinite(sampler.prior(np.array([.01,.99]))).all()


def test_improper_flat_option_is_explicit_and_never_invents_flux_caps(renderer):
    with nucleus_scene(renderer) as scene:
        with pytest.raises(ValueError, match='Proper independent prior required'):
            PolyChordSampler(scene)
        posterior = PolyChordPosterior(scene)
        assert not posterior.evidence_defined
        assert set(posterior.linear) == set(scene.nuclear_flux_names())
        assert all(np.isinf(scene.registry.parameters[n].upper) for n in posterior.linear)
        assert all(not p['proper'] for p in nuclear_prior_metadata(scene,posterior.linear)['fluxes'].values())


def test_proper_priors_never_disable_evidence_in_flat_marginal_mode(renderer):
    with nucleus_scene(renderer, {b: Prior('uniform', lower=0, upper=8) for b in ['blue','red']}) as scene:
        sampler = PolyChordPosterior(scene)
        assert sampler.linear == [] and sampler.evidence_defined
        assert set(sampler.names) == set(scene.nuclear_flux_names())


def test_nuclear_prior_shape_factorizes_with_distinct_distributions(renderer):
    from chromorph.priors import prior_for
    priors = {'blue':Prior('normal', mean=2, sigma=1), 'red':Prior('lognormal', mean=0, sigma=.7)}
    with nucleus_scene(renderer,priors) as scene:
        sampler=PolyChordSampler(scene)
        for cube in [np.array([.1,.8]),np.array([.7,.2])]:
            vector=sampler.prior(cube);values=scene.registry.unpack(vector,sampler.names)
            expected=sum(prior_for(scene.registry.parameters[n]).logpdf(values[n]) for n in sampler.names)
            assert scene.registry.logprior(values)==pytest.approx(expected)
        # A change of the prior in blue cannot alter the red transform.
        before=sampler.prior(np.array([.6,.6]))
        scene.registry.parameters['central.flux.blue']=replace(
            scene.registry.parameters['central.flux.blue'],prior=Prior('normal',mean=0,sigma=50))
        after=PolyChordSampler(scene).prior(np.array([.6,.6]))
        assert before[sampler.names.index('central.flux.red')]==after[sampler.names.index('central.flux.red')]


def test_uniform_prior_width_changes_evidence_without_nuclear_colors(renderer):
    log_evidences=[]
    for width in [5.,10.,20.]:
        with nucleus_scene(renderer,{b:Prior('uniform',lower=0,upper=width) for b in ['blue','red']}) as scene:
            truth=scene.registry.unpack()
            truth.update({'central.flux.blue':1.3,'central.flux.red':.7})
            models,_,_=scene.predict(truth)
            for o in scene.observations:o.update(data=models[o.name])
            sampler=PolyChordSampler(scene)
            center=np.array([truth[n] for n in sampler.names])
            peak=sampler.evaluate(center)[0]
            actual=peak;expected=0.
            basis=scene.basis(truth)
            for j,name in enumerate(sampler.names):
                def relative_likelihood(flux):
                    vector=center.copy();vector[j]=flux
                    return np.exp(sampler.evaluate(vector)[0]-peak)/width
                actual+=np.log(quad(relative_likelihood,0,width,points=[center[j]],epsabs=1e-11)[0])
                band=name.split('.')[-1];o=next(o for o in scene.observations if o.band==band)
                whitening=scene.gaussian_data(o).whitening
                column=whitening.whiten((basis[o.name]['central']/o.ujy_per_rate)[o.mask])
                target=whitening.whiten(o.data[o.mask])
                sigma=1/np.linalg.norm(column);mean=(column@target)*sigma**2
                residual=target-column*mean
                expected+=whitening.log_normalization-.5*(residual@residual)
                expected+=np.log(sigma*np.sqrt(2*np.pi)*(ndtr((width-mean)/sigma)-ndtr(-mean/sigma))/width)
            assert actual==pytest.approx(expected,abs=1e-8)
            log_evidences.append(actual)
    np.testing.assert_allclose(np.diff(log_evidences),-2*np.log(2),atol=1e-9)


def test_registry_edits_cannot_bypass_nuclear_independence(renderer):
    with nucleus_scene(renderer,{b:Prior('uniform',lower=0,upper=8) for b in ['blue','red']}) as scene:
        scene.registry.parameters['central.flux.red']=replace(scene.registry.parameters['central.flux.red'],prior=None)
        scene.registry.tie('central.flux.red','central.flux.blue')
        with pytest.raises(ValueError,match='cannot be tied'):PolyChordSampler(scene)
        with pytest.raises(ValueError,match='cannot be tied'):scene.fit(objective='ml')


def test_nuclear_priors_in_yaml_need_no_opt_in(renderer):
    config=dict(components=[dict(name='central',spatial='point',nuclear=True,
        parameters=dict(x=6.,y=6.),sed=dict(model='free',parameters={
        'blue':dict(value=1.,min=0.,prior=dict(distribution='uniform',lower=0.,upper=15.)),
        'red':dict(value=1.,min=0.,prior=dict(distribution='normal',mean=0.,sigma=40.))}))])
    with build_scene(config,observations(size=13),renderer=renderer) as scene:
        sampler=PolyChordSampler(scene)
        assert sampler.evidence_defined and len(sampler.names)==2
        assert scene.registry.parameters['central.flux.blue'].upper==np.inf
