"""Independent rendering, fixed-noise reuse and exact Gaussian comparisons."""
from dataclasses import replace
import numpy as np
import pytest
from chromorph import Scene,Sersic,PointSource,FreeBandSED,Parameter,EmpiricalPSF
from chromorph.rendering import GalSimRenderer
from chromorph.direct import FullPixelLikelihood
from chromorph.noise import DenseCovariance
from conftest import observations,clump


@pytest.mark.parametrize('correlated',[False,True])
def test_one_noise_preparation_across_all_roles(monkeypatch,renderer,correlated):
    import chromorph.noise as noise
    obs=observations(('b',),size=11)
    if correlated:
        n=obs[0].data.size
        obs[0].covariance=DenseCovariance(.002**2*(.97*np.eye(n)+.03*np.ones((n,n))))
    count=[0];original=noise.cholesky
    def counted(*args,**kwargs):count[0]+=1;return original(*args,**kwargs)
    monkeypatch.setattr(noise,'cholesky',counted)
    with Scene(obs,renderer=renderer,cache_bytes=0) as scene:
        scene.add_component(clump('one',x=4,y=5));v=scene.registry.unpack()
        for _ in range(5):
            residual=scene.residual(v,include_priors=False)
            score=scene.pixel_loglikelihood(v)
            scene.project(v,scene._eligible_linear(),scene.basis(v))
            assert score==pytest.approx(FullPixelLikelihood(scene).loglikelihood(v),abs=1e-9)
        assert scene.metrics['noise_preparations']==1
        assert count[0]==int(correlated)
        obs[0].update(data=obs[0].data+.001)
        scene.residual(v)
        assert count[0]==int(correlated)  # target changed, covariance did not
        mask=obs[0].mask.copy();mask[0,0]=False;obs[0].update(mask=mask)
        scene.residual(v)
        assert scene.metrics['noise_preparations']==2
        assert count[0]==2*int(correlated)


def real_scene(index,re,q,matrix):
    obs=observations(('blue','red'),size=41,noise=.01)
    for i,o in enumerate(obs):
        o.psf=EmpiricalPSF(o.psf.image,.04,center=(7.17,6.83))
        o.matrix=np.asarray(matrix);o.offset=np.array([2.1,-1.7])+i*.1
        if np.linalg.det(o.matrix)<0:o.offset[0]+=40.
    scene=Scene(obs,renderer=GalSimRenderer())
    scene.add_component(PointSource('nucleus',FreeBandSED({'blue':1.2,'red':.8}),
        parameters={'x':Parameter(18.3,vary=False),'y':Parameter(19.7,vary=False)}))
    for j in range(2):
        scene.add_component(Sersic('c'+str(j),FreeBandSED({'blue':2.+j,'red':1.+j}),parameters={
            'x':Parameter(13.7+7*j,vary=False),'y':Parameter(17.3+4*j,vary=False),
            're':Parameter(re*(1+j),.01,60.,vary=False),'n':Parameter(index,.3,6.2,vary=False),
            'q':Parameter(q,.1,1.,vary=False),'pa':Parameter(23.+19*j,-180,180,vary=False)}))
    return scene


@pytest.mark.parametrize('index,re,q,matrix',[
    (.5,2.,.9,[[1.,0.],[0.,1.]]),
    (1.3,.2,.4,[[.97,.08],[-.04,1.03]]),
    (5.5,.08,.25,[[-1.,.04],[.03,1.]]),
    (3.7,8.,.7,[[1.,.03],[.04,1.]])])
def test_aggregate_matches_reference_affine_psf_centers_and_wings(index,re,q,matrix):
    with real_scene(index,re,q,matrix) as scene:
        values=scene.registry.unpack();fluxes=scene.component_fluxes(values)
        models,_,_=scene.predict(values)
        for o in scene.observations:
            morphologies=[scene.morphology(c,values) for c in scene.components]
            amplitudes=[fluxes[c.name][o.band]/o.ujy_per_rate for c in scene.components]
            reference=models[o.name]-values['observation.'+o.name+'.background']
            before=scene.renderer.metrics['component_images_materialized']
            result=scene.renderer.render_total(scene.components,morphologies,amplitudes,o)
            assert scene.renderer.metrics['component_images_materialized']==before+1
            assert result.columns=={} and result.draw_count==2
            # Adopted budget: <1% of one pixel uncertainty, and <0.01 logL.
            assert np.max(np.abs(result.image-reference)/o.sigma)<.01
            o.update(data=reference+np.random.default_rng(5).normal(0,o.sigma,o.data.shape))
        direct=FullPixelLikelihood(scene);aggregate=FullPixelLikelihood(scene,rendering='aggregate')
        assert abs(direct.loglikelihood(values)-aggregate.loglikelihood(values))<.01
        linear=scene.nuclear_flux_names()
        a,ca=direct.marginalized_loglikelihood(values,linear)
        b,cb=aggregate.marginalized_loglikelihood(values,linear)
        assert abs(a-b)<.01
        for name in linear:np.testing.assert_allclose(ca[name],cb[name],rtol=1e-5,atol=1e-7)
        for o in scene.observations:
            morphologies=[scene.morphology(c,values) for c in scene.components]
            zero=scene.renderer.render_total(scene.components,morphologies,np.zeros(len(scene.components)),o)
            assert zero.draw_count==0 and not zero.image.any()


def test_aggregate_dense_mask_calibration_and_multiple_observations_one_band():
    original=real_scene(1.1,1.7,.6,[[1.,0.],[0.,1.]])
    extra=replace(original.observations[0],name='blue_repeat')
    with Scene([*original.observations,extra],renderer=original.renderer) as scene:
        for c in original.components:scene.add_component(c)
        for o in scene.observations:
            o.update(data=o.data[:11,:11],sigma=o.sigma[:11,:11],mask=o.mask[:11,:11])
            n=o.data.size;o.update(covariance=DenseCovariance(.01**2*(.98*np.eye(n)+.02*np.ones((n,n)))))
        values=scene.registry.unpack();values['observation.red.log_calibration']=.03
        values['observation.blue_repeat.log_calibration']=-.02
        for c in scene.components:
            values[c.name+'.morphology.x']=4.+.3*scene.components.index(c)
            values[c.name+'.morphology.y']=5.-.2*scene.components.index(c)
        direct=FullPixelLikelihood(scene);aggregate=FullPixelLikelihood(scene,rendering='aggregate')
        assert abs(direct.loglikelihood(values)-aggregate.loglikelihood(values))<.01
        x,cx=direct.marginalized_loglikelihood(values,scene.nuclear_flux_names())
        y,cy=aggregate.marginalized_loglikelihood(values,scene.nuclear_flux_names())
        assert abs(x-y)<.01
        for name in cx:np.testing.assert_allclose(cx[name],cy[name],rtol=1e-5,atol=1e-7)
        values['nucleus.morphology.x']+=.37
        values['observation.blue_repeat.dx']+=.21
        x,cx=direct.marginalized_loglikelihood(values,scene.nuclear_flux_names())
        y,cy=aggregate.marginalized_loglikelihood(values,scene.nuclear_flux_names())
        assert abs(x-y)<.01
    original.close()


def test_off_field_profiles_fall_back_without_losing_wings():
    with real_scene(4.1,3.,.6,[[1.,0.],[0.,1.]]) as scene:
        values=scene.registry.unpack();values['c0.morphology.x']=-10.
        o=scene.observations[0];m=[scene.morphology(c,values) for c in scene.components]
        a=[scene.component_fluxes(values)[c.name][o.band] for c in scene.components]
        reference=sum(x*scene.renderer.render(c,p,o) for x,c,p in zip(a,scene.components,m))
        actual=scene.renderer.render_total(scene.components,m,a,o)
        np.testing.assert_array_equal(actual.image,reference)
        assert scene.renderer.metrics['aggregate_fallbacks']==1
