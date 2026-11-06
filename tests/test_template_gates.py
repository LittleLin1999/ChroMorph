import numpy as np
import pytest
from chromorph import Scene,SEDContext,Parameter
from chromorph.sed import FSPSStellarSED
from chromorph.templates import BandFluxLibrary,PrecomputedFSPSSED
from test_template_sed_engine import metadata
from conftest import observations,clump


class ConstantReference(FSPSStellarSED):
    def unit_fluxes(self,values,name,observations,context):return {o.band:1. for o in observations}


def test_signed_loglikelihood_gate_cannot_pass_large_negative_error():
    obs=observations();ctx=SEDContext(distance_mpc=10);ref=ConstantReference()
    values={'a.sed.'+k:p.value for k,p in ref.parameters.items()}
    library=BandFluxLibrary({'log_age':[-2.,0.]},np.ones((2,2)),metadata(obs,ctx,'constant'))
    points=[dict(age=.2)]
    with pytest.raises(ValueError,match='nonnegative'):library.validate(ref,values,'a',obs,ctx,points,likelihood_error=lambda *a:-100.)
    report=library.validate(ref,values,'a',obs,ctx,points,loglikelihood_error=lambda *a:-100.,validation_domain={'test':True})
    assert not report['production_ready'] and report['maximum_absolute_delta_loglikelihood']==100
    report=library.validate(ref,values,'a',obs,ctx,points,loglikelihood_error=lambda *a:0.)
    assert not report['production_ready']  # a declared image domain is required
    with pytest.raises(TypeError):library.metadata['reference']['sfh']='ssp'
    with pytest.raises(ValueError):library.flux[0,0]=2.


def test_process_local_sed_state_and_mass_only_calls(renderer,monkeypatch):
    obs=observations();ctx=SEDContext(distance_mpc=10)
    library=BandFluxLibrary({'log_age':[-2.,0.]},np.array([[1.,3.],[2.,.5]]),metadata(obs,ctx,'constant'))
    parameters=dict(metallicity=Parameter(-.3,vary=False),av=Parameter(.2,vary=False))
    with Scene(obs,context=ctx,renderer=renderer) as scene:
        definitions=[]
        for name in ['a','b']:
            sed=PrecomputedFSPSSED(library,parameters=parameters,experimental=True,fallback='raise')
            definitions.append(sed);scene.add_component(clump(name,sed=sed))
        v=scene.registry.unpack();v['a.sed.age']=.1;v['b.sed.age']=.5
        before=scene.component_fluxes(v);engine=scene.prepare()._spectral_engine
        assert all(engine.backends[n] is not d and engine.backends[n].library is library for n,d in zip(['a','b'],definitions))
        assert all(d.metrics['template_calls']==0 and d._executor is None for d in definitions)
        counts={n:b.metrics['template_calls'] for n,b in engine.backends.items()}
        v['a.sed.log_mass']+=1
        after=scene.component_fluxes(v)
        assert after['b']==before['b']
        np.testing.assert_allclose(list(after['a'].values()),10*np.array(list(before['a'].values())))
        assert counts=={n:b.metrics['template_calls'] for n,b in engine.backends.items()}
        v['a.sed.age']=.2;scene.component_fluxes(v)
        assert engine.backends['a'].metrics['template_calls']==counts['a']+1
        assert engine.backends['b'].metrics['template_calls']==counts['b']


def test_image_domain_gate_and_signed_full_pixel_discrepancy(renderer):
    from chromorph.templates import image_loglikelihood_discrepancy
    from chromorph.provenance import observation_signature
    obs=observations();ctx=SEDContext(distance_mpc=10)
    library=BandFluxLibrary({'log_age':[-2.,0.]},np.ones((2,2)),metadata(obs,ctx,'constant'))
    pars=dict(log_mass=Parameter(0.,-1,1),metallicity=Parameter(-.3,vary=False),av=Parameter(.2,vary=False))
    with Scene(obs,context=ctx,renderer=renderer) as scene:
        sed=PrecomputedFSPSSED(library,parameters=pars,experimental=True,fallback='raise')
        scene.add_component(clump('a',sed=sed));v=scene.registry.unpack();v['a.sed.age']=.1
        model=scene.predict(v)[0]
        for o in obs:o.update(data=model[o.name]+.001)
        truth=np.ones(2);predicted=np.array([1.002,.997]);bands=library.metadata['bands']
        exact=image_loglikelihood_discrepancy(scene,v,'a',predicted,truth,bands)
        reference=scene.pixel_loglikelihood(v);delta=0.
        for i,o in enumerate(obs):
            unit=scene.prepare().unit_template(scene.components[0],o,v)
            r=(o.data-model[o.name])[o.mask]/o.sigma[o.mask]
            d=(predicted[i]-1)*unit[o.mask]/o.sigma[o.mask]
            delta+=float(-.5*((r-d)@(r-d))+.5*(r@r))
        assert exact==pytest.approx(delta,abs=1e-10)
        domain=dict(observation_signatures=[observation_signature(o) for o in obs],scene_parameter_bounds={k:[x-1,x+1] for k,x in v.items()},parameter_bounds={
            'sed.log_mass':[-.1,.1],**{'morphology.'+k:[x-1,x+1] for k,x in scene.morphology(scene.components[0],v).items()}})
        ref=ConstantReference(parameters=pars)
        report=library.validate(ref,v,'a',obs,ctx,[{'age':.1},{'age':.4}],loglikelihood_error=lambda *a:0.,validation_domain=domain)
        assert report['production_ready']
        sed.experimental=False
        assert library.authorizes(v,'a',obs)
        assert np.isfinite(list(scene.component_fluxes(v)['a'].values())).all()
        changed=dict(v);changed['a.sed.log_mass']=.5
        with pytest.raises(ValueError,match='domain authorization'):scene.component_fluxes(changed)
        obs[0].update(sigma=obs[0].sigma*2)
        assert not library.authorizes(v,'a',obs)
        with pytest.raises(ValueError,match='domain authorization'):scene.component_fluxes(v)


def test_loaded_tables_share_immutable_arrays_with_a_bounded_cache(tmp_path):
    from chromorph.templates import _LOADED_LIBRARIES
    obs=observations();ctx=SEDContext(distance_mpc=10)
    library=BandFluxLibrary({'log_age':[-2.,0.]},np.ones((2,2)),metadata(obs,ctx,'constant'))
    library.save(tmp_path/'table.npz')
    a=PrecomputedFSPSSED(tmp_path/'table.npz');b=PrecomputedFSPSSED(tmp_path/'table.npz')
    assert a is not b and a.parameters is not b.parameters and a.library is b.library
    assert not a.library.flux.flags.writeable and _LOADED_LIBRARIES.nbytes<=_LOADED_LIBRARIES.max_bytes
