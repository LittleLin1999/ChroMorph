from pathlib import Path
import json
from dataclasses import replace
import numpy as np
import pytest
from chromorph import Scene,PointSource,FreeBandSED,Parameter,SEDContext
from chromorph.direct import FullPixelLikelihood
from chromorph.policy import ExecutionPolicy
from chromorph.inference import PolyChordPosterior
from chromorph.provenance import model_signature
from chromorph.records import SavedResult,SavedPosterior
from chromorph.config import build_scene
from test_nuclear_priors import nucleus_scene
from conftest import observations,clump


@pytest.mark.parametrize('budget',[0,800,200000])
def test_order_independence_lazy_compression_and_input_changes(renderer,budget):
    with Scene(observations(size=13),renderer=renderer,cache_bytes=budget) as scene:
        scene.add_component(PointSource('nucleus',FreeBandSED(),x=6,y=6))
        scene.add_component(clump('one',x=5,y=6,moving=True))
        engine=ExecutionPolicy(scene);a=scene.registry.unpack();b=dict(a)
        b['one.morphology.x']+=.3;b['nucleus.morphology.y']+=.2
        for v in [a,b,a,dict(a,**{'one.flux.blue':.3}),b,a]:
            assert engine.loglikelihood(v)==pytest.approx(FullPixelLikelihood(scene).loglikelihood(v),abs=1e-8)
            assert scene._compressed is None
        count=scene.metrics['render_calls'];v=dict(a,**{'nucleus.flux.blue':2.})
        engine.loglikelihood(v)
        if budget>20000:assert scene.metrics['render_calls']==count
        scene.compressed.loglikelihood(a)
        for v in [a,b,a]:
            assert engine.loglikelihood(v)==pytest.approx(FullPixelLikelihood(scene).loglikelihood(v),abs=1e-8)
        for o in scene.observations:
            o.update(data=o.data+.01,sigma=o.sigma*1.2)
        assert engine.loglikelihood(a)==pytest.approx(FullPixelLikelihood(scene).loglikelihood(a),abs=1e-8)
        scene.observations[0].psf=replace(scene.observations[0].psf,center=(7.1,7.))
        assert engine.loglikelihood(a)==pytest.approx(FullPixelLikelihood(scene).loglikelihood(a),abs=1e-8)
        assert scene._basis_cache.nbytes<=budget//2


def test_conditional_restoration_uses_saved_untruncated_moments_without_replay(renderer,tmp_path,monkeypatch):
    with nucleus_scene(renderer) as scene:
        sampler=PolyChordPosterior(scene);score,moments=sampler.loglikelihood(np.empty(0))
        samples=np.empty((1000,0));derived=np.tile(moments,(1000,1))
        names,expected=sampler.restore_fluxes(samples,seed=19)
        count=scene.metrics['render_calls']
        monkeypatch.setattr(sampler,'evaluate',lambda *a:(_ for _ in ()).throw(AssertionError('replayed')))
        actual=sampler.restore_fluxes(samples,derived=derived,seed=19)[1]
        np.testing.assert_array_equal(actual,expected)
        assert scene.metrics['render_calls']==count and np.all(actual>=0)
        np.savetxt(tmp_path/'posterior.txt',np.column_stack([np.ones(1),[0],np.asarray(moments)[None,:]]))
        metadata=dict(posterior_schema_version=2,model_signature=sampler.signature(),parameter_names=[],derived_parameter_names=sampler.derived_names)
        (tmp_path/'inference.json').write_text(json.dumps(metadata))
        saved=SavedPosterior.read(tmp_path)
        assert saved.restore(sampler,count=100,seed=1)[1].shape==(100,2)
        metadata.pop('derived_parameter_names');metadata.pop('posterior_schema_version')
        (tmp_path/'inference.json').write_text(json.dumps(metadata));np.savetxt(tmp_path/'posterior.txt',[[1,0]])
        with pytest.raises(ValueError,match='Legacy conditional'):SavedPosterior.read(tmp_path).restore(sampler)


def test_analytic_sampling_and_result_roundtrip(renderer,tmp_path):
    with nucleus_scene(renderer) as scene:
        sampler=PolyChordPosterior(scene);out=sampler.run(tmp_path,restore_samples=25,seed=9)
        assert out.logZ is None
        saved=SavedPosterior.read(tmp_path);assert saved.restore(sampler,count=10)[1].shape==(10,2)
        report=json.loads((tmp_path/'inference.json').read_text())
        assert report['restoration']['restoration_replay_evaluations']==0
    result=SavedResult(dict(values={'a':2.},parameter_definitions={'a':dict(value=2.,lower=None,upper=None,vary=True,prior=None)}))
    result.write(tmp_path/'result.json');loaded=SavedResult.read(tmp_path/'result.json')
    assert loaded.values=={'a':2.} and np.isneginf(loaded.parameters()['a'].lower)
    result.document['result_schema_version']=999;result.write(tmp_path/'result.json')
    with pytest.raises(ValueError,match='Unsupported result'):SavedResult.read(tmp_path/'result.json')


def test_stable_model_signatures_and_config_migrations(renderer):
    with nucleus_scene(renderer) as scene:
        first=model_signature(scene);name=scene.registry.names()[0];p=scene.registry.parameters[name]
        scene.registry.parameters[name]=replace(p,value=2.)
        assert model_signature(scene)==first  # starting guesses are not priors
        scene.registry.parameters[name]=replace(p,upper=30.)
        assert model_signature(scene)!=first
        scene.registry.parameters[name]=p
        assert model_signature(scene)==first
        scene.observations[0].update(data=scene.observations[0].data+.01)
        assert model_signature(scene)!=first
    config=dict(components=[dict(name='one',spatial='sersic',sed=dict(model='free'))])
    with build_scene(config,observations(),renderer=renderer) as scene:assert scene.policy_options=={}
    config['execution']={'rendering':'aggregate'}
    with pytest.raises(ValueError,match='schema_version'):build_scene(config,observations(),renderer=renderer)
    config['schema_version']=2
    with build_scene(config,observations(),renderer=renderer) as scene:assert scene.policy_options=={'rendering':'aggregate'}
    config['schema_version']=99
    with pytest.raises(ValueError,match='Unsupported configuration'):build_scene(config,observations(),renderer=renderer)


def test_polychord_derived_schema_resume_and_mpi_writer(renderer,tmp_path,monkeypatch):
    import sys,types
    from chromorph import Prior
    captured={};rank=[0]
    class Settings:
        def __init__(self,ndim,nderived,**kwargs):
            self.base_dir=kwargs['base_dir'];captured.update(ndim=ndim,nderived=nderived)
    def run(callback,ndim,nderived,settings,prior,progress):
        x=prior(np.full(ndim,.5));ll,derived=callback(x)
        assert len(derived)==nderived
        if rank[0]==0:
            np.savetxt(Path(settings.base_dir)/'posterior.txt',[np.r_[1.,-2*ll,x,derived]])
        return types.SimpleNamespace(logZ=0.,logZerr=.1,nlike=1)
    module=types.ModuleType('pypolychord');module.run_polychord=run
    settings=types.ModuleType('pypolychord.settings');settings.PolyChordSettings=Settings
    mpi=types.ModuleType('mpi4py');mpi.MPI=types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(rank=0))
    monkeypatch.setitem(sys.modules,'pypolychord',module);monkeypatch.setitem(sys.modules,'pypolychord.settings',settings)
    monkeypatch.setitem(sys.modules,'mpi4py',mpi)
    with nucleus_scene(renderer) as scene:
        scene.add_component(clump('host',x=5,y=6,sed=FreeBandSED(parameters={b:Parameter(1.,0.,5.) for b in scene.bands})))
        sampler=PolyChordPosterior(scene);sampler.run(tmp_path,restore_samples=8,seed=17)
        assert captured==dict(ndim=2,nderived=4)
        saved=SavedPosterior.read(tmp_path)
        assert saved.samples.shape==(1,2) and saved.derived.shape==(1,4)
        assert sampler.metrics['restoration_replay_evaluations']==0
        (tmp_path/'posterior.resume').write_text('checkpoint')
        name=sampler.names[0];scene.registry.parameters[name]=replace(scene.registry.parameters[name],upper=6.)
        with pytest.raises(ValueError,match='Resume model'):PolyChordPosterior(scene).run(tmp_path)
        rank[0]=1;mpi.MPI.COMM_WORLD.rank=1
        PolyChordPosterior(scene).run(tmp_path/'rank1',restore_samples=0)
        assert not (tmp_path/'rank1/inference.json').exists()
