import numpy as np
import pytest
from chromorph import Scene,FreeBandSED,PointSource,Parameter
from chromorph.posterior import ExactPosteriorTarget
from chromorph.transactions import TransactionalEvaluator,SumTree
from chromorph.blocked import BlockedMetropolis,component_blocks
from chromorph.inference import PolyChordPosterior
from chromorph.noise import DenseCovariance
from conftest import observations,clump


def target_scene(renderer,correlated=False):
    obs=observations(size=13,noise=.02)
    if correlated:
        for o in obs:
            n=o.data.size;o.covariance=DenseCovariance(.02**2*(.97*np.eye(n)+.03*np.ones((n,n))))
    scene=Scene(obs,renderer=renderer,fit_background=True,fit_registration=True,fit_calibration=True)
    scene.add_component(PointSource('nucleus',FreeBandSED(),parameters={'x':Parameter(6.,3.,9.),'y':Parameter(6.,3.,9.)}))
    for name,x in [('a',5.),('b',7.)]:
        scene.add_component(clump(name,x=x,y=6.,moving=True,sed=FreeBandSED(parameters={b:Parameter(1.,0.,5.) for b in scene.bands})))
    # Background bounds are explicit simulation priors, not package defaults.
    from dataclasses import replace
    for name,p in list(scene.registry.parameters.items()):
        if name.endswith('.background'):scene.registry.parameters[name]=replace(p,lower=-1,upper=1)
    return scene


@pytest.mark.parametrize('correlated',[False,True])
def test_transactional_posterior_matches_full_and_rolls_back(renderer,correlated):
    with target_scene(renderer,correlated) as scene:
        target=ExactPosteriorTarget(scene,likelihood='full');x=scene.registry.initial(target.names)
        engine=TransactionalEvaluator(target);first=engine.initialize(x)
        for name,delta in [('a.morphology.x',.2),('a.flux.blue',.1),('nucleus.morphology.y',.1),
            ('observation.red.dx',.1),('observation.red.log_calibration',.01),('observation.blue.background',.001)]:
            y=x.copy();y[target.names.index(name)]+=delta
            accepted_before=engine.accepted;before=engine.metrics['render_calls']
            p=engine.propose(y);expected,conditional=target.evaluate(y)
            assert p.state.loglikelihood==pytest.approx(expected,abs=1e-8)
            assert p.logpdf==pytest.approx(target.logpdf(y),abs=1e-8)
            for n in target.linear:np.testing.assert_allclose(p.state.conditionals[n],conditional[n],rtol=1e-11,atol=1e-12)
            renders=engine.metrics['render_calls']-before
            assert renders== (len(scene.observations) if '.morphology.' in name else len(scene.components) if name.endswith('.dx') else 0)
            engine.rollback(p);assert engine.accepted is accepted_before
            p=engine.propose(y);engine.commit(p)
            p=engine.propose(x);engine.commit(p)
            assert engine.accepted.logpdf==first.logpdf
            for a,b in zip(first.frames,engine.accepted.frames):np.testing.assert_array_equal(a.residual,b.residual)
        p=engine.propose(x)
        with pytest.raises(RuntimeError,match='outstanding'):engine.propose(x)
        engine.rollback(p)
        with pytest.raises(RuntimeError,match='Stale'):engine.commit(p)
        invalid=x.copy();invalid[0]=1e99;p=engine.propose(invalid)
        assert p.logpdf==-np.inf;engine.rollback(p)
        scene.observations[0].update(data=scene.observations[0].data+.001,sigma=scene.observations[0].sigma*2)
        with pytest.raises(RuntimeError,match='Scientific inputs'):engine.propose(x)
        engine=TransactionalEvaluator(target);p=engine.propose(x)
        assert p.state.loglikelihood==pytest.approx(target.evaluate(x)[0],abs=1e-8);engine.rollback(p)


def test_coupled_transaction_and_priors_equal_existing_adapter(renderer):
    with target_scene(renderer) as scene:
        target=ExactPosteriorTarget(scene,likelihood='full');old=PolyChordPosterior(scene,likelihood='full')
        assert target.names==old.names and target.linear==old.linear and target.signature()==old.signature()
        x=scene.registry.initial(target.names);engine=TransactionalEvaluator(target);engine.initialize(x)
        y=x.copy()
        for i,n in enumerate(target.names):
            if '.morphology.' in n and n.endswith('.x'):y[i]+=.1
        p=engine.propose(y)
        assert engine.metrics['render_calls']==2*len(scene.components)*len(scene.observations)
        assert p.state.loglikelihood==pytest.approx(old.evaluate(y)[0],abs=1e-8)
        blocks=component_blocks(target)
        assert any(b.exchange for b in blocks) and any(b.name=='coupled.morphology' for b in blocks)
        assert set(i for b in blocks for i in b.indices)==set(range(len(target.names)))
        engine.rollback(p)


def test_fixed_order_tree_not_dependent_on_updates():
    rng=np.random.default_rng(6);leaves=[rng.normal(size=100)*10**j for j in range(7)]
    tree=SumTree.build(leaves);other=[x.copy() for x in leaves];other[2]*=1.7;other[5]*=.3
    sequential=tree.update({2:other[2]}).update({5:other[5]})
    simultaneous=tree.update({5:other[5],2:other[2]})
    np.testing.assert_array_equal(sequential.total,simultaneous.total)
    np.testing.assert_array_equal(sequential.total,SumTree.build(other).total)
    np.testing.assert_array_equal(sequential.update({2:leaves[2],5:leaves[5]}).total,tree.total)


def test_frozen_kernel_reference_and_incremental_accept_identical_stream(renderer):
    with target_scene(renderer) as scene:
        target=ExactPosteriorTarget(scene,likelihood='full');initial=scene.registry.initial(target.names)
        sampler=BlockedMetropolis(target)
        a=sampler.run(initial,draws=80,warmup=40,seed=9,transactional=True)
        b=sampler.run(initial,draws=80,warmup=40,seed=9,transactional=False)
        np.testing.assert_array_equal(a.accepted,b.accepted)
        np.testing.assert_allclose(a.samples,b.samples,rtol=0,atol=0)
        np.testing.assert_allclose(a.logposterior,b.logposterior,atol=1e-8,rtol=0)
        assert a.metadata['production_kernels_frozen'] and a.metadata['warmup_discarded']


def test_tied_geometry_changes_all_affected_columns(renderer):
    with target_scene(renderer) as scene:
        scene.tie('b.morphology.x','a.morphology.x')
        target=ExactPosteriorTarget(scene,likelihood='full');x=scene.registry.initial(target.names)
        e=TransactionalEvaluator(target);e.initialize(x);before=e.metrics['render_calls']
        y=x.copy();y[target.names.index('a.morphology.x')]+=.1
        p=e.propose(y)
        assert e.metrics['render_calls']-before==2*len(scene.observations)
        assert p.logpdf==pytest.approx(target.logpdf(y),abs=1e-8);e.rollback(p)
        expected=sum(d.logpdf(v) for d,v in zip(target._prior_transform.priors,x))
        assert target.logprior(x)==pytest.approx(expected,abs=1e-12)


def test_saved_blocked_chain_and_pocomc_conditionals(renderer,tmp_path,monkeypatch):
    from chromorph.blocked import BlockedResult
    from chromorph.population import PocoMCPosterior
    with target_scene(renderer) as scene:
        target=ExactPosteriorTarget(scene,likelihood='full');x=scene.registry.initial(target.names)
        result=BlockedMetropolis(target).run(x,draws=10,warmup=5,seed=8)
        result.save(tmp_path);restored=BlockedResult.load(tmp_path,target=target)
        np.testing.assert_array_equal(restored.samples,result.samples)
        np.testing.assert_array_equal(restored.derived,result.derived)
        cube=np.full(len(target.names),.5);adapter=PocoMCPosterior(target)
        vector=target.prior(cube);score,conditional=target.evaluate(vector)
        value=adapter.loglikelihood_with_conditionals(cube)
        assert value[0]==score
        np.testing.assert_array_equal(value[1:],[v for n in target.linear for v in conditional[n]])
        assert adapter.loglikelihood([cube])[0]==score
        monkeypatch.setattr(target,'evaluate',lambda *a:(_ for _ in ()).throw(AssertionError('Replayed')))
        target.restore_fluxes(restored.samples,derived=restored.derived,seed=9)


def test_vector_restoration_preserves_scalar_random_stream(renderer):
    from scipy.stats import truncnorm
    with target_scene(renderer) as scene:
        target=ExactPosteriorTarget(scene);n=100
        samples=np.tile(scene.registry.initial(target.names),(n,1))
        moments=np.tile([-1.,.2,2.,.1],(n,1));moments[:,0]=np.linspace(-3,3,n)
        rng=np.random.default_rng(73);expected=np.empty((n,2))
        for i in range(n):
            for j in range(2):
                mean,sigma=moments[i,2*j:2*j+2]
                expected[i,j]=truncnorm.rvs(-mean/sigma,np.inf,loc=mean,scale=sigma,random_state=rng)
        result=target.restore_fluxes(samples,derived=moments,seed=73)[1]
        np.testing.assert_array_equal(result[:,-2:],expected)
