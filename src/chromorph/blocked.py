"""Experimental reversible random-scan blocked Metropolis-within-Gibbs.

Proposal adaptation is confined to discarded warmup. Production uses fixed
symmetric Gaussian or involutive exchange kernels, with full joint MH ratios.
No parameter is profiled, fixed or removed except exact nuclear integration.
"""
from dataclasses import dataclass
import time
import numpy as np
from .transactions import TransactionalEvaluator


@dataclass(frozen=True)
class Block:
    name: str
    indices: tuple
    exchange: tuple = ()
    period: float = 0.


def component_blocks(target):
    """Every independent root is covered, including tied cross-component roots."""
    names=target.names;groups={};covered=set();morph=[];component_indices={}
    for i,name in enumerate(names):
        parts=name.split('.')
        if len(parts)>2 and parts[1]=='morphology':key=parts[0]+'.morphology';morph.append(i)
        elif len(parts)>2 and parts[1] in {'sed','flux'}:key=parts[0]+'.sed'
        elif name.startswith('observation.'):
            key='shared.registration' if parts[-1] in {'dx','dy'} else 'shared.background' if parts[-1]=='background' else 'shared.calibration'
        else:key='shared.other'
        groups.setdefault(key,[]).append(i);covered.add(i)
        if parts[0]!='observation':component_indices.setdefault(parts[0],{})['.'.join(parts[1:])]=i
    blocks=[Block(key,tuple(indices)) for key,indices in groups.items()]
    if len(morph)>1:blocks.append(Block('coupled.morphology',tuple(morph)))
    for i,name in enumerate(names):
        if name.endswith('.morphology.pa'):blocks.append(Block('symmetry.'+name,(i,),period=180.))
    # Pairwise involutions are symmetric even with different component priors:
    # the complete target ratio decides acceptance. Skip overlapping tied roots.
    from .spatial import Sersic
    components=[c.name for c in target.scene.components if isinstance(c,Sersic)]
    for ai,a in enumerate(components):
        for b in components[ai+1:]:
            aa,bb=component_indices.get(a,{}),component_indices.get(b,{})
            if aa and aa.keys()==bb.keys():
                left=tuple(aa[k] for k in sorted(aa));right=tuple(bb[k] for k in sorted(aa))
                if not set(left)&set(right):blocks.append(Block('exchange.'+a+'.'+b,left,right))
    for name in components:
        indices=tuple(component_indices.get(name,{}).values())
        if indices:blocks.append(Block('joint.'+name,indices))
    if covered!=set(range(len(names))):raise ValueError('Incomplete parameter blocking')
    return blocks


@dataclass
class BlockedResult:
    names: list
    samples: np.ndarray
    logposterior: np.ndarray
    derived: np.ndarray
    accepted: np.ndarray
    block_ids: np.ndarray
    metadata: dict

    def save(self,path):
        from pathlib import Path
        import json
        from .provenance import json_value
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'chain.npz',samples=self.samples,logposterior=self.logposterior,
            derived=self.derived,accepted=self.accepted,block_ids=self.block_ids,names=np.asarray(self.names))
        (path/'chain.json').write_text(json.dumps(json_value(self.metadata),indent=2,allow_nan=False)+'\n')


    @classmethod
    def load(cls,path,*,target=None):
        from pathlib import Path
        import json
        path=Path(path);metadata=json.loads((path/'chain.json').read_text())
        if metadata.get('chain_schema_version')!=1:raise ValueError('Incompatible blocked chain schema')
        with np.load(path/'chain.npz',allow_pickle=False) as data:
            names=data['names'].tolist()
            result=cls(names,data['samples'],data['logposterior'],data['derived'],data['accepted'],data['block_ids'],metadata)
        if result.samples.shape!=(len(result.logposterior),len(names)):raise ValueError('Chain columns are inconsistent')
        if result.derived.shape!=(len(result.samples),len(metadata['derived_names'])):raise ValueError('Conditional columns are inconsistent')
        if target is not None and (names!=target.names or metadata['model_signature']!=target.signature()):
            raise ValueError('Chain target/prior/input signature mismatch')
        return result


class BlockedMetropolis:
    def __init__(self,target,*,scales=None,coupled_weight=.1,exchange_weight=.05,joint_component_weight=0.):
        if not target.names:raise ValueError('Use the analytic conditional for a target with no sampled parameters')
        self.target=target;self.blocks=component_blocks(target)
        if not 0<coupled_weight<1 or not 0<exchange_weight<1:raise ValueError('Proposal weights must be positive and below one')
        if not 0<=joint_component_weight<1:raise ValueError('Joint component proposal weight must be in [0,1)')
        self.scales=np.asarray(scales if scales is not None else self.default_scales(target),dtype=float)
        if self.scales.shape!=(len(target.names),) or not np.isfinite(self.scales).all() or np.any(self.scales<=0):raise ValueError('Positive scale required for every sampled parameter')
        weights=np.array([joint_component_weight if b.name.startswith('joint.') else exchange_weight if b.exchange or b.period else coupled_weight if b.name=='coupled.morphology' else 1. for b in self.blocks])
        self.weights=weights/weights.sum();self.joint_component_weight=joint_component_weight

    @staticmethod
    def default_scales(target):
        from .objectives import parameter_scales
        initial=target.scene.registry.initial(target.names)
        # Proposal tuning only. It does not alter the prior or numerical bounds.
        return .1*parameter_scales(target.names,initial)

    def run(self,initial,*,draws=1000,warmup=500,seed=0,max_wall_seconds=None,transactional=True):
        if draws<1 or warmup<0:raise ValueError('Positive draws and nonnegative warmup required')
        rng=np.random.default_rng(seed);target=self.target;started=time.perf_counter();cpu=time.process_time()
        engine=TransactionalEvaluator(target)
        state=engine.initialize(initial);dimensions=len(target.names);nblocks=len(self.blocks)
        factors=[np.diag(self.scales[list(b.indices)]) for b in self.blocks]
        logsteps=np.zeros(nblocks);attempts=np.zeros(nblocks,dtype=int);accepts=np.zeros(nblocks,dtype=int)
        block_wall=np.zeros(nblocks);block_renders=np.zeros(nblocks,dtype=int)
        history=[];records=[];scores=[];conditionals=[];accepted=[];block_ids=[];frozen=False
        for step in range(warmup+draws):
            if max_wall_seconds is not None and time.perf_counter()-started>=max_wall_seconds:break
            if step==warmup:
                # Fixed shrinkage covariance from discarded joint warmup states.
                # Subsequent draws obey detailed balance for the frozen kernels.
                if len(history)>2:
                    history_array=np.asarray(history)
                    for j,b in enumerate(self.blocks):
                        if b.exchange or b.period or self.weights[j]==0:continue
                        ids=list(b.indices);base=np.diag(self.scales[ids]**2)
                        empirical=np.atleast_2d(np.cov(history_array[:,ids],rowvar=False))
                        covariance=.8*empirical+.2*base
                        factors[j]=np.linalg.cholesky(covariance)*2.38/np.sqrt(len(ids))
                logsteps[:]=0.;frozen=True
            bid=int(rng.choice(nblocks,p=self.weights));block=self.blocks[bid];ids=list(block.indices)
            trial=state.vector.copy()
            if block.period:
                trial[ids]+=rng.choice([-block.period,block.period])
            elif block.exchange:
                other=list(block.exchange);trial[ids]=state.vector[other];trial[other]=state.vector[ids]
            else:trial[ids]+=np.exp(logsteps[bid])*(factors[bid]@rng.normal(size=len(ids)))
            attempts[bid]+=1
            block_started=time.perf_counter();renders_before=target.scene.metrics['render_calls']
            if transactional:
                proposal=engine.propose(trial);candidate=proposal.logpdf
            else:
                prior=target.logprior(trial)
                answer=None if not np.isfinite(prior) else target.evaluate(trial)
                candidate=-np.inf if answer is None else prior+answer[0]
            take=bool(np.log(rng.uniform())<candidate-state.logpdf)
            if transactional:
                if take:state=engine.commit(proposal)
                else:engine.rollback(proposal)
            elif take:
                # Full reference path uses the same posterior and proposal stream.
                from .transactions import PosteriorState
                state=PosteriorState(trial,{},(),answer[0],prior,answer[1])
            block_wall[bid]+=time.perf_counter()-block_started
            block_renders[bid]+=target.scene.metrics['render_calls']-renders_before
            if take:accepts[bid]+=1
            if step<warmup:
                history.append(state.vector.copy())
                if not block.exchange and not block.period:
                    rate=(10+attempts[bid])**-.6
                    logsteps[bid]=np.clip(logsteps[bid]+rate*(float(take)-.234),-6.,6.)
            else:
                records.append(state.vector.copy());scores.append(state.logpdf);accepted.append(take);block_ids.append(bid)
                conditionals.append([x for name in target.linear for x in state.conditionals[name]])
        elapsed=time.perf_counter()-started
        metadata=dict(sampler='experimental_blocked_metropolis',chain_schema_version=1,kernel_version=2,joint_component_weight=self.joint_component_weight,derived_names=target.derived_names,marginalized_parameters=target.linear,model_signature=target.signature(),seed=seed,
            requested_draws=draws,completed_draws=len(records),warmup_requested=warmup,warmup_completed=min(len(history),warmup),
            warmup_discarded=True,production_kernels_frozen=frozen,transactional=transactional,
            blocks=[dict(name=b.name,parameters=[target.names[i] for i in b.indices],exchange=[target.names[i] for i in b.exchange],period=b.period,weight=float(w)) for b,w in zip(self.blocks,self.weights)],
            attempts=attempts.tolist(),accepts=accepts.tolist(),block_wall_seconds=block_wall.tolist(),block_render_calls=block_renders.tolist(),wall_seconds=elapsed,main_process_cpu_seconds=time.process_time()-cpu,
            transaction_metrics=dict(engine.metrics),evidence_defined=target.evidence_defined,evidence_estimated=False,convergence_validated=False,
            likelihood='normalized Gaussian with exact eligible nuclear integral',density='registered parameter coordinates')
        return BlockedResult(list(target.names),np.asarray(records).reshape((-1,dimensions)),np.asarray(scores),
            np.asarray(conditionals).reshape((len(records),2*len(target.linear))),np.asarray(accepted),np.asarray(block_ids),metadata)
