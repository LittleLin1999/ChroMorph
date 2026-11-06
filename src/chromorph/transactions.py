"""Transactional exact Gaussian image updates from the accepted state.

A persistent fixed-order sum tree updates only affected leaves/ancestors.
Unlike repeated residual += delta, its answer has no accumulated path drift.
Memoized reference spectra may survive rejection; accepted state never does.
"""
from dataclasses import dataclass
import time
import numpy as np
from .gaussian import GaussianAccumulator


def _readonly(array):
    array=np.asarray(array);array.setflags(write=False);return array


@dataclass(frozen=True)
class SumTree:
    nodes: tuple
    size: int

    @classmethod
    def build(cls,leaves):
        size=1
        while size<len(leaves):size*=2
        nodes=[None]*(2*size)
        zero=_readonly(np.zeros_like(leaves[0]))
        for i in range(size):nodes[size+i]=_readonly(leaves[i]) if i<len(leaves) else zero
        for i in range(size-1,0,-1):nodes[i]=_readonly(nodes[2*i]+nodes[2*i+1])
        return cls(tuple(nodes),size)

    def update(self,changes):
        nodes=list(self.nodes);parents=set()
        for index,array in changes.items():
            node=self.size+index;nodes[node]=_readonly(array)
            while node>1:node//=2;parents.add(node)
        for node in sorted(parents,reverse=True):nodes[node]=_readonly(nodes[2*node]+nodes[2*node+1])
        return SumTree(tuple(nodes),self.size)

    @property
    def total(self):return self.nodes[1]


@dataclass(frozen=True)
class Frame:
    input_token: tuple
    noise_token: tuple
    keys: tuple
    raw: tuple
    white: tuple
    amplitudes: tuple
    sky: float
    tree: SumTree
    residual: np.ndarray


@dataclass(frozen=True)
class PosteriorState:
    vector: np.ndarray
    values: dict
    frames: tuple
    loglikelihood: float
    logprior: float
    conditionals: dict

    @property
    def logpdf(self):return self.loglikelihood+self.logprior


@dataclass(frozen=True)
class Proposal:
    generation: int
    state: object
    spectral_state: object

    @property
    def logpdf(self):return -np.inf if self.state is None else self.state.logpdf


class TransactionalEvaluator:
    def __init__(self,target):
        self.target=target;self.e=target.scene.prepare();self.accepted=None;self.pending=None;self.generation=0
        self.metrics=dict(proposals=0,commits=0,rejections=0,render_calls=0,whitened_columns=0,
                          tree_leaf_updates=0,seconds=0.,invalid_prior=0)
        self.revision=target.scene.registry.revision
        self._binding=None

    def _spectral_snapshot(self):
        engine=self.e._spectral_engine
        return None if engine is None else dict(engine._unit)

    def _restore_spectral(self,snapshot):
        if self.e._spectral_engine is not None:
            self.e._spectral_engine._unit={} if snapshot is None else dict(snapshot)

    def propose(self,vector):
        if self.pending is not None:raise RuntimeError('Resolve the outstanding proposal first')
        if self.target.scene.registry.revision!=self.revision:raise RuntimeError('Parameter definitions changed during sampling')
        self.e._ensure_renderer()
        for o in self.e.observations:self.e.gaussian_data(o)
        binding=(self.e.context,self.e.renderer.cache_token() if hasattr(self.e.renderer,'cache_token') else id(self.e.renderer),
            tuple((o.likelihood_token(),o.psf.cache_token(),tuple(o.matrix.ravel()),tuple(o.offset),o.ujy_per_rate,
                getattr(o.throughput,'fingerprint',None)) for o in self.e.observations),
            tuple((id(c.sed),tuple(sorted(getattr(c.sed,'configuration',{}).items())),getattr(c.sed,'sfh',None),
                getattr(c.sed,'dust',None),getattr(c.sed,'nebular',None),getattr(c.sed,'experimental',None),
                id(getattr(c.sed,'library',None))) for c in self.e.components))
        if self._binding is None:self._binding=binding
        elif binding!=self._binding:raise RuntimeError('Scientific inputs changed; initialize a new transaction evaluator')
        started=time.perf_counter();self.metrics['proposals']+=1
        prior=self.target.logprior(vector);snapshot=self._spectral_snapshot()
        if not np.isfinite(prior):
            self.metrics['invalid_prior']+=1
            self.pending=Proposal(self.generation,None,snapshot);return self.pending
        try:
            state=self._evaluate(vector,prior)
        except BaseException:
            self._restore_spectral(snapshot);raise
        self.pending=Proposal(self.generation,state,snapshot)
        self.metrics['seconds']+=time.perf_counter()-started
        return self.pending

    def _evaluate(self,vector,prior):
        e=self.e;e._ensure_renderer();target=self.target
        values=e.registry.unpack(vector,target.names)
        for name in target.linear:values[name]=0.
        fluxes=e.component_fluxes(values);plans=e.nuclear_plan(target.linear)
        frames=[];accumulator=GaussianAccumulator()
        for oi,o in enumerate(e.observations):
            fixed=e.gaussian_data(o);old=None if self.accepted is None else self.accepted.frames[oi]
            keys=tuple(e.spatial_key(c,o,values) for c in e.components)
            raw=[];white=[];changes={}
            a,sky,cal=e.image_coefficients(o,values,fluxes)
            same_noise=old is not None and old.noise_token==fixed.noise_token
            for i,c in enumerate(e.components):
                same=old is not None and keys[i]==old.keys[i]
                if same:unit=old.raw[i]
                else:
                    unit=_readonly(e.renderer.render(c,e.morphology(c,values),o,e.registration(o,values)))
                    self.metrics['render_calls']+=1;e.metrics['render_calls']+=1
                if same and same_noise:column=old.white[i]
                else:
                    column=_readonly(e.whiten(o,unit.ravel()[fixed.included]));self.metrics['whitened_columns']+=1
                raw.append(unit);white.append(column)
                if old is None or not same or not same_noise or a[i]!=old.amplitudes[i]:changes[i]=a[i]*column
            if old is None or not same_noise or sky!=old.sky:
                changes[len(e.components)]=sky*e.whiten(o,np.ones(len(fixed.target)))
            if old is None or not same_noise:
                tree=SumTree.build([changes[i] for i in range(len(e.components)+1)])
            else:tree=old.tree.update(changes)
            residual=_readonly(fixed.target-tree.total)
            frame=Frame(fixed.token,fixed.noise_token,keys,tuple(raw),tuple(white),tuple(a),sky,tree,residual)
            frames.append(frame);self.metrics['tree_leaf_updates']+=len(changes)
            parameter=plans.get(o.band);nuclear=None
            if parameter is not None:
                name=parameter.split('.')[0];i=next(i for i,c in enumerate(e.components) if c.name==name)
                nuclear=white[i]*cal
            accumulator.add(residual,fixed.whitening.log_normalization,parameter=parameter,column=nuclear)
        answer=accumulator.finish()
        return PosteriorState(_readonly(np.array(vector,copy=True)),values,tuple(frames),answer.loglikelihood,prior,answer.conditionals)

    def commit(self,proposal):
        self._check(proposal)
        if proposal.state is None:raise ValueError('Cannot accept an invalid-prior proposal')
        self.accepted=proposal.state;self.pending=None;self.generation+=1;self.metrics['commits']+=1
        return self.accepted

    def rollback(self,proposal):
        self._check(proposal);self._restore_spectral(proposal.spectral_state)
        self.pending=None;self.metrics['rejections']+=1

    def _check(self,proposal):
        if proposal is not self.pending or proposal.generation!=self.generation:raise RuntimeError('Stale or foreign transaction')

    def initialize(self,vector):return self.commit(self.propose(vector))
