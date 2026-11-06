"""Normalized scalar priors; bounds and initial guesses have distinct roles."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.stats import uniform, truncnorm


@dataclass(frozen=True)
class Prior:
    distribution: str
    mean: float = 0.
    sigma: float = 1.
    lower: float = -np.inf
    upper: float = np.inf
    value: float | None = None

    def __post_init__(self):
        names={'gaussian':'normal','truncated_gaussian':'normal','truncated_normal':'normal'}
        object.__setattr__(self,'distribution',names.get(self.distribution,self.distribution))
        if self.distribution not in {'uniform','normal','lognormal','fixed'}:raise ValueError('Unknown prior distribution')
        if self.lower>=self.upper:raise ValueError('Prior support requires distinct bounds')
        if self.distribution in {'normal','lognormal'} and (not np.isfinite([self.mean,self.sigma]).all() or self.sigma<=0):
            raise ValueError('Prior mean and positive sigma must be finite')
        if self.distribution=='fixed' and (self.value is None or not np.isfinite(self.value)):raise ValueError('Fixed prior needs a finite value')


def prior_spec(value):
    if isinstance(value,Prior):return value
    if isinstance(value,dict):
        row=dict(value)
        if 'log_mean' in row:row['mean']=row.pop('log_mean')
        if 'log_sigma' in row:row['sigma']=row.pop('log_sigma')
        return Prior(**row)
    if value is None:return Prior('uniform')
    if len(value)!=2:raise ValueError('Gaussian prior requires mean and sigma')
    return Prior('normal',float(value[0]),float(value[1]))


class ScalarDistribution:
    def __init__(self,spec,lower,upper):
        self.spec=spec
        lo,hi=max(lower,spec.lower),min(upper,spec.upper)
        if lo>=hi:raise ValueError('Prior and numerical bounds have disjoint support')
        self.lower,self.upper=lo,hi
        if spec.distribution=='uniform':
            if not np.isfinite([lo,hi]).all():raise ValueError('A proper uniform prior requires finite bounds')
            self.frozen=uniform(lo,hi-lo);self.mode=(lo+hi)/2
        elif spec.distribution=='fixed':
            if not lo<=spec.value<=hi:raise ValueError('Fixed prior outside bounds')
            self.mode=spec.value
        else:
            if spec.distribution=='lognormal':
                if hi<=0:raise ValueError('Lognormal prior requires positive support')
                lo=max(0.,lo);a=-np.inf if lo==0 else np.log(lo);b=np.log(hi)
                self.mode=float(np.clip(np.exp(spec.mean-spec.sigma**2),lo,hi))
            else:
                a,b=lo,hi;self.mode=float(np.clip(spec.mean,lo,hi))
            self.frozen=truncnorm((a-spec.mean)/spec.sigma,(b-spec.mean)/spec.sigma,loc=spec.mean,scale=spec.sigma)

    def ppf(self,cube):
        if self.spec.distribution=='fixed':return np.full_like(np.asarray(cube,dtype=float),self.spec.value)
        x=self.frozen.ppf(cube)
        return np.exp(x) if self.spec.distribution=='lognormal' else x

    def logpdf(self,value):
        if self.spec.distribution=='fixed':return np.where(np.asarray(value)==self.spec.value,0.,-np.inf)
        if self.spec.distribution=='lognormal':
            x=np.asarray(value,dtype=float)
            with np.errstate(divide='ignore',invalid='ignore'):
                return np.where(x>0,self.frozen.logpdf(np.log(x))-np.log(x),-np.inf)
        return self.frozen.logpdf(value)

    def residual(self,value):
        if self.spec.distribution=='normal' and self.mode==self.spec.mean:
            return float((value-self.spec.mean)/self.spec.sigma)
        if self.spec.distribution=='lognormal' and self.mode==np.exp(self.spec.mean-self.spec.sigma**2):
            return float((np.log(value)-self.spec.mean+self.spec.sigma**2)/self.spec.sigma)
        penalty=2*(self.logpdf(self.mode)-self.logpdf(value))
        return float(np.sign(value-self.mode)*np.sqrt(max(0.,penalty)))


@lru_cache(maxsize=4096)
def distribution(spec,lower,upper):
    return ScalarDistribution(spec,float(lower),float(upper))


def prior_for(parameter,lower=None,upper=None):
    return distribution(prior_spec(parameter.prior),parameter.lower if lower is None else lower,
                        parameter.upper if upper is None else upper)


class CubePriorTransform:
    """Vectorized normalized independent transforms, with stable tail fallback."""
    def __init__(self,registry,names):
        from scipy.special import ndtr
        self.names=list(names);self.lower,self.upper=registry.bounds(names)
        self.priors=[prior_for(registry.parameters[n],lo,hi) for n,lo,hi in zip(names,self.lower,self.upper)]
        self.uniform=[];self.normal=[];self.fallback=[]
        self.lo=np.zeros(len(names));self.span=np.ones(len(names));self.mean=np.zeros(len(names));self.sigma=np.ones(len(names));self.exp=np.zeros(len(names),dtype=bool)
        for i,d in enumerate(self.priors):
            spec=d.spec
            if spec.distribution=='uniform':
                self.uniform.append(i);self.lo[i]=d.lower;self.span[i]=d.upper-d.lower
            elif spec.distribution in {'normal','lognormal'}:
                lo,hi=d.lower,d.upper
                if spec.distribution=='lognormal':lo=-np.inf if lo<=0 else np.log(lo);hi=np.log(hi);self.exp[i]=True
                a=ndtr((lo-spec.mean)/spec.sigma);b=ndtr((hi-spec.mean)/spec.sigma)
                if b-a<1e-10:self.fallback.append(i)
                else:
                    self.normal.append(i);self.lo[i]=a;self.span[i]=b-a;self.mean[i]=spec.mean;self.sigma[i]=spec.sigma
            else:self.fallback.append(i)

    def __call__(self,cube):
        from scipy.special import ndtri
        cube=np.asarray(cube,dtype=float)
        if cube.shape!=(len(self.names),) or not np.isfinite(cube).all() or np.any((cube<0)|(cube>1)):
            raise ValueError('Invalid prior unit cube')
        cube=np.clip(cube,np.finfo(float).eps,1-np.finfo(float).eps)
        result=np.empty_like(cube)
        u=np.array(self.uniform,dtype=int);g=np.array(self.normal,dtype=int)
        result[u]=self.lo[u]+cube[u]*self.span[u]
        probability=np.clip(self.lo[g]+cube[g]*self.span[g],np.finfo(float).tiny,1-np.finfo(float).eps)
        result[g]=self.mean[g]+self.sigma[g]*ndtri(probability)
        mask=self.exp[g];result[g[mask]]=np.exp(result[g[mask]])
        for i in self.fallback:result[i]=self.priors[i].ppf(cube[i])
        return result
