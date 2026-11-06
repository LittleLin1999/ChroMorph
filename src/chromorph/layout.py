"""Compiled parameter ordering and independent/tied dependency groups."""
from dataclasses import dataclass
import numpy as np
from .priors import prior_spec, prior_for


class DefinitionDict(dict):
    """A small mapping whose explicit edits invalidate prepared definitions."""
    def __init__(self, values=(), changed=None):
        super().__init__(values);self.changed=changed
    def _changed(self):
        if self.changed is not None:self.changed()
    def __setitem__(self,key,value):super().__setitem__(key,value);self._changed()
    def __delitem__(self,key):super().__delitem__(key);self._changed()
    def update(self,*args,**kwargs):super().update(*args,**kwargs);self._changed()
    def clear(self):super().clear();self._changed()
    def pop(self,*args):value=super().pop(*args);self._changed();return value
    def popitem(self):value=super().popitem();self._changed();return value
    def setdefault(self,key,default=None):
        if key not in self:self[key]=default
        return self[key]
    def __ior__(self,other):self.update(other);return self


class ParameterLayout:
    """One immutable layout per registry revision; no per-proposal tie search."""
    def __init__(self, registry):
        self.revision=registry.revision
        self.all_names=tuple(registry.parameters)
        self.indices={n:i for i,n in enumerate(self.all_names)}
        self.roots={n:registry.resolve(n) for n in self.all_names}
        self.aliases=tuple((n,root) for n,root in self.roots.items() if n!=root)
        self.independent_names=tuple(n for n,p in registry.parameters.items() if p.vary and n not in registry.ties)
        self.defaults={n:p.value for n,p in registry.parameters.items()}
        self.groups={}
        for n,root in self.roots.items():self.groups.setdefault(root,[]).append(n)
        self.support={};self.parameters=registry.parameters
        for root,members in self.groups.items():
            parameters=[registry.parameters[n] for n in members]
            lo,hi=max(p.lower for p in parameters),min(p.upper for p in parameters)
            specifications=[prior_spec(p.prior) for p in parameters if p.prior is not None]
            if specifications:
                lo=max([lo]+[p.lower for p in specifications]);hi=min([hi]+[p.upper for p in specifications])
                if any(p.distribution=='lognormal' for p in specifications):lo=max(0.,lo)
            p=registry.parameters[root]
            if root in self.independent_names and (lo>=hi or not lo<=p.value<=hi):
                raise ValueError('Incompatible bounds for tied parameter '+root)
            self.support[root]=(float(lo),float(hi))
        def cost(n):
            if '.morphology.' in n or n.endswith(('.dx','.dy')):return 0
            if '.sed.' in n and not n.endswith('.log_mass'):return 1
            return 2
        self.costs={root:min(cost(n) for n in members) for root,members in self.groups.items()}
        self.prior_terms=[]
        for n in self.independent_names:
            p=registry.parameters[n]
            if p.prior is not None:self.prior_terms.append((n,prior_for(p,*self.support[n])))

    def bounds(self,names):
        support=[self.support[self.roots[n]] for n in names]
        if not support:return np.array([]),np.array([])
        lower,upper=zip(*support)
        return np.array(lower),np.array(upper)

    def unpack(self,vector=None,names=None):
        values=self.defaults.copy()
        if vector is not None:
            names=self.independent_names if names is None else names
            if len(vector)!=len(names):raise ValueError('Wrong parameter vector length')
            values.update(zip(names,map(float,vector)))
        for name,root in self.aliases:values[name]=values[root]
        return values

    def prior_residuals(self,values):
        return np.array([distribution.residual(values[name]) for name,distribution in self.prior_terms])

    def logprior(self,values):
        return float(sum(prior_for(self.parameters[n],*self.support[n]).logpdf(values[n]) for n in self.independent_names))
