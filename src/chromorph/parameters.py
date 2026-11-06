"""Named parameters with explicit ties; no implicit component coupling."""
from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np


@dataclass(frozen=True)
class Parameter:
    value: float
    lower: float = -np.inf
    upper: float = np.inf
    vary: bool = True
    prior: object = None

    def __post_init__(self):
        if not np.isfinite(self.value) or self.lower > self.value or self.upper < self.value:
            raise ValueError("Parameter value must be finite and within bounds")
        if self.vary and self.lower >= self.upper:
            raise ValueError("A varying parameter needs distinct bounds")
        if self.prior is not None:
            from .priors import prior_spec, prior_for, Prior
            spec=prior_spec(self.prior)
            if isinstance(self.prior,(dict,Prior)):
                object.__setattr__(self,'prior',spec)
            else:
                object.__setattr__(self,'prior',tuple(self.prior))
            if spec.distribution=='fixed':
                if self.value!=spec.value:raise ValueError('Initial and fixed-prior values disagree')
                object.__setattr__(self,'vary',False)
            if not np.isfinite(prior_for(self).logpdf(self.value)):
                raise ValueError('Initial value outside prior support')


def identifier(name):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid identifier: {name!r}")
    return name


class Registry:
    def __init__(self):
        from .layout import DefinitionDict
        self._revision = 0
        self._layout = None
        self._parameters = DefinitionDict(changed=self._changed)
        self._ties = DefinitionDict(changed=self._changed)

    @property
    def revision(self): return self._revision

    def _changed(self):
        self._revision += 1
        self._layout = None

    @property
    def parameters(self): return self._parameters

    @parameters.setter
    def parameters(self, value):
        from .layout import DefinitionDict
        self._parameters = DefinitionDict(value, self._changed)
        self._changed()

    @property
    def ties(self): return self._ties

    @ties.setter
    def ties(self, value):
        from .layout import DefinitionDict
        self._ties = DefinitionDict(value, self._changed)
        self._changed()

    @property
    def layout(self):
        from .layout import ParameterLayout
        if self._layout is None: self._layout = ParameterLayout(self)
        return self._layout

    def __getstate__(self):
        return dict(parameters=dict(self.parameters), ties=dict(self.ties))

    def __setstate__(self, state):
        self.__init__()
        self.parameters = state.get('parameters',state.get('_parameters',{}))
        self.ties = state.get('ties',state.get('_ties',{}))

    def add(self, name, parameter):
        if name in self.parameters:
            raise ValueError(f"Duplicate parameter: {name}")
        self.parameters[name] = parameter

    def resolve(self, name):
        seen = set()
        while name in self.ties:
            if name in seen:
                raise ValueError("Cyclic parameter tie")
            seen.add(name)
            name = self.ties[name]
        return name

    def tie(self, name, target):
        if name == target or name not in self.parameters or target not in self.parameters:
            raise ValueError("A tie requires two different registered parameters")
        if self.parameters[name].prior is not None:
            raise ValueError("Put a tied parameter's prior on its root explicitly")
        old = self.ties.copy()
        self.ties[name] = target
        try:
            self.resolve(name)
            for root in {self.resolve(n) for n in self.parameters}:
                members = [p for n, p in self.parameters.items() if self.resolve(n) == root]
                lo, hi = max(p.lower for p in members), min(p.upper for p in members)
                p = self.parameters[root]
                if lo > hi or not lo <= p.value <= hi or (p.vary and lo == hi):
                    raise ValueError(f"Incompatible tied parameter bounds for {root}")
            self.bounds()
        except ValueError:
            self.ties = old
            raise

    def names(self, exclude=()):
        excluded=set(exclude)
        return [n for n in self.layout.independent_names if n not in excluded]

    def bounds(self, names=None):
        return self.layout.bounds(self.layout.independent_names if names is None else names)

    def initial(self, names=None):
        return np.array([self.parameters[n].value for n in (self.names() if names is None else names)])

    def unpack(self, vector=None, names=None): return self.layout.unpack(vector,names)
    def prior_residuals(self, values): return self.layout.prior_residuals(values)
    def logprior(self, values): return self.layout.logprior(values)

    def cost_groups(self, names=None):
        names=self.layout.independent_names if names is None else names
        groups=[[],[],[]]
        for name in names: groups[self.layout.costs[name]].append(name)
        return dict(zip(['slow','medium','fast'],groups))
