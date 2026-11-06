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
        self.parameters = {}
        self.ties = {}

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
        return [n for n, p in self.parameters.items()
                if p.vary and n not in self.ties and n not in exclude]

    def bounds(self, names=None):
        names = self.names() if names is None else names
        lower, upper = [], []
        for name in names:
            tied = [p for n, p in self.parameters.items() if self.resolve(n) == name]
            lo, hi = max(p.lower for p in tied), min(p.upper for p in tied)
            from .priors import prior_spec
            specs=[prior_spec(p.prior) for p in tied if p.prior is not None]
            if specs:
                lo=max([lo]+[p.lower for p in specs]);hi=min([hi]+[p.upper for p in specs])
                if any(p.distribution=='lognormal' for p in specs):lo=max(0.,lo)
            if lo >= hi or not lo <= self.parameters[name].value <= hi:
                raise ValueError(f"Incompatible bounds for tied parameter {name}")
            lower.append(lo); upper.append(hi)
        return np.array(lower), np.array(upper)

    def initial(self, names=None):
        return np.array([self.parameters[n].value for n in (self.names() if names is None else names)])

    def unpack(self, vector=None, names=None):
        values = {n: p.value for n, p in self.parameters.items()}
        if vector is not None:
            names = self.names() if names is None else names
            if len(vector) != len(names):
                raise ValueError("Wrong parameter vector length")
            values.update(zip(names, map(float, vector)))
        for name in self.ties:
            values[name] = values[self.resolve(name)]
        return values

    def prior_residuals(self, values):
        from .priors import prior_for
        names=[n for n,p in self.parameters.items() if n not in self.ties and p.vary and p.prior is not None]
        if not names:return np.array([])
        lo,hi=self.bounds(names)
        return np.array([prior_for(self.parameters[n],a,b).residual(values[n]) for n,a,b in zip(names,lo,hi)])

    def logprior(self, values):
        from .priors import prior_for
        names=self.names();lo,hi=self.bounds(names)
        return float(sum(prior_for(self.parameters[n],a,b).logpdf(values[n]) for n,a,b in zip(names,lo,hi)))

    def cost_groups(self, names=None):
        """Highest dependency cost wins for explicitly tied parameters."""
        names=self.names() if names is None else names
        def cost(n):
            if '.morphology.' in n or n.endswith(('.dx','.dy')):return 0
            if '.sed.' in n and not n.endswith('.log_mass'):return 1
            return 2
        groups=[[],[],[]]
        for n in names:
            grade=min(cost(alias) for alias in self.parameters if self.resolve(alias)==n)
            groups[grade].append(n)
        return dict(zip(['slow','medium','fast'],groups))
