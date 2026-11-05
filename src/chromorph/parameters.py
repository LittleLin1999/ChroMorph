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
    prior: tuple[float, float] | None = None

    def __post_init__(self):
        if not np.isfinite(self.value) or self.lower > self.value or self.upper < self.value:
            raise ValueError("Parameter value must be finite and within bounds")
        if self.vary and self.lower >= self.upper:
            raise ValueError("A varying parameter needs distinct bounds")
        if self.prior is not None:
            if len(self.prior) != 2 or not np.isfinite(self.prior).all() or self.prior[1] <= 0:
                raise ValueError("Gaussian prior must have finite mean and positive sigma")


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
        return np.array([(values[n] - p.prior[0]) / p.prior[1]
                         for n, p in self.parameters.items()
                         if n not in self.ties and p.prior is not None])
