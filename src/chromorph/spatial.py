"""Independent reference-grid morphology, shared across observations."""
from __future__ import annotations
from dataclasses import dataclass, field
from .parameters import Parameter, identifier


@dataclass
class PointSource:
    name: str
    sed: object
    x: float = 0.0
    y: float = 0.0
    nuclear: bool = False
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.name)
        self.parameters = dict(self.parameters)
        self.parameters.setdefault("x", Parameter(self.x))
        self.parameters.setdefault("y", Parameter(self.y))
        if set(self.parameters) != {"x", "y"}:
            raise ValueError("Point morphology accepts x and y only")
        if self.nuclear:
            from .sed import FreeBandSED
            if type(self.sed) is not FreeBandSED:
                raise ValueError("The nuclear point source must use unconstrained FreeBandSED")


@dataclass
class Sersic:
    name: str
    sed: object
    x: float = 0.0
    y: float = 0.0
    re: float = 2.0  # intrinsic semimajor effective radius, reference pixels
    n: float = 1.0
    q: float = 1.0
    pa: float = 0.0  # degrees counterclockwise from reference +x
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.name)
        self.parameters = dict(self.parameters)
        for key, value, lo, hi in [("x", self.x, -float("inf"), float("inf")),
                                    ("y", self.y, -float("inf"), float("inf")),
                                    ("re", self.re, .02, 1000), ("n", self.n, .3, 6.2),
                                    ("q", self.q, .05, 1.0), ("pa", self.pa, -360, 360)]:
            self.parameters.setdefault(key, Parameter(value, lo, hi))
        if set(self.parameters) != {"x", "y", "re", "n", "q", "pa"}:
            raise ValueError("Unknown Sérsic morphology parameter")
        if self.parameters['re'].lower <= 0 or self.parameters['q'].lower <= 0 or self.parameters['q'].upper > 1 or self.parameters['n'].lower < .3 or self.parameters['n'].upper > 6.2:
            raise ValueError("Sersic bounds require re>0, 0<q<=1, .3<=n<=6.2 (GalSim)")
