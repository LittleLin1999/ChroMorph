"""Bounded variable projection using whitened QR factors, never G^-1."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear
from .sed import FreeBandSED, FSPSStellarSED


@dataclass(frozen=True)
class LinearAmplitude:
    name: str
    kind: str
    component: str | None
    band: str | None
    reference_log_mass: float = 0.


class LinearProjection:
    def __init__(self, scene, *, stellar_mass=True, background=True):
        self.scene = scene
        registry = scene.registry
        linked = set(registry.ties) | set(registry.ties.values())
        self.amplitudes = []
        for component in scene.components:
            if isinstance(component.sed, FreeBandSED):
                for band in scene.bands:
                    name = f'{component.name}.flux.{band}'
                    p = registry.parameters[name]
                    if p.vary and p.prior is None and name not in linked:
                        self.amplitudes.append(LinearAmplitude(name, 'flux', component.name, band))
            elif stellar_mass and isinstance(component.sed, FSPSStellarSED):
                name = f'{component.name}.sed.log_mass'
                p = registry.parameters[name]
                if p.vary and p.prior is None and name not in linked and np.isfinite([p.lower, p.upper]).all():
                    self.amplitudes.append(LinearAmplitude(name, 'mass', component.name, None, p.value))
        if background:
            for o in scene.observations:
                name = f'observation.{o.name}.background'
                p = registry.parameters[name]
                if p.vary and p.prior is None and name not in linked:
                    self.amplitudes.append(LinearAmplitude(name, 'background', None, o.name))
        self.names = [a.name for a in self.amplitudes]
        self.last_diagnostics = {}

    def solve(self, values):
        if not self.amplitudes:
            return dict(values)
        scene = self.scene
        engine = scene.compressed
        statistics = engine.statistics(values)
        coefficients = engine.coefficients(values)
        components = {c.name: i for i, c in enumerate(scene.components)}
        matrices, targets = [], []
        for observation in scene.observations:
            name = observation.name
            s = statistics[name]
            base = coefficients[name].copy()
            mapping = np.zeros((len(base), len(self.amplitudes)))
            calibration = np.exp(values[f'observation.{name}.log_calibration'])/observation.ujy_per_rate
            for j, amplitude in enumerate(self.amplitudes):
                if amplitude.kind == 'background' and amplitude.band == name:
                    mapping[-1, j] = 1.
                    base[-1] = 0.
                elif amplitude.kind == 'flux' and amplitude.band == observation.band:
                    i = components[amplitude.component]
                    mapping[i, j] = calibration
                    base[i] = 0.
                elif amplitude.kind == 'mass':
                    i = components[amplitude.component]
                    mapping[i, j] = coefficients[name][i]*10**(amplitude.reference_log_mass-values[amplitude.name])
                    base[i] = 0.
            matrices.append(s.R@mapping)
            targets.append(s.z-s.R@base)
        design, target = np.vstack(matrices), np.concatenate(targets)
        lower, upper = [], []
        for amplitude in self.amplitudes:
            p = scene.registry.parameters[amplitude.name]
            if amplitude.kind == 'mass':
                lower.append(10**(p.lower-amplitude.reference_log_mass))
                upper.append(10**(p.upper-amplitude.reference_log_mass))
            else:
                lower.append(p.lower);upper.append(p.upper)
        # Normalize columns before the bounded QR/SVD solve. This is a change
        # of numerical units, not a stellar-population or flux-ratio constraint.
        scale = np.linalg.norm(design, axis=0)
        scale[scale == 0] = 1.
        normalized = design/scale
        fit = lsq_linear(normalized, target, bounds=(np.array(lower)*scale, np.array(upper)*scale),
                         method='bvls', tol=1e-12, max_iter=300)
        if not fit.success:
            raise RuntimeError('Bounded amplitude solve failed: '+fit.message)
        singular = np.linalg.svd(normalized, compute_uv=False)
        rank = int(np.count_nonzero(singular > max(singular[0]*1e-10, 1e-14)))
        self.last_diagnostics = dict(rank=rank, n_amplitudes=len(self.amplitudes),
            condition_number=float(singular[0]/singular[-1]) if rank == len(self.amplitudes) else None,
            active_bounds=fit.active_mask.tolist(), optimality=float(fit.optimality))
        result = dict(values)
        for amplitude, value in zip(self.amplitudes, fit.x/scale):
            result[amplitude.name] = (amplitude.reference_log_mass+np.log10(value)
                                      if amplitude.kind == 'mass' else float(value))
        return result
