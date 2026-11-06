"""Isolated process workers for SciPy's numerical Jacobian evaluations."""
import copy
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor

_SCENE = None
_NAMES = None
_ELIGIBLE = None


class PrecomputedSED:
    """Read-only component flux answers; never starts nested FSPS workers."""
    def __init__(self):
        self.answer = None

    def fluxes(self, values, name, observations, context):
        if self.answer is None:
            raise RuntimeError('Missing precomputed component fluxes')
        return dict(self.answer)

    def close(self):
        pass


def _initialize(scene, names, eligible):
    global _SCENE, _NAMES, _ELIGIBLE
    from multiprocessing.util import Finalize
    _SCENE, _NAMES, _ELIGIBLE = scene, names, eligible
    from .sed import FreeBandSED
    for component in scene.components:
        if not isinstance(component.sed, FreeBandSED):
            component.sed = PrecomputedSED()
    Finalize(None, scene.close, exitpriority=10)


def _evaluate(task):
    vector, flux_answers = task
    for component in _SCENE.components:
        if isinstance(component.sed, PrecomputedSED):
            component.sed.answer = flux_answers[component.name]
    values = _SCENE.registry.unpack(vector, _NAMES)
    basis = _SCENE.basis(values)
    values = _SCENE.project(values, _ELIGIBLE, basis)
    return _SCENE.residual(values, basis=basis)


class JacobianWorkers:
    """Parallel rendering with serial, independently owned spectral backends.

    Computing exact spectral answers before dispatch avoids nested process
    pools, duplicated FSPS initialization and problematic worker cleanup.
    """
    def __init__(self, scene, names, eligible, workers):
        from .rendering import GalSimRenderer
        snapshot = copy.copy(scene)
        self.scene = scene
        self.names = names
        snapshot._basis_cache = scene._basis_cache.empty_like() if hasattr(scene._basis_cache, 'empty_like') else {}
        snapshot._compressed = None
        self.pool = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
            initargs=(snapshot, names, eligible))
        self.validated = False

    def __call__(self, function, vectors):
        vectors = list(vectors)
        from .sed import FreeBandSED
        tasks = []
        for vector in vectors:
            values = self.scene.registry.unpack(vector, self.names)
            answers = {c.name: c.sed.fluxes(values, c.name, self.scene.observations, self.scene.context)
                       for c in self.scene.components if not isinstance(c.sed, FreeBandSED)}
            tasks.append((vector, answers))
        results = self.pool.map(_evaluate, tasks)
        if not self.validated and vectors and function is not None:
            checks = {0}
            checks.update(i for i, n in enumerate(self.names)
                          if n.endswith('.sed.age') and i < len(vectors))
            def checked():
                for i, result in enumerate(results):
                    if i in checks:
                        np.testing.assert_allclose(result, function(vectors[i]), rtol=1e-10, atol=1e-10,
                            err_msg='Spawned Jacobian worker does not reproduce the main likelihood')
                    yield result
                self.validated = True
            return checked()
        return results

    def close(self):
        self.pool.shutdown(wait=True)
