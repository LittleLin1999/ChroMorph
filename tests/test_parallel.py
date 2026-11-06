import numpy as np
from chromorph import Scene
from chromorph.parallel import JacobianWorkers
from conftest import observations, clump


def test_spawned_workers_match_global_likelihood(renderer):
    scene = Scene(observations(), renderer=renderer)
    scene.add_component(clump('one', moving=True))
    scene.add_component(clump('two', x=17., moving=True))
    models, _, _ = scene.predict()
    rng = np.random.default_rng(17)
    for observation in scene.observations:
        observation.data[:] = models[observation.name]+rng.normal(0, .001, observation.data.shape)
    eligible = scene._eligible_linear()
    names = scene.registry.names(exclude=eligible)
    vector = scene.registry.initial(names)
    vectors = [vector, vector+np.array([.1, 0, 0, 0, 0, 0])]
    expected = []
    for x in vectors:
        v = scene.registry.unpack(x, names); basis = scene.basis(v)
        expected.append(scene.residual(scene.project(v, eligible, basis), basis=basis))
    workers = JacobianWorkers(scene, names, eligible, 2)
    try:
        actual = list(workers(None, vectors))
    finally:
        workers.close()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
