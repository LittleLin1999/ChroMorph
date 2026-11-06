"""Definition/execution separation and migration of useful conveniences."""
import pickle
import subprocess
import sys
import numpy as np
from chromorph import Scene
from chromorph.direct import FullPixelLikelihood
from chromorph.gaussian import marginal_linear_flux
from chromorph.inference import marginal_linear_flux as legacy_integral
from conftest import observations, clump


def test_definition_serialization_excludes_live_workspace(renderer):
    scene=Scene(observations(),renderer=renderer)
    scene.add_component(clump('one'))
    expected=scene.predict()[0]
    workspace=scene.prepare()
    workspace._likelihood_engines['not_picklable']=lambda x:x
    restored=pickle.loads(pickle.dumps(scene))
    assert restored._evaluator is None
    assert restored.prepare() is not workspace
    for name,image in restored.predict()[0].items():
        np.testing.assert_array_equal(image,expected[name])
    restored.close();scene.close()


def test_one_workspace_owns_all_likelihood_execution(renderer):
    with Scene(observations(),renderer=renderer) as scene:
        scene.add_component(clump('one'))
        workspace=scene.prepare()
        assert scene.prepare() is workspace
        full=FullPixelLikelihood(scene)
        assert full.scene is workspace
        assert scene.compressed.scene is workspace
        assert scene._basis_cache is workspace._basis_cache
        values=scene.registry.unpack()
        assert scene.loglikelihood(values,mode='full')==workspace.loglikelihood(values,mode='full')
    assert workspace.closed
    assert scene._evaluator is None


def test_nuclear_integral_has_one_sampler_independent_implementation():
    assert marginal_linear_flux is legacy_integral
    code='''import sys
import chromorph.gaussian
import chromorph.direct
assert "chromorph.inference" not in sys.modules
assert "pypolychord" not in sys.modules
assert "matplotlib.pyplot" not in sys.modules
assert "jax" not in sys.modules
assert "fsps" not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code],check=True)
