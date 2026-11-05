import json
import numpy as np
import pytest
from astropy.io import fits
from chromorph import Scene, EmpiricalPSF, Observation
from chromorph.config import load_config, build_scene
from chromorph.simulation import simulate
from conftest import observations, clump


def test_invalid_observation_rejected():
    o = observations()[0]
    with pytest.raises(ValueError, match='scales disagree'):
        Observation('bad', 'blue', o.data, o.sigma, o.psf, .08, 1.)
    with pytest.raises(ValueError, match='no usable'):
        Observation('bad', 'blue', o.data, np.zeros_like(o.sigma), o.psf, .04, 1.)
    qa = o.psf.qa()
    assert abs(qa['sum']-1) < 1e-10
    assert qa['encircled_energy_radii_arcsec']['0.9'] > qa['encircled_energy_radii_arcsec']['0.5']


def test_write_complete_outputs_and_flux_accounting(tmp_path, renderer):
    truth = Scene(observations(), renderer=renderer)
    truth.add_component(clump('one', {'blue': 2, 'red': 3}))
    scene = Scene(simulate(truth, seed=16), renderer=renderer)
    scene.add_component(clump('one'))
    result = scene.fit(multistart=1); result.write(tmp_path)
    for name in ['result.json', 'parameters.ecsv', 'photometry.ecsv', 'images.fits', 'uncertainty.fits', 'diagnostic.pdf', 'sed.pdf', 'index.html']:
        assert (tmp_path/name).exists()
    data = json.loads((tmp_path/'result.json').read_text())
    assert data['component_sed_classes']['one'] == 'FreeBandSED'
    with fits.open(tmp_path/'images.fits', checksum=True) as hd:
        for o in scene.observations:
            np.testing.assert_allclose(hd[o.name+'_MODEL'].data,
                                       hd[o.name+'_COMP_one'].data+result.values[f'observation.{o.name}.background'])
        assert all(h.verify_checksum() == 1 for h in hd)


def test_yaml_has_independent_models_and_explicit_defaults(tmp_path, renderer):
    path = tmp_path/'model.yaml'
    path.write_text('''components:
  - name: nucleus
    spatial: point
    parameters: {x: 12, y: 12}
    sed: {model: free}
  - name: young
    spatial: sersic
    sed: {model: fsps, sfh: young, nebular: true}
  - name: old
    spatial: sersic
    sed: {model: fsps, sfh: old, nebular: false}
''')
    config = load_config(path)
    with build_scene(config, observations(), renderer=renderer) as scene:
        assert scene.components[1].sed is not scene.components[2].sed
        assert scene.registry.parameters['young.sed.age'].value != scene.registry.parameters['old.sed.age'].value
        assert not scene.registry.ties
    path.write_text('components: []\nmisspelled_option: true\n')
    with pytest.raises(ValueError, match='Unknown'):
        load_config(path)


def test_same_band_repeated_exposures_share_filter_flux(renderer):
    from dataclasses import replace
    a = observations(('b',))[0]; b = replace(a, name='b_second')
    scene = Scene([a, b], renderer=renderer)
    scene.add_component(clump('a', {'b': 3}))
    assert len([n for n in scene.registry.names() if '.flux.' in n]) == 1
    np.testing.assert_array_equal(*scene.predict()[0].values())
