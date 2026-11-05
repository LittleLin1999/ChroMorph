from chromorph.cli import example_configs, fit_options
from chromorph.config import build_scene
from chromorph import SEDContext
from conftest import observations


def test_generated_defaults_and_explicit_systematic_families():
    def row(name, spatial):
        return dict(id=name, type=spatial, enabled=True, parameters={
            'x': dict(value=12., min=5., max=20., free=True),
            'y': dict(value=12., min=5., max=20., free=True)})
    report = dict(root='/synthetic', redshift=.6,
        initial_guess=dict(components=[row('point', 'point'), row('one', 'sersic'), row('two', 'sersic')]))
    configs = example_configs(report)
    assert set(configs) == {'A_free', 'B_constant_sfr', 'C_ssp', 'D_exponential'}
    for name, sfh in [('B_constant_sfr', 'constant'), ('C_ssp', 'ssp'), ('D_exponential', 'exponential')]:
        config = configs[name]
        assert config['components'][0]['sed']['model'] == 'free'
        assert all(c['sed']['sfh'] == sfh for c in config['components'][1:])
        with build_scene(config, observations(), context=SEDContext(redshift=.6)) as scene:
            assert scene.components[1].sed is not scene.components[2].sed
            assert not scene.registry.ties
            assert not scene.components[1].sed.parameters['gas_logu'].vary
            assert 'one.sed.age' in scene.registry.names() and 'two.sed.age' in scene.registry.names()
    assert fit_options(configs['B_constant_sfr'])['optimizer'] == 'trf'
