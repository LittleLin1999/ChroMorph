import numpy as np
from chromorph import Scene, FreeBandSED, PointSource, Parameter
from chromorph.linear import LinearProjection
from chromorph.sed import FSPSStellarSED
from conftest import observations, clump


class AnalyticStellarSED(FSPSStellarSED):
    """An exact linear mass model; no optional FSPS dependency needed."""
    def fluxes(self, values, name, observations, context):
        age = values[name+'.sed.age']
        mass = 10**values[name+'.sed.log_mass']
        return {o.band: mass*1e-8*(1+age*(i+1)*.1) for i, o in enumerate(observations)}


def test_mass_nuclear_flux_and_signed_sky_recover_together(renderer):
    scene = Scene(observations(('b', 'v', 'r', 'i')), renderer=renderer, fit_background=True)
    scene.add_component(PointSource('nucleus', FreeBandSED(), parameters={
        'x': Parameter(12., vary=False), 'y': Parameter(12., vary=False)}))
    scene.add_component(clump('one', x=8., sed=AnalyticStellarSED()))
    scene.add_component(clump('two', x=16., sed=AnalyticStellarSED()))
    truth = scene.registry.unpack()
    truth['one.sed.log_mass'] = 8.4;truth['two.sed.log_mass'] = 8.1
    for i, o in enumerate(scene.observations):
        truth['nucleus.flux.'+o.band] = .3+i*.4
        truth['observation.'+o.name+'.background'] = -.001+i*.001
    model, _, _ = scene.predict(truth)
    for o in scene.observations:
        o.data[:] = model[o.name]
    projection = LinearProjection(scene)
    fitted = projection.solve(scene.registry.unpack())
    for name in projection.names:
        np.testing.assert_allclose(fitted[name], truth[name], rtol=1e-9, atol=1e-10)
    assert scene.compressed.chi2(fitted) < 1e-15
    assert projection.last_diagnostics['rank'] == len(projection.names)


def test_projected_amplitudes_respect_bounds_and_flag_flux_exchange(renderer):
    scene = Scene(observations(), renderer=renderer)
    scene.add_component(clump('one'));scene.add_component(clump('two'))
    for o in scene.observations:
        o.data[:] = -1.
    projection = LinearProjection(scene)
    fitted = projection.solve(scene.registry.unpack())
    assert all(fitted[n] >= 0 for n in projection.names)
    assert projection.last_diagnostics['rank'] < len(projection.names)
