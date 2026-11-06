"""Scientific display checks: physical units, no nuclear curve, matched RGBs."""
from types import SimpleNamespace
import numpy as np
from chromorph import Scene, PointSource, FreeBandSED, Parameter
from chromorph.plotting import microjy_to_scaled_flam, component_rgbs, plot_component_sed
from conftest import observations, clump


def test_microjy_to_pivot_equivalent_flam():
    # 1 microJy at 1 micron is 2.99792458e-19 erg/s/cm2/Angstrom.
    np.testing.assert_allclose(microjy_to_scaled_flam([1., 4.], [1e4, 2e4]),
                               [.0299792458, .0299792458], rtol=1e-13)


def test_rgb_alignment_and_free_nuclear_display_preserve_science(tmp_path, renderer):
    scene = Scene(observations(), renderer=renderer)
    scene.add_component(PointSource('nucleus', FreeBandSED({'blue': 1., 'red': 2.}), parameters={
        'x': Parameter(12., vary=False), 'y': Parameter(12., vary=False)}))
    scene.add_component(clump('one', {'blue': 2., 'red': 1.}))
    values = scene.registry.unpack()
    models, images, fluxes = scene.predict(values)
    for o in scene.observations:o.update(data=models[o.name])
    result = SimpleNamespace(scene=scene, values=values, models=models, images=images, fluxes=fluxes)
    original = {o.name: o.data.copy() for o in scene.observations}
    rgb = component_rgbs(result)
    np.testing.assert_allclose(rgb['images']['extended'], rgb['images']['data_minus_psf'], atol=1e-14)
    assert not rgb['smoothing']
    figure = plot_component_sed(result, tmp_path)
    lines = {line.get_gid(): line for line in figure.axes[0].lines if line.get_gid()}
    assert lines['photometry:nucleus'].get_linestyle() == 'None'
    assert not any(name.startswith('stellar-spectrum:') for name in lines)
    np.testing.assert_allclose(lines['photometry:nucleus'].get_ydata(),
        microjy_to_scaled_flam([1., 2.], [4000., 7000.]))
    for o in scene.observations:np.testing.assert_array_equal(o.data, original[o.name])
    assert (tmp_path/'sed_display.json').exists()
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_full_range_stellar_sum_and_synthetic_marker_convention(tmp_path, renderer):
    from test_sfh_comparison import ReportPopulation
    scene = Scene(observations(), renderer=renderer)
    scene.add_component(PointSource('nucleus', FreeBandSED(), parameters={
        'x': Parameter(12., vary=False), 'y': Parameter(12., vary=False)}))
    scene.add_component(clump('one', sed=ReportPopulation(sfh='constant')))
    scene.add_component(clump('two', x=17., sed=ReportPopulation(sfh='ssp')))
    values = scene.registry.unpack();models, images, fluxes = scene.predict(values)
    result = SimpleNamespace(scene=scene, values=values, models=models, images=images, fluxes=fluxes)
    measured = {name: {b: f*1.1 for b, f in bands.items()} for name, bands in fluxes.items()}
    figure = plot_component_sed(result, tmp_path, measured_fluxes=measured, xlim=(.22, 2.5))
    lines = {line.get_gid(): line for line in figure.axes[0].lines if line.get_gid()}
    assert lines['synthetic:one'].get_markerfacecolor() == 'white'
    assert lines['photometry:one'].get_markerfacecolor() != 'white'
    assert 'stellar-spectrum:nucleus' not in lines
    total = lines['stellar-spectrum:total']
    assert total.get_xdata().max() == 2.5
    expected = sum(np.interp(total.get_xdata(), lines['stellar-spectrum:'+n].get_xdata(),
                            lines['stellar-spectrum:'+n].get_ydata()) for n in ['one', 'two'])
    np.testing.assert_allclose(total.get_ydata(), expected)
    assert {'component:one', 'component:two', 'total'} <= component_rgbs(result)['images'].keys()
    import matplotlib.pyplot as plt
    plt.close(figure)
