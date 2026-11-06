"""Production renderer tests; skipped explicitly when GalSim is unavailable."""
import numpy as np
import pytest
pytest.importorskip('galsim')
from chromorph import Scene, PointSource, FreeBandSED, Sersic, Parameter
from chromorph.rendering import GalSimRenderer
from chromorph.simulation import simulate
from conftest import observations, clump


def test_empirical_psf_roundtrip_no_double_pixel_convolution():
    renderer = GalSimRenderer()
    o = observations()[0]
    qa = renderer.validate_psf_sampling(o)
    assert qa['method'] == 'no_pixel' and qa['pass_same_grid']
    assert abs(qa['half_pixel_sum']-1) < .01


def test_subpixel_centroid_and_affine():
    o = observations(size=41)[0]
    o.matrix = np.array([[.98, .03], [-.02, 1.01]])
    o.offset = np.array([.4, -.7])
    p = dict(x=19.2, y=20.4)
    im = GalSimRenderer().render(PointSource('probe', FreeBandSED()), p, o, (.2, -.1))
    yy, xx = np.indices(im.shape)
    actual = np.array([(xx*im).sum(), (yy*im).sum()])/im.sum()
    expected = o.matrix@np.array([p['x'], p['y']])+o.offset+np.array([.2, -.1])
    np.testing.assert_allclose(actual, expected, atol=.04)


def test_small_frame_does_not_renormalize_lost_flux():
    o = observations(size=15)[0]
    im = GalSimRenderer().render(Sersic('a', FreeBandSED()),
                                 dict(x=0., y=0., re=4., n=1., q=.8, pa=15.), o)
    assert 0 < im.sum() < .6


def test_galsim_joint_moving_clump_and_nucleus_recovery():
    obs = observations(size=31, noise=.0004)
    truth = Scene(obs)
    truth.add_component(PointSource('nucleus', FreeBandSED({'blue': 2., 'red': 4.}),
                                     parameters={'x': Parameter(15.2, 12, 18), 'y': Parameter(15.4, 12, 18)}))
    truth.add_component(clump('a', {'blue': 3., 'red': .4}, x=11, y=17))
    data = simulate(truth, seed=8)
    fit = Scene(data)
    fit.add_component(PointSource('nucleus', FreeBandSED(),
                                   parameters={'x': Parameter(15., 12, 18), 'y': Parameter(15., 12, 18)}))
    fit.add_component(clump('a', x=11.2, y=16.8, moving=True))
    result = fit.fit(multistart=2, max_nfev=100)
    np.testing.assert_allclose(list(result.fluxes['a'].values()), [3., .4], atol=.08)
    np.testing.assert_allclose([result.values['nucleus.morphology.x'], result.values['nucleus.morphology.y']], [15.2, 15.4], atol=.05)


def test_gaussian_index_machine_precision_limit():
    renderer = GalSimRenderer()
    obs = observations(size=41)[0]
    component = Sersic('a', FreeBandSED())
    morphology = dict(x=20., y=20., re=7.9, n=.5, q=.15, pa=81.6)
    exact = renderer.render(component, morphology, obs)
    morphology['n'] = .5000000000002432
    nearby = renderer.render(component, morphology, obs)
    np.testing.assert_allclose(nearby, exact, rtol=1e-9, atol=1e-12)
