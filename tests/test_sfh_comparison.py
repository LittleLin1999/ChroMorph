"""Report assembly and input-identity checks, using analytic test populations."""
import json
import numpy as np
import pytest
from chromorph import Scene, FreeBandSED, PointSource, Parameter
from chromorph.result import FitResult
from chromorph.comparison import compare_sfh_results
from conftest import observations, clump
from test_linear_projection import AnalyticStellarSED


class ReportPopulation(AnalyticStellarSED):
    def spectrum(self, values, name, context):
        wave = np.linspace(2000., 25000., 100)
        fnu = 10**values[name+'.sed.log_mass']*1e-8
        return wave, fnu*1e-29*2.99792458e18/wave**2


def test_sfh_report_keeps_mass_and_photometry_systematics_explicit(tmp_path, renderer):
    obs = observations(('b', 'v', 'r', 'i'), size=17)
    paths = {}
    for model, sfh in [('A_free', None), ('B_constant_sfr', 'constant'), ('C_ssp', 'ssp')]:
        scene = Scene(obs, renderer=renderer)
        scene.add_component(PointSource('nucleus', FreeBandSED(), parameters={
            'x': Parameter(8., vary=False), 'y': Parameter(8., vary=False)}))
        scene.add_component(clump('one', x=5., sed=FreeBandSED() if sfh is None else ReportPopulation(sfh=sfh)))
        values = scene.registry.unpack();residual = scene.residual(values)
        run = dict(values=values, objective=float(residual@residual), success=True, message='Test fixture', nfev=1)
        result = FitResult(scene, values, [run], [])
        result.estimate_uncertainty()
        paths[model] = tmp_path/model
        result.write(paths[model])
    summary = compare_sfh_results(paths, tmp_path)
    assert summary['identical_inputs_verified']
    assert len(summary['mass_systematics']) == 1
    assert (tmp_path/'morphology_comparison.ecsv').exists()
    assert (tmp_path/'photometry_comparison.ecsv').exists()
    assert (tmp_path/'component_photometry_sfh.pdf').exists()
    populations = (tmp_path/'population_comparison.ecsv').read_text()
    assert 'SF_duration_Gyr' in populations and 'SSP_age_Gyr' in populations
