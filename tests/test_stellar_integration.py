"""Actual FSPS + HST bandpass integration, opt-in with installed reference data."""
import os
import numpy as np
import pytest
pytestmark = pytest.mark.integration
if os.environ.get('CHROMORPH_STELLAR_TESTS') != '1':
    pytest.skip('Set CHROMORPH_STELLAR_TESTS=1 with FSPS and STSynphot reference data', allow_module_level=True)
pytest.importorskip('fsps'); pytest.importorskip('stsynphot')
from chromorph import Scene, FSPSStellarSED, SEDContext
from chromorph.photometry import HSTThroughput
from conftest import observations, clump


def test_real_fsps_components_have_distinct_processes_and_independent_fluxes():
    obs = observations(('F275W', 'F336W', 'F621M', 'F689M'))
    for o in obs:
        o.throughput = HSTThroughput('wfc3,uvis2,'+o.band.lower())
    with Scene(obs, context=SEDContext(redshift=.608)) as scene:
        one, two = FSPSStellarSED(sfh='young', nebular=True), FSPSStellarSED(sfh='old')
        scene.add_component(clump('one', sed=one)); scene.add_component(clump('two', sed=two))
        values = scene.registry.unpack()
        a = one.fluxes(values, 'one', obs, scene.context)
        b = two.fluxes(values, 'two', obs, scene.context)
        values['one.sed.av'] = 1.
        changed = one.fluxes(values, 'one', obs, scene.context)
        assert a != changed and b == two.fluxes(values, 'two', obs, scene.context)
        assert one._executor is not two._executor
        assert a != b
