import json
from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from chromorph import load_observations
from chromorph.config import load_config
from chromorph.cli import observations_for
from conftest import observations

@pytest.mark.parametrize('target', ['example_alpha', 'example_beta'])
def test_arbitrary_target_manifest_roundtrip_and_integrity(tmp_path, target, monkeypatch):
    original=observations(('custom_blue','custom_red'))
    rows=[]
    for o in original:
        image=tmp_path/(o.name+'.fits');psf=tmp_path/(o.name+'_psf.fits')
        fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(o.data,name='SCI'),
            fits.ImageHDU(o.sigma,name='ERR'),fits.ImageHDU(o.mask.astype('i2'),name='INCLUDE')]).writeto(image)
        fits.writeto(psf,o.psf.image)
        from chromorph.observations import sha256
        rows.append(dict(name=o.name,band=o.band,data=dict(path=image.name,ext='SCI',sha256=sha256(image)),
            sigma=dict(path=image.name,ext='ERR'),mask=dict(path=image.name,ext='INCLUDE'),
            psf=dict(path=psf.name,pixel_scale=o.psf.pixel_scale,center=o.psf.center),
            pixel_scale=o.pixel_scale,ujy_per_rate=o.ujy_per_rate,pivot_angstrom=o.pivot_angstrom))
    manifest=tmp_path/'observations.yaml'
    manifest.write_text(json.dumps(dict(format='chromorph-observations-v1',target=target,observations=rows)))
    config=tmp_path/'model.yaml';config.write_text(json.dumps(dict(target=target,observations=manifest.name,components=[dict(name='nucleus',spatial='point',sed=dict(model='free'))])))
    monkeypatch.chdir(tmp_path.parent)
    loaded,report,_=observations_for(load_config(config))
    assert report['target']==target
    for x,y in zip(original,loaded):
        for attr in ['data','sigma','mask','matrix','offset']:np.testing.assert_array_equal(getattr(x,attr),getattr(y,attr))
        np.testing.assert_array_equal(x.psf.image,y.psf.image)
        assert x.ujy_per_rate==y.ujy_per_rate
    with fits.open(tmp_path/rows[0]['data']['path'],mode='update') as hd:hd['SCI'].data[1,1]=99
    with pytest.raises(ValueError,match='provenance hash mismatch'):load_observations(manifest)

def test_tabulated_bandpass_matches_reference():
    pytest.importorskip('synphot')
    from chromorph.photometry import TabulatedThroughput
    response=TabulatedThroughput([3000,4000,5000,6000],[0,1,.6,0])
    wave=np.linspace(2000,7000,1000);flux=1e-17*(wave/4000)**-.8
    assert response.microjy(wave,flux)==pytest.approx(response.microjy_reference(wave,flux),rel=1e-12)
