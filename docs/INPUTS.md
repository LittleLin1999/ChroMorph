# General observation manifests

ChroMorph accepts any target and any explicitly calibrated filter through the
Python `Observation` API or a YAML/JSON manifest. Target discovery, reduction
QA and private data belong outside the installed package.

```yaml
format: chromorph-observations-v1
target: example
context:
  redshift: 0.1
observations:
  - name: visit_blue
    band: blue
    data: {path: blue.fits, ext: SCI}
    sigma: {path: blue.fits, ext: ERR}
    mask: {path: blue.fits, ext: INCLUDE}
    psf:
      path: blue_psf.fits
      ext: 0
      pixel_scale: 0.04
      center: [15, 15]
      pixel_integrated: true
    pixel_scale: 0.04
    ujy_per_rate: 0.5
    pivot_angstrom: 4500
    matrix: [[1, 0], [0, 1]]
    offset: [0, 0]
    background: 0
    throughput:
      wave_angstrom: [3500, 4000, 4500, 5000, 5500]
      response: [0, 0.5, 1, 0.5, 0]
```

Paths are relative to the manifest. Every file block may include `sha256`;
a mismatch fails loading. Arrays and flux calibration must use consistent
image-rate units; `ujy_per_rate` converts one unit of image rate to microJy.
An included pixel has mask value True/nonzero. Invalid pixels are excluded by
`Observation`. The mask may be omitted; the calibrated sigma map is required.
WHT is not interpreted as a complete variance model. Pixel scales are arcsec,
centers are zero-based pixels, and throughput wavelengths are Angstrom.
An identity matrix and zero offset mean the images already share the reference
pixel geometry. Supply an affine transform for other grids; alignment is not
inferred. The empirical PSF must have the declared sampling.

Free-band fitting does not require throughput curves. Physical SED components
require one for every observation. Use an explicit STSynphot HST obsmode
(`throughput: {obsmode: 'wfc3,uvis2,f621m'}`) or a tabulated response for any
instrument. The tabulated integration uses the documented synphot
[SpectralElement/Empirical1D interface](https://synphot.readthedocs.io/en/stable/synphot/from_pysyn_iraf.html).

Select the manifest in the component configuration:

```yaml
target: example
observations: observations.yaml
components:
  - name: nucleus
    spatial: point
    nuclear: true
    sed: {model: free}
```

`observations` is relative to the configuration. Alternatively, `data_root`
selects a directory containing `observations.yaml`. Every clump independently
chooses its own spatial and SED model, parameters, bounds and priors. An input
manifest does not tie any intrinsic component parameter. No target name is
interpreted specially.

Use `chromorph inspect observations.yaml`, then `chromorph fit model.yaml
--output results/model`. Keep package source in a code checkout, input manifests
and arrays in a data folder, and analysis scripts/results in a target folder.
