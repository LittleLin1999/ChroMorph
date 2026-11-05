# ChroMorph

Joint multi-band astronomical image decomposition with **independent SEDs for
every component** and a completely free nuclear point-source spectrum.

ChroMorph fits all components and all images in one likelihood. Every Sérsic
has constant reference-grid morphology across filters, independent centers and
shape parameters by default, and its own SED object and parameter namespace.
Free per-filter fluxes, separate FSPS stellar populations, and mixtures are
supported. No common age, SFH, metallicity, dust, mass-to-light ratio or fixed
colors are imposed between clumps. `nucleus` and explicitly nuclear point
sources must use unconstrained `FreeBandSED`; they are not classified as AGN.

## Install

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest -ra
python -m chromorph doctor
```

For physical populations, install `.[stellar]`, configure FSPS with its stellar
libraries (`SPS_HOME` where required by python-fsps), and install STSynphot's
HST reference tables (`PYSYN_CDBS`). Optional `.[bayes]` enables emcee sampling.
Missing stellar libraries do not prevent free-SED fitting. GalSim is required
for the production spatial renderer. Nothing silently falls back to a different
science renderer or a surrogate stellar spectrum.

## Python interface

```python
from chromorph import Scene, Observation, EmpiricalPSF, Parameter, SEDContext
from chromorph.spatial import PointSource, Sersic
from chromorph.sed import FreeBandSED, FSPSStellarSED

# observations: calibrated Observation instances, with a per-observation PSF,
# uncertainty map, inclusion mask and reference-to-observation transform.
with Scene(observations, context=SEDContext(redshift=0.608)) as scene:
    scene.add_component(PointSource(
        'nucleus', FreeBandSED(), x=50., y=50., nuclear=True))
    scene.add_component(Sersic(
        'clump_1', FSPSStellarSED(sfh='young', nebular=True), x=46., y=52.))
    scene.add_component(Sersic(
        'clump_2', FSPSStellarSED(sfh='old', nebular=False), x=55., y=48.))
    scene.add_component(Sersic(
        'clump_3', FreeBandSED(), x=50., y=60.))
    result = scene.fit(multistart=3, seed=42, variable_projection=True)
    result.write('results/hybrid')
```

Initialize positions and sensible bounds for the target. Explicit `Parameter`
objects set bounds, fixed parameters, and independent Gaussian priors. A scalar
in YAML means fixed; a parameter mapping varies unless `vary: false` is given.
`scene.tie_centroids(['clump_1', 'clump_2'])` explicitly groups centroids;
`scene.tie(alias, target)` ties other parameters. Nuclear band fluxes cannot be
tied, fixed, given color priors or given finite upper bounds. All fluxes are
nonnegative. SED instance reuse between components is rejected.

`young` and `old` are initialization presets for SSP ages, not fixed physical
classifications. `ssp`, `exponential`, and `delayed` choose SFH families per
component. `dust_type` selects its independent FSPS attenuation family. Mass
means **formed stellar mass**, in solar masses; luminosities are divided by
FSPS `formed_mass` before applying that normalization. Age is in Gyr,
metallicity is log10(Z/Zsun), and `av` is diffuse V-band attenuation in mag.
Each component gets a dedicated spawned process with one FSPS instance. This
avoids FSPS process-level mutable-state contamination; call `scene.close()` or
use the context manager. Redshift, distance and foreground extinction are
observational context, not shared intrinsic stellar populations.

## J1042

Discover the actual products without inventing missing images, errors or PSFs:

```sh
chromorph discover-j1042 /home/linxiaojing/data/lrd-squared/hst \
  --output results/j1042_inventory.json --config-dir examples/generated
chromorph fit examples/generated/A_free.yaml --output results/A_free
chromorph fit examples/generated/B_independent_stellar.yaml --output results/B_stellar
chromorph fit examples/generated/C_hybrid.yaml --output results/C_hybrid
chromorph compare examples/generated/{A_free,B_independent_stellar,C_hybrid}.yaml \
  --output results/comparison
chromorph validate examples/generated/A_free.yaml \
  --repeats 10 --output results/recovery_A.json
```

The checked-in example configurations contain illustrative starting values,
not fitted J1042 measurements. Regenerate them against your actual data root. Model B
uses a **separate** FSPS object for each extended component; Model C uses FSPS
for two clumps and free-band SEDs for the remaining components. Edit component
initial guesses, bounds, SFH, nebular choices and dust laws independently.
Generated Sérsic-index bounds are intersected with GalSim's supported
0.3–6.2 range; this is a numerical renderer restriction, not a common population
prior. Fits compare identical images, PSFs, masks and uncertainty maps.

The loader verifies existing source/PSF hashes and uses the selected fit's
stored data, masks, noise, sky and registration; WHT is not assumed to be a full
variance model. See [J1042 setup](docs/J1042_SETUP.md) and
[validation status](docs/VALIDATION.md). No J1042 ChroMorph science fit is
claimed until the required renderer, physical libraries and recovery QA pass.

## Conventions and diagnostics

- Intrinsic x/y and semimajor effective radius are zero-based reference pixels.
  PA is degrees counterclockwise from reference +x. Every observation carries
  a full affine transform and optional registration corrections.
- `FreeBandSED` fluxes are total microJy per filter, independent across bands.
  Spatial rendering preserves aperture losses; components are not renormalized
  to the fit stamp. FITS image planes are electrons/s. Output photometry
  distinguishes total flux from frame-integrated rate.
- Empirical drizzled PSFs are treated as already pixel-integrated and rendered
  with GalSim `drawImage(method='no_pixel')`; no second detector-pixel convolution
  is added. Declared fractional PSF centers are preserved. PSF QA records
  normalization, centroid, area-equivalent FWHM, encircled energy and edge flux.
- Optional variable projection solves eligible linear free-band amplitudes in
  one global NNLS. Stellar SEDs are evaluated inside every likelihood call.
  Their mass normalization remains in the nonlinear global vector. Spatial
  bases and stellar fluxes are cached per component with their relevant
  morphology, SED, source-context and throughput settings.
- Backgrounds, registration, and relative calibration can be fitted. The first
  observation anchors calibration and registration gauges. Default nuisance
  settings are explicit in the configuration.
- Outputs include FITS model/component/residual planes, ECSV parameters and
  photometry, JSON configuration/provenance/multistart results, local covariance
  and correlations, PDF/PNG diagnostics, and a standalone HTML report.
- Nuclear fluxes are plotted as independent band points. Physical spectra are
  plotted as curves. No nuclear smooth SED is assumed.
- Rank-deficient directions, near-bound solutions and local prior-information
  fractions are reported. Unidentifiable local errors are null, not zero.
  Local linear errors are insufficient for nonlinear or multimodal solutions;
  `result.sample()` provides optional emcee chains for further investigation.

See [scientific limits](docs/SCIENCE.md) for noise, PSF, identifiability and
validation caveats, and [public API references](docs/API_REFERENCES.md).
