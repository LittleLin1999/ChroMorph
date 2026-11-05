# ChroMorph

Joint multi-band astronomical image decomposition with **independent SEDs for
every component** and a completely free nuclear point-source spectrum.

ChroMorph fits all components and all images in one likelihood. Every Sérsic
has constant reference-grid morphology across filters, independent centers and
shape parameters by default, and its own SED object and parameter namespace.
Free per-filter fluxes, separate FSPS stellar populations, and mixtures are
supported. No common age, SFH parameters, metallicity, dust, mass-to-light ratio or fixed
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
        'clump_1', FSPSStellarSED(sfh='constant', nebular=True), x=46., y=52.))
    scene.add_component(Sersic(
        'clump_2', FSPSStellarSED(sfh='constant', nebular=True), x=55., y=48.))
    result = scene.fit(multistart=3, seed=42, variable_projection=True)
    result.write('results/independent_populations')
```

Initialize positions and sensible bounds for the target. Explicit `Parameter`
objects set bounds, fixed parameters, and independent Gaussian priors. A scalar
in YAML means fixed; a parameter mapping varies unless `vary: false` is given.
`scene.tie_centroids(['clump_1', 'clump_2'])` explicitly groups centroids;
`scene.tie(alias, target)` ties other parameters. Nuclear band fluxes cannot be
tied, fixed, given color priors or given finite upper bounds. All fluxes are
nonnegative. SED instance reuse between components is rejected.

The default physical SED uses **constant star formation**, with documented FSPS
settings `sfh=1, const=1, fburst=0, sf_start=0`. Its `age`/`tage` is the duration
of ongoing star formation. Each clump independently fits its duration, formed
stellar mass, metallicity and dust attenuation. For this model the intrinsic
constant SFR is formed mass divided by duration in years; it is not an extra
normalization shared across clumps. `ssp` and `exponential` are explicit
systematic alternatives. Legacy `young` and `old` select SSP starting ages and
are not the J1042 default. `constant`, `ssp`, `exponential`, and `delayed` choose SFH families per
component. `dust_type` selects its independent FSPS attenuation family. Mass
means **formed stellar mass**, in solar masses; luminosities are divided by
FSPS `formed_mass` before applying that normalization. Age is in Gyr,
metallicity is log10(Z/Zsun), and `av` is diffuse V-band attenuation in mag.
Each component gets a dedicated spawned process with one FSPS instance. This
avoids FSPS process-level mutable-state contamination; call `scene.close()` or
use the context manager. Redshift, distance and foreground extinction are
observational context, not shared intrinsic stellar populations.

With `nebular=True`, stellar-powered nebular lines and continuum are enabled.
`gas_logu=-2` is fixed independently in each component by default: four HST
bands do not justify automatically fitting an additional ionization parameter.
An explicit per-component `Parameter(..., vary=True)` can enable it. Gas
metallicity follows that component's stellar metallicity unless overridden.
The nuclear point source always retains its free per-filter SED.

## J1042

Discover the actual products without inventing missing images, errors or PSFs:

```sh
chromorph discover-j1042 /home/linxiaojing/data/lrd-squared/hst \
  --output results/j1042_inventory.json --config-dir examples/generated
chromorph fit examples/generated/A_free.yaml --output results/A_free
chromorph fit examples/generated/B_constant_sfr.yaml --output results/B_constant_sfr
chromorph compare examples/generated/{A_free,B_constant_sfr,C_ssp}.yaml \
  --output results/comparison
chromorph validate examples/generated/A_free.yaml \
  --repeats 10 --output results/recovery_A.json
```

The checked-in example configurations contain illustrative starting values,
not fitted J1042 measurements. Regenerate them against your actual data root. Model B
uses a **separate constant-SFR** FSPS object with nebular emission enabled for
each extended component. Model C uses independent SSPs; an exponentially
declining alternative is also generated but is not fitted automatically. Edit component
initial guesses, bounds, SFH, nebular choices and dust laws independently.
Generated Sérsic-index bounds are intersected with GalSim's supported
0.3–6.2 range; this is a numerical renderer restriction, not a common population
prior. Fits compare identical images, PSFs, masks and uncertainty maps.

The loader verifies existing source/PSF hashes and uses the selected fit's
stored data, masks, noise, sky and registration; WHT is not assumed to be a full
variance model. See [J1042 setup](docs/J1042_SETUP.md) and
[validation status](docs/VALIDATION.md). Private same-input SFH reports separate
best-fit sensitivity from statistical significance. Four-band population
constraints, PSF mismatch and correlated-noise uncertainty coverage require
additional validation before strong scientific conclusions.

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
- Optional variable projection solves eligible free-band amplitudes, stellar
  masses and linear backgrounds together using a bounded whitened QR solve.
  Stellar SEDs are evaluated inside the global likelihood. A full-parameter
  fitting fallback is retained. Spatial
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

## First best fit, then PolyChord

Use bounded local least squares for a first best-fit solution, then sample the
joint posterior with PolyChord. The first best fit does not truncate or replace
the configured priors. Numerical derivatives can use independent spawned worker
processes with `scene.fit(likelihood="pixel", workers=12)`. The compressed
optimizer maintains one dependency-aware scene; independent model comparisons
can run in separate processes. L-BFGS-B evaluates the compressed scalar;
TRF keeps the full pixel residual for reliable morphology Gauss-Newton steps.
Parallel derivatives require SciPy >=1.16. Each rendering worker owns its own
scene. Rendering workers receive exact component flux answers and do not
start nested FSPS pools; their residuals are checked against the main likelihood.

HST synthetic photometry caches a linear integration operator on the original
spectrum/bandpass wavelength grid. This reproduces synphot photon interpolation
and trapezoidal integration without constructing a new Observation every call.
`HSTThroughput(..., integration="reference")` retains direct synphot evaluation
for numerical checks or reproducing an earlier optimization.

Install the official [PolyChordLite](https://github.com/PolyChord/PolyChordLite)
library and Python bindings with MPI and a Fortran compiler, plus
`pip install '.[polychord]'`. Then run, for example:

```sh
mpiexec -n 24 chromorph sample-polychord examples/generated/B_constant_sfr.yaml \
  --output results/B_polychord --nlive 1300 --num-repeats 104 --precision 0.01
```

The supplied settings are illustrative for a 52-dimensional sampled model.
Check independent seeds, slice lengths, live-point counts and prior sensitivity
before interpreting a posterior. All non-nuclear sampled parameters need proper
priors; configured Gaussian priors are transformed as bounded, normalized
Gaussians and are not counted twice in the likelihood.

The nucleus retains independent nonnegative fluxes with no upper limit or
spectral prior. For the supported one-nucleus-per-band Gaussian likelihood,
these amplitudes are analytically **integrated**, not maximized. Independent
conditional truncated-normal draws restore the nuclear flux posterior in
`posterior_with_nucleus.ecsv`. The weighted nonlinear posterior is also saved.
This flat unbounded nuclear measure does not define an absolute Bayesian
evidence; `inference.json` explicitly records `log_evidence: null`. PolyChord's
internal log integral must not be used as a Bayes factor. Additional free-band
components require explicit proper priors for their sampled amplitudes.
