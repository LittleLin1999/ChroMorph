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

Keep package source, target inputs and analysis results in separate directories.
The package has no target-specific discovery rules or filesystem paths.
See [observation manifests](docs/INPUTS.md) for instrument-independent input loading.

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
with Scene(observations, context=SEDContext(redshift=0.1)) as scene:
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
tied, fixed or given spectral/color constraints. Independent, normalized
statistical flux priors are supported, including finite-support priors for
Bayesian evidence. All fluxes are nonnegative. SED instance reuse between
components is rejected.

The default physical SED uses **constant star formation**, with documented FSPS
settings `sfh=1, const=1, fburst=0, sf_start=0`. Its `age`/`tage` is the duration
of ongoing star formation. Each clump independently fits its duration, formed
stellar mass, metallicity and dust attenuation. For this model the intrinsic
constant SFR is formed mass divided by duration in years; it is not an extra
normalization shared across clumps. `ssp` and `exponential` are explicit
systematic alternatives. Legacy `young` and `old` select SSP starting ages and
do not select a universal population model. `constant`, `ssp`, `exponential`, and `delayed` choose SFH families per
component. `dust_type` selects its independent FSPS attenuation family. Mass
means **formed stellar mass**, in solar masses; luminosities are divided by
FSPS `formed_mass` before applying that normalization. Age is in Gyr,
metallicity is log10(Z/Zsun), and `av` is diffuse V-band attenuation in mag.
Each component gets a dedicated spawned process with one FSPS instance. This
avoids FSPS process-level mutable-state contamination; call `scene.close()` or
use the context manager. Redshift, distance and foreground extinction are
observational context, not shared intrinsic stellar populations.

With `nebular=True`, stellar-powered nebular lines and continuum are enabled.
`gas_logu=-2` is fixed independently in each component by default: limited photometry does not justify automatically fitting an additional ionization parameter.
An explicit per-component `Parameter(..., vary=True)` can enable it. Gas
metallicity follows that component's stellar metallicity unless overridden.
The nuclear point source always retains its free per-filter SED.

## Command line

Supply an explicit observation manifest and independent component configuration:

```sh
chromorph inspect observations.yaml
chromorph fit examples/A_free.yaml --output results/free
chromorph compare examples/A_free.yaml examples/B_constant_sfr.yaml examples/C_ssp.yaml --output results/comparison
chromorph validate examples/A_free.yaml --repeats 10 --output results/recovery.json
```

The example component configurations are illustrative. Provide your own images,
uncertainties, inclusion masks, PSFs, calibration, geometry and optional filter
throughputs in the manifest. No target name is interpreted specially. The Python
`Scene`/`Observation` interface also accepts arbitrary calibrated arrays.
Target discovery and reduction QA belong in external analysis adapters.

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
- SED figures use pivot-equivalent observed f_lambda, full-range individual
  and summed stellar spectra, filter curves and an aligned RGB footer for
  the total, extended model, PSF, data minus PSF and each extended component.
  Filled markers show free-band photometry; white markers with colored edges
  show synthetic/model band predictions. Optional reference photometry and
  its local errors are supplied explicitly, with their source saved in metadata.
  Extended and PSF-subtracted data use a common noise-aware intensity stretch
  without smoothing. `result.plot_sed(directory, ...)` accepts optional
  display-only spectra/photometry and redraws without refitting.
  `separate_components=True` adds individual panels without shifting component
  fluxes. An isolated post-fit FSPS backend can generate higher-resolution
  spectra at the saved low-resolution optimum; its band fluxes are checked
  against the actual likelihood predictions. See [post-fit spectra](docs/POSTFIT_SPECTRA.md).
  The PSF detail panel can overlay an explicitly supplied total-minus-extended
  spectrum at matched resolution. It uses a linear axis to retain negative
  residuals; stellar detail limits follow each component's continuum and
  photometry, rather than the tallest nebular line in another component.
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
`pip install '.[polychord]'`. See [nuclear flux priors](docs/NUCLEAR_SED.md)
for runnable prior configuration and the distinction between physical SED
constraints and statistical amplitude priors.

The default `sample-polychord` mode explicitly samples every free amplitude
with its own normalized prior and returns posterior samples and Bayesian
evidence. Configure proper priors for every sampled parameter, including each
nuclear band. Uniform, Gaussian, truncated Gaussian and lognormal scalar
priors are supported; no nuclear colors, templates, smoothness or cross-filter
prior correlations are introduced. Check independent seeds, slice lengths,
live-point counts and prior-width sensitivity before interpreting evidence.

For an explicitly chosen improper flat nonnegative nuclear measure, use
`--nuclear-mode flat_marginal`. Supported nuclear amplitudes are analytically
integrated, never profiled, and conditional draws restore their individual
flux posteriors. Evidence is undefined only when an improper measure remains;
that case is labeled `posterior_only` and has `log_evidence: null` in
`inference.json`. A physically unconstrained nuclear SED alone does **not**
prevent evidence calculation. The sampler never invents finite flux caps.

Maximum-likelihood fitting remains available with
`scene.fit(objective="ml")` or `fitting: {objective: ml}` in YAML. Statistical
priors are ignored in this mode; explicitly configured hard numerical bounds
and fixed morphology parameters still apply. Nuclear SED plots display
independent band measurements without a fitted continuum.

See [likelihood paths and experimental acceleration](docs/PERFORMANCE.md) for
full-rendering dispatch, incremental statistics, template accuracy gates and
reproducible benchmark commands.
