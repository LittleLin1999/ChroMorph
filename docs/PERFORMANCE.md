# Likelihood paths and acceleration status

GalSim and direct FSPS photometry are the reference backends. The spatial,
spectral and amplitude layers have independent dependencies. Caches use exact
parameter states with explicit memory limits; parameter rounding is not used.
Every physical component owns an isolated FSPS process. Optional spectral
workers can request calculations from multiple independent processes together.

## Full and compressed Gaussian likelihoods

`FullPixelLikelihood` renders the complete scene at each evaluation and reuses
only parameter-independent PSF/data preparation. It requires neither a spatial
template cache nor a Gram matrix. Any number of components can change at once.

`scene.loglikelihood(values, mode="auto")` uses the full pixel path when the
spatial state changes and compressed statistics for flux/SED-only changes.
Explicit `full` and `compressed` modes remain available for numerical checks
and workload-specific benchmarking.

Compression is exact for fixed Gaussian covariance and preserves likelihood
normalization. QR factors evaluate the quadratic stably. Incremental updates
retain unchanged spatial columns and Gram entries when beneficial; proposals
changing many components rebuild the factorization. Model-dependent variance
requires a different likelihood. No large covariance inverse is formed.

SciPy variable projection solves eligible linear amplitudes with constrained
whitened least squares. TRF retains full pixel residuals for morphology steps;
L-BFGS-B uses the compressed scalar objective. Statistical priors can prevent
an amplitude from being eligible for projection. Full-parameter optimization
is retained. Bayesian evidence never uses this profiled objective.

## Stellar templates

`BandFluxLibrary` stores filter-integrated reference spectra, including dust
and configured nebular emission before filter integration. `SSPBandLibrary`
can obtain the native FSPS full-age grid and integrate its interpolation rule
in linear time to form a constant-SFR population per unit formed mass.

`PrecomputedFSPSSED` requires matching observational context and configuration.
Without a passed accuracy gate it falls back to direct reference calculations,
or raises when configured to do so. `experimental=True` is an explicit opt-in.
Photometric agreement alone does not enable production inference: validation
must also quantify image-level errors over the intended parameter domain.

## Experimental differentiable backends

`chromorph.backends.jax_spatial.JAXRenderer` implements batched FFT convolution
with precomputed empirical-PSF transforms and float64 arithmetic. It requires
explicit experimental opt-in. Compact, centrally concentrated high-index
profiles failed reference accuracy tests; this renderer is not approved for
general production inference. No GPU speed improvement is claimed.

`chromorph.backends.dsps.DSPSBackend` consumes the same FSPS SSP band library.
DSPS age weighting and a differentiable analytic SSP integral are separate
options, with separate accuracy requirements. Stored line luminosities are
not assumed to reproduce the full FSPS nebular model. General DSPS equivalence
has not been established across the required physical parameter domain.

`chromorph.backends.numpyro.NumPyroNUTSSampler` is an experimental adapter. It
refuses sampling until its likelihood/gradient gate passes against a reference
scene. End-to-end NUTS performance, recovery and posterior coverage have not
been validated for the general model. NUTS does not estimate evidence.
These modules import optional dependencies only when explicitly selected;
the minimal CPU/free-band installation does not require them.

## Reproducible CPU benchmarks

```sh
python -m chromorph.benchmarks --output /tmp/chromorph-benchmarks.json \
  --stamps 64 128 256 --filters 2 4 6 --clumps 1 2 4 --repeats 3
```

The primary workload changes every component's morphology simultaneously.
Additional cases isolate one-component geometry and band-amplitude changes.
Results include initialization, update timings, actual rendering counts,
likelihood equivalence, process peak memory and optional MAP recovery through
`--map-nfev`. Process peak memory is a high-water mark. Warm-cache likelihood
speedups are not end-to-end posterior-inference speedups. Sampler convergence,
effective samples per resource cost and model-mismatch validation require
separate experiments before selecting a production acceleration backend.
