# Likelihood paths and acceleration status

GalSim and direct FSPS photometry are the reference backends. The spatial,
spectral and amplitude layers have independent dependencies. Caches use exact
parameter states with explicit memory limits; parameter rounding is not used.
Each prepared physical component owns an isolated, lazily started FSPS process. Optional spectral
workers can request calculations from multiple independent processes together.

## Full and compressed Gaussian likelihoods

`FullPixelLikelihood` renders the complete scene at each evaluation and reuses
only parameter-independent PSF/data preparation. It requires neither a spatial
template cache nor a Gram matrix. Any number of components can change at once.

`scene.loglikelihood(values, mode="auto")` uses the full pixel path when the
spatial state changes. Flux/SED-only changes reuse fixed residual anchors; QR statistics are used only when already available or explicitly amortized.
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


## CPU refactor measurements (Erebus, 2026-11-06)

Baseline d67018c; behavior-preserving structural checkpoint f2ee144. Process
CPU affinity exposes 48 CPUs; no NVIDIA device was detected. NumPy/BLAS/OpenMP
threads were fixed to one. `scripts/benchmark_cpu_execution.py` runs each
strategy in a fresh process, records versions, initialization, process wall
and CPU time, peak RSS, actual rendering calls and QR updates. Its mixed
trace deliberately includes revisits; warmed mixed timing is not a prediction
of a production sampler's total cost.

Initial bounded measurement, four calls changing every morphology parameter
(including fresh continuous n), four clumps, four filters, 64 square pixels:

| Strategy | Four global changes (s) | Complete scripted workload (s) | CPU-hours | Peak RSS (MiB) |
|---|---:|---:|---:|---:|
| Independent full reference | 0.285 | 2.213 | 0.000614 | 148.5 |
| Component policy | 0.290 | 1.435 | 0.000398 | 155.3 |
| Aggregate extended rendering | 0.354 | 2.314 | 0.000643 | 149.9 |
| Compressed / incremental | 0.299 | 1.445 | 0.000401 | 159.6 |

These short workloads include imports/preparation and are subject to timing
variation. Global shape changes still require real rendering; cache/compression
is not assumed to save it. The component and compressed scores differed from
the reference by at most 1.46e-11; aggregate rearrangement reached 0.00374 in
absolute log likelihood across the measured cases, below the adopted 0.01
pilot gate. This does not prove accuracy everywhere in a scientific prior.

For the 64-pixel, two-filter, one-clump three-step optimization, 1/2/4 workers
took 0.98/3.74/6.19 s including setup, with 0.000273/0.001169/0.002050 CPU-hours.
All three returned the same objective to numerical precision. Worker startup
outweighed derivative parallelism. These are bounded steps, not convergence.
Peak child memory is not a simultaneous whole-machine memory measurement.

Fixed-noise reuse reduced a repeated dense-covariance test from 20 Cholesky
factorizations to one (five residual / pixel-score / projection cycles).
The preparation test took 0.00872 s before and 0.00395 s after. Synthetic
reference arrays, fluxes and normalized likelihoods were unchanged bitwise.

Recommended measured defaults: GalSim component rendering, lazy compression,
one derivative worker, controlled inner threads. Keep aggregate and explicit
compression available for matched workload pilots. A compression threshold
must include extra column rendering, whitening and QR build cost:
`build_cost + N_fast * compressed_cost < N_fast * direct_cost`.
No one global threshold is selected from these tiny benchmarks.

Aggregate Sum/Convolve combines transformed extended profiles before drawing.
Native shifted empirical point responses are drawn separately: including them
in the combined FFT failed the initial subpixel comparison. Out-of-field
extended centers and unsupported FFT sizes fall back to component rendering.
No pixel response is applied twice, and no cropped profile is renormalized.
PSF, component, whitened-column and QR caches are bounded. GalSim C++ resource
sizes are only estimated by the cache; resident memory is reported separately.

The broad high-n nested-sampling pilot exceeded its initialization wall budget
and was stopped with its files preserved. Native sampler/derived-column smoke
checks use a separately declared compact synthetic prior. They do not change
real-target priors or establish posterior/evidence convergence. No overall
scientific-inference speedup or converged ESS/s is claimed from short pilots.
A radial Fourier approximation is deferred: the exact CPU path is retained,
and no approximation has passed the required likelihood/recovery gates.

Reproduce the bounded CPU workload with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark_cpu_execution.py --output execution.json --repeats 4 --optimization
python scripts/validate_execution_sampling.py --output pilot --max-dead 40
```

The second command needs the public compiled PolyChord installation and should
be wrapped with scripts/run_with_env.py where that environment is configured.
JAX, DSPS and NUTS remain experimental and gated; this refactor adds no GPU
infrastructure and makes no CPU/GPU inference comparison claim.


The bounded native compact sampling check favored the independent full path
(7.71 s) over the component policy (9.44 s); aggregate with two MPI ranks took
10.09 s. ESS=1 prevents any useful posterior-efficiency comparison. Use full
pixel evaluation as an explicit candidate when global shapes dominate; use the
component policy for partial changes/revisits. The production sampling default
cannot be selected from these unconverged smoke checks alone. They demonstrate
why cache-hit throughput is insufficient for inference-backend selection.
