# Validation status

The implementation is tested; this does not establish unique stellar
populations or unbiased decomposition for every real galaxy.

## Core and numerical checks

The synthetic suite includes independent component namespaces, flux exchange and label
switching, GalSim rendering, exact filter
integration, bounded variable projection, optimizer comparisons and SFH report
assembly. Optional integration modules require installed stellar/instrument reference data. Synthetic analytic profiles/SEDs are explicitly testing fixtures.

Compressed Gaussian likelihoods are checked against direct pixel residuals,
including masks and fixed non-diagonal covariance. Gaussian normalization is
retained. Linear amplitudes use whitened QR factors and a constrained solver;
normal matrices are not inverted. Ill-conditioned template sets report rank
deficiency. Cached arrays and statistics have explicit byte budgets.

The cached photometry operator is checked against public synphot calculations
for continua and narrow-line spectra. Unvalidated stellar
template-grid interpolation is not used.

## Live stellar and instrument validation

With `CHROMORPH_FSPS_TESTS=1`, tests exercise actual FSPS process isolation,
independent component metallicity/dust changes, mass scaling and recovery in
deliberately constrained six-filter simulations. Recovery tests explicitly
specify their injected SFH; they do not assert that four HST bands uniquely
recover age, dust, metallicity and mass.

The constant-SFR implementation has passed an independent comparison to native
FSPS using `sfh=1, const=1, fburst=0, sf_start=0`, including spectral normalization
and SFR versus formed mass/duration. Mixed-family tests exercise constant-SFR
and SSP components with isolated mutable backend state.

`CHROMORPH_HST_TESTS=1` enables actual STSynphot filter checks.
`CHROMORPH_STELLAR_TESTS=1` enables joint stellar/HST and deliberate spectral
model-mismatch checks. The CI workflow downloads only public stellar and
instrument reference libraries. Private kernels and fitted products are not
published with the source.

## Real-data validation

Target discovery, upstream reduction QA and private empirical-PSF injection
experiments are external analysis tasks. They are not bundled in this package.
An explicit manifest provides calibrated image arrays, validated uncertainties,
masks, affine geometry and the appropriate empirical PSF. WHT alone is not
assumed to supply a full noise model. Use identical inputs when comparing
free-band and physical stellar fits. A physical-looking SED is not a model
selection criterion.

Broader mismatch ensembles, correlated-noise uncertainty coverage, posterior
multimodality and prior sensitivity remain necessary before strong science
claims. Independent proper nuclear-flux priors allow posterior and absolute
Bayesian evidence calculations while retaining a physically unconstrained
nuclear SED. The optional improper-flat nuclear marginalization defines a
posterior when integrable, and explicitly marks its evidence undefined.
See [nuclear SED and priors](NUCLEAR_SED.md). PolyChord production settings and
numerical convergence require separate validation for selected identifiable
models.


## CPU-first refactor acceptance

The pre-refactor bounded suite passed 88 tests, with four optional modules
skipped. Structural checkpoint f2ee144 passed 91, also with four skipped.
The expanded core suite passes 115, with four optional modules skipped.
Erebus native FSPS validation passes all six opt-in tests (500 s), covering
constant-SFR formed-mass normalization and independent population recovery.
These constrained synthetic recoveries do not establish identifiability in
four-band real data.

New independent comparisons cover aggregate/component rendering with affine
parity, fractional empirical-PSF centers, n=5.5, Re=0.08 pixel, extended wings,
masked correlated covariance, calibration/centroid/registration changes and
repeated observations in one band. Pilot aggregate gates are maximum pixel
error below 0.01 sigma and absolute delta logL below 0.01. Reference precision
is checked without loosening tests. Off-field aggregate failures trigger the
independent component path rather than altering profiles.

A -> B -> A, rejected-state revisits, tiny/zero caches, direct/compressed
switches, uncertainty/data changes and PSF replacement are tested. Dense noise
factorization is counted across residual, likelihood and amplitude roles.
Profiled derivatives are compared against independently built full-pixel
bounded least-squares solutions, including tied centroids and active bounds.
Parallel stellar-mass re-profiling is compared to serial solutions.

Stored untruncated nuclear Gaussian locations/scales produce the same random
flux draws as replay at the same seed, with zero extra renderer/FSPS calls.
Result and posterior schemas, explicit legacy replay, parameter ordering,
configuration migration, lazy heavy imports, workspace serialization and
worker shutdown have regression coverage. Native PolyChord pilots check actual
derived-column parsing and MPI-safe writing; their deliberately limited sample
budgets cannot validate multimodal convergence, interval coverage or evidence.

## Stellar-table failure gate

A deliberately coarse direct constant-SFR 3 x 3 x 3 library over duration,
metallicity and attenuation was tested with nebular lines/continuum and actual
HST throughputs. Held-out queries were not grid nodes. Image checks varied
formed mass and compact versus saved morphology. The pilot failed: maximum
photometric error 0.611 and maximum absolute delta logL about 5.20 million.
It remains reference-fallback only. A coarse library is not an acceptable
production acceleration just because its SFH is physically plausible.

Signed log-likelihood discrepancies use their absolute magnitude. The legacy
likelihood_error callback must return a finite nonnegative chi2 error bound;
negative or nonfinite values are rejected. Production authorization additionally
requires matching input signatures and a declared complete scene/morphology/
mass domain. Out-of-domain queries fall back or raise; they never extrapolate,
clip inputs or alter priors. Shared immutable tables do not tie clump SEDs.
Finite held-out tests are not a proof over the full stated prior.

Reproduce native tests with CHROMORPH_FSPS_TESTS=1 and the installed FSPS data.
The reusable table pilot accepts --config, --result and --output through
scripts/validate_fsps_library.py; target discovery remains outside the package.
Validation outputs belong in an external analysis directory, not the checkout.


## Final real-data and sampler checks

Four live STSynphot flat-microJy bandpass checks pass separately. The legacy
combined HST/FSPS integration module was not rerun; no unavailable optional
backend is represented as a passing test. Wheel and source-distribution builds
succeed, and importing the installed wheel loads neither FSPS, PolyChord, JAX
nor pyplot.

The J1042 saved free-band / constant-SFR / SSP solutions reproduce chi2
26214.2497559671 / 26224.3037526060 / 26227.3357100345. At the saved state and
an inward, in-bounds change of all morphology parameters, aggregate maximum
absolute delta logL is 0.00389 / 0.00579 / 0.00050 respectively. Maximum pixel
differences are 0.00067 / 0.00100 / 0.00141 sigma. These are bounded comparisons,
not a validation over every possible prior state. A 45-second soft-budget free
fit stopped without convergence at chi2 26214.2497402881; saved science results
and input products were not overwritten. Cold physical initialization took
about 122 seconds per SFH; a subsequent all-morphology change took 8.6 seconds.
No SED assumptions were changed for timing.

Native compact synthetic PolyChord checks sample 10 parameters and store four
conditional columns. Component/full/aggregate-MPI pilots take approximately
9.44/7.71/10.09 seconds, with zero restoration replay and 3000 restored nuclear
flux draws each. All have weight ESS=1 at the deliberate 40-dead-point cap:
credible intervals, recovery and effective posterior samples/second cannot be
inferred from them. Aggregate MPI rank-zero CPU time is not total MPI CPU cost.
The earlier broad-prior initialization was aborted and remains a recorded
failure; no real-target prior was narrowed.

Proper-prior two-filter evidence is checked independently against its analytic
integral with 64 and 192 live points and two seeds. Absolute log-evidence errors
are 0.0732 and 0.1911 at 64 live points, and 0.0100 and 0.00214 at 192; all are
within 0.64 reported standard errors. This validates normalized toy evidence,
not galaxy model selection or posterior coverage in blended clumps.
