# CPU cost per reliable posterior sample: experimental sampling

The production sampler is not changed. No full J1042 PolyChord production run
is launched by this work. Short pilots and failed gates are retained.

## One exact posterior target

`chromorph.posterior.ExactPosteriorTarget` owns the original normalized joint
Gaussian image likelihood, configured normalized priors in registered
coordinates and eligible exact nonnegative nuclear-amplitude integrals.
`PolyChordPosterior` delegates to this same target. Basic posterior mathematics
imports neither PolyChord nor plotting/training libraries.

Every extended component keeps its independent morphology and SED. Morphology
is constant across filters, not between proposals. Nuclear fluxes remain
independent per filter, with the selected flat nonnegative reference measure
integrated exactly. Conditional untruncated Gaussian location/scale are retained
for restoring individual nuclear photometry without replay. An improper flat
measure leaves absolute evidence undefined. Explicit proper nuclear priors
remain sampled, with evidence defined; no caps or half-normal defaults are added.
Other sampled priors must be proper, as in the existing PolyChord interface.
The target does not substitute a variable-projection objective for a posterior.

`target.logpdf(vector)` includes the configured prior. Population/nested
adapters use its normalized prior transform and marginalized likelihood
separately, so density Jacobians are not applied twice. Stellar log_mass remains
log10 formed mass. No clump population, metallicity, dust, gas or SFH setting is
changed to obtain timings.

## Transactional execution

`TransactionalEvaluator` keeps immutable raw/whitened component columns for the
accepted state. Proposals rerender only genuinely changed spatial keys, including
all affected components under a tie. SED/amplitude changes reuse columns;
registration changes update the affected observations. Proposals can change
all components simultaneously.

A persistent fixed-order sum tree updates changed contributions and ancestors.
The residual is the fixed whitened data minus the canonical tree root. This
avoids cumulative `residual += delta` roundoff while still updating only changed
branches. Rejection drops candidate frames and restores accepted spectral query
state; acceptance commits atomically. Bounded pure spectral memoization may
retain rejected calculations but never changes the accepted model. Only one
transaction can be outstanding. Fixed-input or prior changes require a new
transaction evaluator, preventing an old score from being compared with a new
target. Tests cover A -> B -> A, rejected states, simultaneous changes, masks,
correlated covariance, tied parameters, calibration and nuclear centroids.

## Blocked Metropolis-within-Gibbs prototype

`chromorph.blocked.BlockedMetropolis` uses random-scan blocks:

- Independent morphology and SED/flux blocks per component.
- Shared registration, calibration and background groups where active.
- Occasional coupled morphology moves over multiple components.
- Symmetric full-parameter exchanges between compatible component namespaces.
- Symmetric +/-180-degree PA moves, accepted using the actual prior.
- Optional component-level morphology-plus-SED moves via
  `joint_component_weight`; disabled by default until benchmarked.

Gaussian proposals and involutive exchanges are symmetric. The complete joint
posterior ratio is used, including different component priors. Warmup adapts
proposal scales and estimates shrinkage covariances; all tuning stops before
retained draws. Rejections remain in the chain. Initial states and proposal
scales do not alter priors. No component is permanently subtracted or fixed.

```python
from chromorph.posterior import ExactPosteriorTarget
from chromorph.blocked import BlockedMetropolis, BlockedResult

target = ExactPosteriorTarget(scene)
prototype = BlockedMetropolis(target)
result = prototype.run(scene.registry.initial(target.names),
                       warmup=500, draws=500, seed=20,
                       max_wall_seconds=120)
result.save(output_directory)
loaded = BlockedResult.load(output_directory, target=target)
names, photometry = target.restore_fluxes(
    loaded.samples, derived=loaded.derived, seed=21)
```

This example is a bounded pilot, not a production configuration or convergence
claim. Chain schema 1 stores named sampled and derived columns plus a stable
model/input/prior signature. Loading is supported; production MCMC checkpoint
resume is not implemented. Each process owns its target and resources.

## pocoMC and PolyChord comparison

The optional pocoMC adapter samples the same target in its normalized prior
cube. Exact likelihoods remain in NumPy/GalSim/FSPS, not a learned surrogate.
Flow training only changes proposal efficiency. No importance weights are
trimmed. Nuclear conditional metadata use the public scalar blob API because
pocoMC 1.2.6 disallows vectorized likelihoods with blobs. Restoration requires
no new model evaluations. Live evaluators are not shared with pocoMC pools;
use isolated outer processes. Optional dependencies are lazy and CPU-only
installation remains sufficient.

PolyChord retains its real slow/medium/fast grades and uses the same lazy
execution capability. Comparisons distinguish full-recomputation reference
runs from optimized auto-path runs. PolyChord remains available for evidence
and multimodal problems; no evidence is reported for improper-flat targets.

The benchmark watchdog includes imports, initialization, training, worker
startup and shutdown. Complete process CPU time includes joined FSPS children.
Timed-out pilots additionally use sampled process-family CPU as a lower-bound
cost estimate. Peak family RSS is sampled aggregate RSS, not unique shared-memory
PSS. Outer-process concurrency reduces wall time without pretending to reduce
total CPU-hours.

## Reliability and measured scope

Predeclared gates: four independently initialized chains, rank-normalized split
R-hat <=1.01, minimum bulk/tail ESS >=400, plus explicit mode checks. A gate pass
on a synthetic model does not establish J1042 convergence. Importance-weight
ESS is not autocorrelation ESS. Between-run mean-ESS estimates for populations
are reported with their very wide finite-replicate uncertainty and are not
used alone to select a production sampler. Incomplete runs cannot pass gates.

In the declared overlapping two-clump analytic-color test, four 48,000-draw
chains pass with maximum R-hat 1.005 and minimum ESS 1,255. Incremental and full
recomputation have the same sampled chains. Total CPU cost is 0.01657 versus
0.02644 CPU-hours: about 1.60 times more effective samples per CPU-hour, or
37% lower CPU cost. The declared Gaussian profile family is a synthetic
regression model, not a shortcut applied to J1042. A separate GalSim case allows
all six morphology parameters to vary for both clumps. The one-dimensional
quadrature/coverage audit tests the kernel only and must not be presented as
full-morphology or physical-population coverage.

All four J1042 blocked cost pilots retain all 48 active parameters and finish
64 transitions (16 warmup plus 48 retained) in about 136--230 seconds of target
execution, including cold physical initialization. The same-seed full-render
reference reaches only six retained draws within its soft budget. These short
chains do not establish independent samples or mode coverage. Full prior-based
pocoMC/PolyChord initialization pilots are bounded and may time out; their
absence of a reliable posterior is an explicit result, not a speedup claim.
No J1042 production sampler is selected on these measurements alone.

## GalSim/Fourier investigation

`profile_spatial_fourier.py` measures constructor time, first/repeated public
kValue calls, native drawing, reusable empirical-PSF k-grid preparation and
inverse FFTs. First kValue includes lazy radial/Hankel work; Python profiling
cannot separate every C++ FFT/Hankel operation. For the tested high-n J1042
profile, first kValue takes about 0.43 seconds and native drawing another 1.31.
The 256-grid approximation fails its likelihood-error gate. A 512 grid passing
selected-state tests is not a prior-wide validation. No Fourier approximation,
parameter rounding, PSF reconstruction or second pixel response is enabled.
Native GalSim remains the accuracy reference.

## Reproducible commands

Use the existing installed checkout, one inner BLAS/OpenMP thread, and an
optional CPU sampling environment. Do not install a GPU dependency into the
minimal environment. On Linux, CPU PyTorch wheels can be installed from the
official CPU wheel index before the `sampling` extra.

```bash
python -m pytest -q -p no:cacheprovider
python scripts/benchmark_posterior.py --output OUTPUT/toy \
  --methods blocked full_blocked pocomc polychord --seeds 20 21 22 23
python scripts/assess_posterior_benchmark.py OUTPUT/toy --exchangeable
python scripts/validate_posterior_coverage.py --output OUTPUT/coverage
python scripts/profile_spatial_fourier.py --output OUTPUT/fourier \
  --config CONFIG.yaml --saved-result RESULT.json
```

Real-target inputs are explicit external manifests/configurations; no private
target paths enter the runtime package. Persisted reports contain complete
commands, signatures, per-block work, samples, priors and failed pilot status.
See the external delivery report for exact machine-readable results and limits.

Public API references:
[pocoMC blobs](https://pocomc.readthedocs.io/en/latest/blobs.html),
[pocoMC sampler](https://pocomc.readthedocs.io/en/latest/api/sampler.html),
[ArviZ ESS](https://python.arviz.org/en/v0.23.4/api/generated/arviz.ess.html),
[GalSim GSObject](https://galsim-developers.github.io/GalSim/_build/html/gsobject.html).


## Additional measured gates and failures

Four additional full-prior-initialized chains on the overlapping analytic-color
model pass with maximum R-hat 1.004 and minimum ESS 1,169. Adding joint component
moves at relative weight 0.5 passes this model too, but yields about 64,245
minimum effective samples per CPU-hour versus 75,752 for the simpler blocked
kernel. That option remains disabled by default.

The fully varying 16-dimensional GalSim synthetic model does NOT pass: maximum
R-hat 1.15 and minimum ESS 24 at the tested budget. The 48-dimensional J1042
pilots likewise do not establish mode coverage. Small-budget pocoMC populations
on the exchangeable synthetic model show label-mode probabilities from 0.116
to 0.972 across independent seeds, where symmetry requires 0.5. These deliberately
limited flow-training/particle settings do not establish the limits of pocoMC;
they fail the adopted validation gate and cannot justify a production choice.
PolyChord population weights are likewise not independent-chain ESS.

In the conditional known-geometry audit, 11 of 12 injected truths lie within
nominal 90% intervals. The binomial 95% interval is approximately [0.615,0.998],
so this small audit is compatible with 90% coverage but imprecise. Maximum
R-hat is 1.0012 and minimum ESS 2,718; MCMC interval endpoints differ from
independent quadrature by at most 0.029 flux units. This is a kernel audit,
not coverage validation for all free morphology or independent FSPS populations.

The final source reproduces the recorded synthetic chain bitwise. The matched
J1042 full/incremental prefix is bitwise identical as well. Vectorized exact
nuclear restoration preserves the scalar loop's random stream: 80,000 draws
using J1042 conditional parameters take 0.019 seconds rather than 11.8 seconds,
about 631 times faster for restoration alone. No rendering or FSPS is performed
by either stored-conditional restoration path.

No production sampler is selected for J1042. Transactional blocked evaluation
is the validated experimental CPU candidate for further pilots; full-model
mixing, morphology--SED dependence, posterior coverage and mode recovery must
pass before any production recommendation. No approximation or narrower prior
is used to force these gates to pass.
