# Nuclear SED and statistical flux priors

For N distinct observed filters, each declared nuclear point source owns N
independent `FreeBandSED` flux parameters in microJy. Repeated observations
through the same filter measure that filter's flux. No nuclear template,
power law, continuum interpolation, fixed color, smoothness penalty or
cross-filter spectral parameter is used. Nuclear fluxes cannot be tied.
The unresolved emission is not automatically classified as AGN emission.

The statistical flux prior factorizes:

\[
\pi(F_1,\ldots,F_N)=\prod_{b=1}^{N}\pi_b(F_b).
\]

Each scalar prior can have its own distribution and width. No shared prior
hyperparameter is fitted. An identical, explicitly configured width in two
filters still gives a product of independent priors. Image decomposition can
produce posterior correlations even when the priors are independent.

## Evidence with a physically free nuclear SED

`PolyChordSampler` explicitly samples all varying amplitudes, including the
nucleus. It uses normalized inverse-CDF prior transforms and the full
normalized Gaussian image likelihood; priors are not added a second time to
the likelihood. Variable projection is never substituted for marginalization.
Proper priors for every sampled parameter permit posterior sampling and
Bayesian evidence without a physical nuclear SED model.

For example, the following are illustrative statistical priors, not
recommended widths for a particular target:

```python
from chromorph import PointSource, FreeBandSED, Parameter, Prior

scene.add_component(PointSource(
    name="nucleus", nuclear=True,
    sed=FreeBandSED(parameters={
        "blue": Parameter(1., 0., float("inf"),
                          prior=Prior("uniform", lower=0., upper=20.)),
        "red": Parameter(1., 0., float("inf"),
                         prior=Prior("normal", mean=0., sigma=50.)),
    }),
))
scene.sample_polychord("results/evidence", nlive=300, num_repeats=30, seed=42)
```

The Gaussian is normalized after truncation to nonnegative flux. Its infinite
upper support is compatible with a proper prior. A lognormal prior is also
available; its `mean` and `sigma` describe the natural logarithm of flux in
microJy. No finite upper flux cap is silently supplied by ChroMorph.

The equivalent YAML SED settings are:

```yaml
sed:
  model: free
  parameters:
    blue:
      value: 1.0
      min: 0.0
      prior: {distribution: uniform, lower: 0.0, upper: 20.0}
    red:
      value: 1.0
      min: 0.0
      prior: {distribution: normal, mean: 0.0, sigma: 50.0}
```

Use `chromorph sample-polychord model.yaml --output results/evidence` with
proper priors for the remaining model parameters. No special permission flag
is required for independent nuclear amplitude priors.

`inference.json` records every nuclear prior, its support and normalization
status, `spectral_model: null`, prior factorization, the sampled parameters,
and `inference_mode: posterior_and_evidence`. Report each band flux and its
uncertainty separately. SED figures display discrete band measurements without
a fitted nuclear continuum.

## Maximum likelihood and posterior-only inference

`scene.fit(objective="ml")` freely optimizes each nuclear amplitude and ignores
statistical priors. An explicitly specified hard `Parameter.upper` bound still
applies. To keep ML unbounded above while using a finite Bayesian uniform prior,
put the finite support in `prior`, as in the examples above.

If an improper flat measure over each nonnegative nuclear amplitude is
intentionally chosen, use `nuclear_mode="flat_marginal"` in Python or
`--nuclear-mode flat_marginal` in the CLI. For the supported one-nucleus-per-band
fixed-Gaussian-noise model, eligible flat amplitudes are analytically integrated
and restored with exact conditional draws. An integrable likelihood can give
a proper posterior and credible intervals. An improper prior has arbitrary
normalization, so its absolute evidence and Bayes factors are undefined.
Only this case is labeled `posterior_only` with `log_evidence: null`.

The flat-marginal option preserves explicitly configured proper priors by
sampling those amplitudes. If every amplitude has a proper prior, evidence
remains defined in that mode too. All other sampled parameters must have
proper priors in either mode.

## Width sensitivity and validation

Repeat selected evidence calculations with independently varied prior widths,
using the same images, noise model and other priors. Compare band posteriors,
boundary occupancy and log evidence with independent seeds and live-point
counts. Choose scientific priors before fitting; do not narrow them around
the best fit to increase evidence. An approximately unchanged posterior does
not imply unchanged evidence: doubling two broad uniform widths decreases
log evidence by approximately `2*log(2)` when the likelihood is well inside
both supports.

`scripts/validate_nuclear_evidence.py` runs a two-filter Gaussian-image test
with analytic evidence, multiple prior widths, independent seeds and increased
live-point counts. It uses GalSim rendering and the actual joint likelihood,
and writes the complete run settings and comparison. This validates the toy
model's evidence, not convergence for an arbitrary multi-clump galaxy.

Reference: the public [PolyChordLite example](https://github.com/PolyChord/PolyChordLite/blob/master/run_pypolychord.py)
keeps the normalized likelihood and unit-cube prior transform separate.
