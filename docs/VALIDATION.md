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
claims. The optional nuclear flat-prior marginalization defines a posterior
but not an absolute Bayesian evidence; its output explicitly marks evidence
undefined. PolyChord production settings and numerical convergence require
separate validation for selected identifiable models.
