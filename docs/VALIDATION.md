# Validation status

The implementation is tested; this does not establish unique stellar
populations or unbiased decomposition for every real galaxy.

## Core and numerical checks

The current local suite passes 74 tests, with four optional integration modules
skipped. It includes independent component namespaces, flux exchange and label
switching, GalSim rendering, private empirical-PSF injections, exact filter
integration, bounded variable projection, optimizer comparisons and SFH report
assembly. Public CI counts differ when private fixtures or optional libraries
are absent. Synthetic analytic profiles/SEDs are explicitly testing fixtures.

Compressed Gaussian likelihoods are checked against direct pixel residuals,
including masks and fixed non-diagonal covariance. Gaussian normalization is
retained. Linear amplitudes use whitened QR factors and a constrained solver;
normal matrices are not inverted. Ill-conditioned template sets report rank
deficiency. Cached arrays and statistics have explicit byte budgets.

The cached photometry operator is checked against public synphot calculations
for continua and narrow-line spectra. The real four-filter HST comparison is
also verified against direct STSynphot/synphot evaluation. Unvalidated stellar
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

## J1042 and remaining scientific limits

The existing four-filter HST DRC products and empirical_v2 kernels have passed
hash, FITS, sampling, drizzle-context and WCS checks. PSFs are already pixel
integrated; native rendering uses `no_pixel`. Actual target kernels also have
private injection/recovery checks. WHT alone is not treated as a full variance
model. J1042 fitting runs on Erebus and does not overwrite hstpipe products.

Controlled free-band, constant-SFR and SSP comparisons use identical data,
PSFs, masks, uncertainties and observational nuisance conventions. Their
private reports separate morphology, formed-mass and photometric sensitivity
from statistical significance. A physical-looking spectrum is not a model
selection criterion.

Broader mismatch ensembles, correlated-noise uncertainty coverage, posterior
multimodality and prior sensitivity remain necessary before strong science
claims. The optional nuclear flat-prior marginalization defines a posterior
but not an absolute Bayesian evidence; its output explicitly marks evidence
undefined. PolyChord production settings and numerical convergence require
separate validation for selected identifiable models.
