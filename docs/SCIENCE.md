# Scientific interpretation and limits

The model predicts each observed image as a background plus the sum of all
PSF-convolved components. Each component's morphology and SED are read from
its own namespace in the same parameter vector. The objective concatenates
all included, uncertainty-weighted pixels and explicit prior residuals.

Physical SEDs are rest-frame FSPS Lsun/Å per formed stellar mass, converted
using luminosity distance and f_lambda_obs = L_lambda_rest /
[4π D_L² (1+z)]. Each component applies its own FSPS dust and stellar/nebular
settings. STSynphot supplies observation-mode HST bandpasses; synphot computes
photon-weighted fluxes. Missing wavelength coverage raises an error. Foreground
CCM89 extinction is applied only over the law's valid wavelength range; a
filter outside it fails coverage checks instead of silently extrapolating.
Cosmic age limits apply independently to component ages. The gas metallicity
defaults to that same component's stellar metallicity, with an independent
component-specific gas-metallicity override available.

Identical co-spatial components can trade flux without changing an image.
Different SED parameters do not make them identifiable by themselves. Four
HST bands are not generally sufficient to determine age, dust, metallicity,
SFH and normalization independently for several overlapping sources. Check
data-Jacobian rank, null directions, posterior correlations, bound hits,
multistart alternatives, label permutations, and injection/recovery. The
reported coordinate-wise prior-information fractions are local curvature
indicators, not a complete basis-independent test of prior dominance.

The covariance uses the full (including projected amplitudes) local Jacobian,
known diagonal noise and explicit Gaussian priors. It is not rescaled to force
reduced chi-square to one. A pseudoinverse is stored; errors along detected
null directions are marked null. Boundaries invalidate symmetric error-bar
interpretation. Relative calibration has a prior when enabled to anchor the
flux/calibration degeneracy. Full posterior sampling and sensitivity runs are
needed for nonlinear results; MCMC convergence is not certified automatically.

The likelihood uses saved fixed uncertainty maps. It does not model drizzle
pixel covariance or refit Poisson variances at every model evaluation. The
BIC-like score uses image chi-square and counts **all** free amplitudes,
including projected ones. It is an informal comparison under correlated noise,
nonregular boundaries and informative priors. It is not evidence that a
physical SED constraint improves true morphology.

PSF normalization changes are recorded through input QA; invalid science/PSF
sampling or provenance is rejected. Measured outer-border energy flags
truncation risk, but cannot infer missing far wings. Same-grid interpolation
checks are numerical tests, not proof that PSF focus/color mismatch is absent.
No empirical PSF is rebuilt for fitting. The deliberate blurred-PSF mismatch
fixture is solely an injection test, never a science PSF product.

The injection/recovery API accepts a different injection Scene and a fresh
fitting Scene factory. Use it to vary injected SFHs outside the fitted family,
PSF and registration, component separation and radius, foreground/calibration
assumptions, and spectral coverage. Quantify biases, uncertainty coverage,
label switching and flux exchange over many noise realizations. A correct
model self-recovery test alone cannot measure model-mismatch bias.
