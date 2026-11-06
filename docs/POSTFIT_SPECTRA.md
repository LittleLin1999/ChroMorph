# Low-resolution fitting and high-resolution post-fit spectra

Fit the images with the configured low-resolution FSPS library. After fitting,
evaluate a separate high-resolution build at exactly the saved component ages,
SFHs, formed masses, metallicities, attenuation, nebular parameters and shared
observational context. This operation does not optimize or rescale parameters,
change spatial templates, or replace the likelihood backend.

python-fsps 0.5.0 defaults to C3K_LR. Its published source supports building
C3K_HR using `FFLAGS="-DC3K_LR=0 -DC3K_HR=1 -DMILES=0"`. Keep the resulting
package in an isolated environment or target directory. Do not replace a
running fitter's compiled library. Each clump still owns its independent FSPS
process. The expected library is checked before spectral evaluation.

Run `chromorph.postfit.export_postfit_spectra(scene, best_values, output,
reference_fluxes)` in the high-resolution environment. This writes per-clump
observed-frame FITS spectra, exact-key disk caches and a band-flux comparison.
The low-resolution best-fit band fluxes are supplied explicitly. Cache keys
include the compiled backend hash, component configuration, SED parameters,
context and filter fingerprints. A high-resolution spectrum cannot silently
reuse a low-resolution cache entry.

The high-resolution library is not assumed to integrate identically to the
fitting library. Review the saved fractional band-flux changes, especially
for emission lines near throughput edges. Do not silently renormalize curves
to force agreement. A large discrepancy can motivate a higher-resolution
likelihood validation or refit; the post-fit export itself is not that refit.

The display can retain white, colored-edge points for the low-resolution
likelihood predictions while drawing the high-resolution curves. Filled
component photometry may be taken from a free-band decomposition of the same
images, with its source and local covariance errors explicitly recorded. These
decomposition estimates are not an independent dataset. The nuclear PSF retains
independent band amplitudes and never receives a stellar spectral curve.

## Nuclear diagnostic subtraction

`chromorph.spectral_resolution` provides flux-conserving bin integration,
Gaussian-equivalent widths from DESI resolution diagonals, and differential
broadening before subtracting an extended model. The template's intrinsic
width and target-bin response are included in the variance budget. If the
instrument is sharper than the template, both spectra are compared at a
common broader resolution; no deconvolution is performed. Gaussian moment
matching does not reproduce all non-Gaussian structure of a measured LSF.

The PSF zoom can display the resulting total-minus-extended spectrum together
with independent nuclear band points. This is a display diagnostic, not a
physical nuclear SED fit. Record the actual instrument resolution source;
nominal slit-limited values must be identified as approximate. Masked regions,
aperture/slit losses, calibration or epoch differences, and unpropagated stellar
population uncertainties can affect the subtraction. Negative residuals are
retained on a linear axis. Individual stellar panels use continuum-based
y-limits; narrow line peaks can lie outside the zoom but remain in the overview.

Reference: [python-fsps installation and library selection](https://python-fsps.readthedocs.io/en/stable/installation/).
