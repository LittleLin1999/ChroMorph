# Validation status

This is an initial implementation, not a validated J1042 science result.

## Local environment

- Python 3.9, NumPy 1.26.4, SciPy 1.12, Astropy 6, Matplotlib 3.8.2.
- Core tests run with a documented analytic Gaussian renderer and an analytic
  testing SED. These are testing fixtures, not fallback science models.
- GalSim is not installed locally. FSPS is not installed/configured. STSynphot
  throughput reference tables are not installed. Installing from PyPI and
  accessing Erebus failed because network/DNS access is unavailable in the
  shell. No GalSim or physical-SED result is claimed from these local tests.
- Actual FSPS integration tests are opt-in with
  `CHROMORPH_STELLAR_TESTS=1 pytest -m integration`, after installing libraries
  and HST reference data. They check independent live evaluators and spectra.

## GitHub CI

[Initial production-renderer CI](https://github.com/LittleLin1999/ChroMorph/actions/runs/42011897276)
passed on Python 3.10 and 3.12: **25 passed, 2 skipped** per matrix job.
Both source distribution and wheel builds passed. The additional four tests
exercise actual GalSim rendering: empirical-kernel round trip/no extra pixel
convolution, subpixel affine centroids, aperture losses, and joint moving
clump plus independent nuclear-band recovery. These use synthetic PSFs.
Local J1042 kernels remain unpublished pending data publication approval.

## Real-data discovery

The requested `/home/linxiaojing/data/lrd-squared/hst` is not accessible from
this macOS session. A local mirror was inspected instead. J1042 has four DRC
filters F275W, F336W, F621M, F689M and measured z=0.6079969853. The selected
101×101 fit provides noise maps, masks, sky, affine transforms and registration.
Hashes, PSF/image sampling, drizzle context and weight replay, FITS checksums,
and WCS affine agreement passed automated discovery checks. Saved raw-image
WHT is not automatically treated as the full uncertainty.

The four copied empirical_v2 PSF test fixtures retain their source hashes and
fractional centers. They include no science image pixels. Tests inject
synthetic sources with realistic median saved per-band noise. They include
PSF-only, PSF+Sérsic, several clumps, overlap, compact sources, independent
centroids, astrometry, interpolation and deliberate PSF mismatch.

## Live FSPS validation

[Live stellar CI](https://github.com/LittleLin1999/ChroMorph/actions/runs/42012960020)
passed all three FSPS tests using downloaded public FSPS reference libraries.
They verify process isolation, component-specific metallicity/dust changes,
linear mass scaling, and joint age/dust/mass recovery for separated and
overlapping components. These deliberately constrained simulations use six
synthetic top-hat filters, known morphology, fixed independently chosen
metallicities, and high S/N. They do **not** validate HST absolute throughputs
or imply that J1042's four bands constrain the same quantities.

## Final regression and real-throughput checks

[Final source CI](https://github.com/LittleLin1999/ChroMorph/actions/runs/42016303397)
passed the expanded GalSim/core tests on Python 3.10 and 3.12, four flat-f_nu
normalization tests using actual STSynphot HST bandpasses, and two live stellar
checks: separate populations through those HST bandpasses, plus a deliberate
out-of-family young-nebular injection fitted with an old non-nebular template.
The constrained wrong family leaves significant image residuals; the free-band
baseline fits the same synthetic images. The FSPS recovery job on the final
implementation separately passed three tests in
[this run](https://github.com/LittleLin1999/ChroMorph/actions/runs/42014971161).
Later commits add calibration/mismatch tests without changing implementation.

Private target kernels and fitted products were not published. Real-PSF
injection tests remain local and could not run without local GalSim. A live
J1042 A/B/C fit and its science interpretation have not been produced.

## Remaining science validation

A/B/C configurations are provided. Actual target-PSF recovery must
pass before applying ChroMorph to J1042. Broader population recovery and
out-of-family FSPS injection ensembles, posterior multimodality, label switching,
foreground/PSF/sky sensitivity and correlated-noise coverage need further
experiments. Four bands may leave many population parameters unconstrained;
the implementation reports this rather than forcing distinct populations.

The local synthetic testing SED verifies parameter mapping, isolation and
likelihood coupling, but cannot validate FSPS stellar physics or the HST
absolute calibration. Physical constraints must not be advertised as more
accurate until those experiments establish the relevant conditions.
