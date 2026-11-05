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

## Remaining science validation

A/B/C configurations are provided. Actual production-renderer recovery must
pass before applying ChroMorph to J1042. Physical population recovery,
out-of-family FSPS injections, posterior multimodality, label switching,
foreground/PSF/sky sensitivity and correlated-noise coverage need further
experiments. Four bands may leave many population parameters unconstrained;
the implementation reports this rather than forcing distinct populations.

The local synthetic testing SED verifies parameter mapping, isolation and
likelihood coupling, but cannot validate FSPS stellar physics or the HST
absolute calibration. Physical constraints must not be advertised as more
accurate until those experiments establish the relevant conditions.
