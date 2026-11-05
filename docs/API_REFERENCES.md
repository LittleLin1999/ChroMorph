# Documented public APIs

Checked during implementation (2026-11-05):

- [GalSim GSObject, transforms and drawImage](https://galsim-developers.github.io/GalSim/_build/html/gsobject.html)
- [GalSim InterpolatedImage](https://galsim-developers.github.io/GalSim/_build/html/arbitrary.html)
- [GalSim Sérsic profiles](https://galsim-developers.github.io/GalSim/_build/html/gal.html)
- [python-fsps StellarPopulation](https://python-fsps.readthedocs.io/en/latest/stellarpop_api/)
- [STSynphot observation modes](https://stsynphot.readthedocs.io/en/latest/stsynphot/obsmode.html)
- [synphot Observation](https://synphot.readthedocs.io/en/latest/api/synphot.observation.Observation.html)
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

In particular, python-fsps recommends one StellarPopulation instance per
program because its underlying model state is shared. ChroMorph gives every
physical component a distinct process; this is not a shared global SED vector.
