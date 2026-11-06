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

- [emcee EnsembleSampler and random state](https://emcee.readthedocs.io/en/stable/user/sampler/)


CPU-refactor API checks (2026-11-06), against installed GalSim 2.8.5 and
SciPy 1.17.0 signatures:

- [GalSim Sum/Add and Convolve](https://galsim-developers.github.io/GalSim/_build/html/composite.html)
- [PolyChordLite public example and derived dimensions](https://github.com/PolyChord/PolyChordLite/blob/master/run_pypolychord.py)

Online documentation can describe a later patch release; installed signatures
and numerical tests determine compatibility. No undocumented renderer or
sampler method is used.
