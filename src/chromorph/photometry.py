"""Public STSynphot bandpasses and synphot photon-weighted synthetic photometry."""
from __future__ import annotations
import hashlib
import numpy as np
from astropy import units as u


class HSTThroughput:
    def __init__(self, obsmode, *, integration='cached'):
        if integration not in {'cached', 'reference'}:
            raise ValueError('integration must be cached or reference')
        self.integration = integration
        try:
            import stsynphot
        except ImportError as exc:
            raise ImportError("Install chromorph[stellar] and STSynphot reference tables") from exc
        self.obsmode = obsmode
        self.bandpass = stsynphot.band(obsmode)
        wave = self.bandpass.waveset.to_value(u.AA)
        response = self.bandpass(self.bandpass.waveset).value
        self.fingerprint = hashlib.sha256(wave.tobytes()+response.tobytes()).hexdigest()
        self._quadrature = None

    def microjy_reference(self, wavelength_angstrom, flam):
        """Direct public synphot calculation, retained as a validation reference."""
        from synphot import SourceSpectrum, Observation
        from synphot.models import Empirical1D
        from synphot.units import FLAM
        source = SourceSpectrum(Empirical1D, points=np.asarray(wavelength_angstrom)*u.AA,
                                lookup_table=np.asarray(flam)*FLAM, keep_neg=True)
        # No silent extrapolation beyond the supplied physical spectrum.
        observation = Observation(source, self.bandpass, force="none")
        return float(observation.effstim(flux_unit=u.uJy).to_value(u.uJy))

    def microjy(self, wavelength_angstrom, flam):
        """Apply synphot's linear photon interpolation and trapezoidal integral.

        Build the integration operator once per wavelength grid. This changes
        neither the bandpass nor the spectral sampling, including narrow lines.
        A real Observation checks spectral coverage before an operator is cached.
        """
        if getattr(self, 'integration', 'cached') == 'reference':
            return self.microjy_reference(wavelength_angstrom, flam)
        wave, flux = np.asarray(wavelength_angstrom, dtype=float), np.asarray(flam, dtype=float)
        if wave.ndim != 1 or flux.shape != wave.shape or len(wave) < 2:
            raise ValueError('Spectrum requires equal-length one-dimensional wavelength and flux arrays')
        if not np.isfinite(wave).all() or not np.isfinite(flux).all() or np.any(np.diff(wave) <= 0):
            raise ValueError('Spectrum must be finite with strictly increasing wavelengths')
        cached = getattr(self, '_quadrature', None)
        weights = None if cached is None else cached[1]
        if cached is None or not np.array_equal(wave, cached[0]):
            from synphot import SourceSpectrum, Observation
            from synphot.models import Empirical1D
            from synphot.units import FLAM, convert_flux
            source = SourceSpectrum(Empirical1D, points=wave*u.AA,
                                    lookup_table=np.ones_like(wave)*FLAM, keep_neg=True)
            observation = Observation(source, self.bandpass, force='none')
            grid = observation.waveset.to_value(u.AA)
            response = self.bandpass(grid*u.AA).value
            trap = np.zeros_like(grid)
            trap[:-1] += np.diff(grid)/2
            trap[1:] += np.diff(grid)/2
            weighted = trap*response
            # SourceSpectrum interpolates PHOTLAM, proportional to lambda*FLAM.
            right = np.clip(np.searchsorted(wave, grid, side='right'), 1, len(wave)-1)
            left = right-1
            fraction = np.clip((grid-wave[left])/(wave[right]-wave[left]), 0., 1.)
            weights = np.bincount(left, weights=weighted*(1-fraction), minlength=len(wave))
            weights += np.bincount(right, weights=weighted*fraction, minlength=len(wave))
            bandwave = self.bandpass.waveset.to_value(u.AA)
            bandresponse = self.bandpass(bandwave*u.AA).value
            from scipy.integrate import trapezoid
            denominator = abs(trapezoid(bandwave*bandresponse, x=bandwave))
            conversion = convert_flux(self.bandpass.pivot(), 1*FLAM, u.uJy).value
            weights *= wave*conversion/denominator
            self._quadrature = (wave.copy(), weights)
        # Keep this call's operator local: another component may concurrently
        # prepare a different spectral grid using the same immutable bandpass.
        value = float(abs(np.dot(weights, flux)))
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Integrated spectrum must have positive finite flux')
        return value

    @property
    def pivot_angstrom(self):
        return float(self.bandpass.pivot().to_value(u.AA))
