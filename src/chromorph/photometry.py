"""Public STSynphot bandpasses and synphot photon-weighted synthetic photometry."""
from __future__ import annotations
import hashlib
import numpy as np
from astropy import units as u


class HSTThroughput:
    def __init__(self, obsmode):
        try:
            import stsynphot
        except ImportError as exc:
            raise ImportError("Install chromorph[stellar] and STSynphot reference tables") from exc
        self.obsmode = obsmode
        self.bandpass = stsynphot.band(obsmode)
        wave = self.bandpass.waveset.to_value(u.AA)
        response = self.bandpass(self.bandpass.waveset).value
        self.fingerprint = hashlib.sha256(wave.tobytes()+response.tobytes()).hexdigest()

    def microjy(self, wavelength_angstrom, flam):
        from synphot import SourceSpectrum, Observation
        from synphot.models import Empirical1D
        from synphot.units import FLAM
        source = SourceSpectrum(Empirical1D, points=np.asarray(wavelength_angstrom)*u.AA,
                                lookup_table=np.asarray(flam)*FLAM, keep_neg=True)
        # No silent extrapolation beyond the supplied physical spectrum.
        observation = Observation(source, self.bandpass, force="none")
        return float(observation.effstim(flux_unit=u.uJy).to_value(u.uJy))

    @property
    def pivot_angstrom(self):
        return float(self.bandpass.pivot().to_value(u.AA))
