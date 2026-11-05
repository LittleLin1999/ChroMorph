"""Each component owns its own parameterized spectrum and normalization."""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import multiprocessing
import numpy as np
from astropy import units as u
from astropy.constants import L_sun
from astropy.cosmology import Planck18
from .parameters import Parameter


@dataclass(frozen=True)
class SEDContext:
    redshift: float = 0.0
    distance_mpc: float | None = None
    foreground_av: float = 0.0
    foreground_rv: float = 3.1

    def __post_init__(self):
        if not np.isfinite([self.redshift, self.foreground_av, self.foreground_rv]).all() or self.redshift < 0 or self.foreground_av < 0 or self.foreground_rv <= 0:
            raise ValueError("Invalid shared observational SED context")
        if self.distance_mpc is not None and (not np.isfinite(self.distance_mpc) or self.distance_mpc <= 0):
            raise ValueError("Distance must be finite and positive")

    def distance_cm(self):
        if self.distance_mpc is not None:
            return (self.distance_mpc*u.Mpc).to_value(u.cm)
        if self.redshift == 0:
            raise ValueError("Stellar SED at z=0 requires an explicit positive luminosity distance")
        return Planck18.luminosity_distance(self.redshift).to_value(u.cm)


class FreeBandSED:
    """Independent nonnegative total microJy flux in every filter."""
    def __init__(self, fluxes=None, parameters=None):
        self.initial_fluxes = dict(fluxes or {})
        self.parameters = dict(parameters or {})

    def describe(self):
        return dict(model='free', initial_fluxes=self.initial_fluxes,
                    parameters={n: asdict(p) for n, p in self.parameters.items()})

    def register(self, registry, name, bands):
        unknown = (set(self.initial_fluxes) | set(self.parameters))-set(bands)
        if unknown:
            raise ValueError(f"Unknown FreeBandSED filters: {unknown}")
        for band in bands:
            p = self.parameters.get(band, Parameter(self.initial_fluxes.get(band, 1.), 0, np.inf))
            if p.lower < 0:
                raise ValueError("Free flux lower bound must be nonnegative")
            registry.add(f"{name}.flux.{band}", p)

    def fluxes(self, values, name, observations, context):
        return {o.band: values[f"{name}.flux.{o.band}"] for o in observations}


_FSPS = None


def _initialize_fsps(configuration):
    """Exactly one FSPS state per isolated worker process."""
    global _FSPS
    import fsps
    _FSPS = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                   **configuration)


def _evaluate_fsps(parameters):
    global _FSPS
    age = parameters.pop("tage")
    for key, value in parameters.items():
        _FSPS.params[key] = value
    wave, luminosity = _FSPS.get_spectrum(tage=age, peraa=True)
    return np.array(wave, copy=True), np.array(luminosity, copy=True), float(_FSPS.formed_mass)


class FSPSStellarSED:
    """Independent FSPS model in a dedicated process, mass = formed Msun.

    'young' and 'old' select starting ages only. They do not fix age or tie
    populations. SFH family and attenuation law are independent per instance.
    """
    def __init__(self, sfh="ssp", nebular=False, dust=True, dust_type=2,
                 parameters=None, cache_size=8):
        families = {"ssp": 0, "young": 0, "old": 0, "exponential": 1, "delayed": 4}
        if sfh not in families:
            raise ValueError(f"SFH must be one of {list(families)}")
        self.sfh = sfh
        self.nebular = bool(nebular)
        self.dust = bool(dust)
        self.configuration = dict(sfh=families[sfh], dust_type=int(dust_type),
                                  add_neb_emission=self.nebular, add_neb_continuum=self.nebular)
        age = .01 if sfh == "young" else 3. if sfh == "old" else 1.
        defaults = dict(log_mass=Parameter(8., 2, 13), age=Parameter(age, .001, 13.5),
                        metallicity=Parameter(-.3, -2., .3),
                        av=Parameter(.2 if dust else 0., 0., 5., vary=dust))
        if families[sfh] != 0:
            defaults["tau"] = Parameter(1., .01, 20.)
        if self.nebular:
            defaults["gas_logu"] = Parameter(-2., -4., -1.)
            # gas metallicity follows this component's stellar metallicity;
            # explicit override permits independent gas metallicity.
        self.parameters = defaults
        allowed = set(defaults) | ({"gas_metallicity"} if nebular else set())
        for key, value in (parameters or {}).items():
            if key not in allowed:
                raise ValueError(f"Unknown stellar parameter: {key}")
            if not isinstance(value, Parameter):
                raise TypeError("Stellar parameter overrides must be Parameter instances")
            self.parameters[key] = value
        if not dust and (self.parameters['av'].value != 0 or self.parameters['av'].vary):
            raise ValueError("dust=False requires fixed av=0")
        self._executor = None
        self._executor_configuration = None
        self._cache = OrderedDict()
        self._flux_cache = OrderedDict()
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")

    def describe(self):
        return dict(model='fsps', sfh=self.sfh, nebular=self.nebular, dust=self.dust,
                    configuration=dict(self.configuration), mass_definition='formed_solar_mass',
                    parameters={n: asdict(p) for n, p in self.parameters.items()})

    def register(self, registry, name, bands):
        for key, p in self.parameters.items():
            registry.add(f"{name}.sed.{key}", p)

    def _spectrum(self, parameters):
        key = tuple(sorted(parameters.items())) + tuple(sorted(self.configuration.items()))
        if key not in self._cache:
            configuration_key = tuple(sorted(self.configuration.items()))
            if self._executor is not None and self._executor_configuration != configuration_key:
                self.close()
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_fsps, initargs=(dict(self.configuration),))
                self._executor_configuration = configuration_key
            wave, spec, formed_mass = self._executor.submit(_evaluate_fsps, dict(parameters)).result()
            if formed_mass <= 0 or not np.isfinite(spec).all():
                raise ValueError("FSPS returned invalid spectrum/mass")
            self._cache[key] = (wave, spec/formed_mass)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return self._cache[key]

    def spectrum(self, values, name, context):
        p = {k: values[f"{name}.sed.{k}"] for k in self.parameters}
        if context.redshift > 0 and p['age'] > Planck18.age(context.redshift).to_value(u.Gyr):
            raise ValueError("Stellar age exceeds the age of the Universe at source redshift")
        fsps_parameters = dict(tage=p['age'], logzsol=p['metallicity'],
                               dust2=p['av']/1.0857362047581296 if self.dust else 0.,
                               dust1=0., gas_logz=p.get('gas_metallicity', p['metallicity']),
                               gas_logu=p.get('gas_logu', -2.))
        if 'tau' in p:
            fsps_parameters['tau'] = p['tau']
        wave, luminosity_per_mass = self._spectrum(fsps_parameters)
        observed_wave = wave*(1+context.redshift)
        flam = (luminosity_per_mass * 10**p['log_mass'] * L_sun.cgs.value
                / (4*np.pi*context.distance_cm()**2*(1+context.redshift)))
        if context.foreground_av:
            from dust_extinction.parameter_averages import CCM89
            law = CCM89(Rv=context.foreground_rv)
            inverse_um = 1e4/observed_wave
            covered = (inverse_um >= law.x_range[0]) & (inverse_um <= law.x_range[1])
            observed_wave, flam = observed_wave[covered], flam[covered]
            flam = flam * law.extinguish(observed_wave*u.AA, Av=context.foreground_av)
        return observed_wave, flam

    def fluxes(self, values, name, observations, context):
        key = (name, tuple((k, values[f"{name}.sed.{k}"]) for k in self.parameters),
               tuple(sorted(self.configuration.items())), self.dust, self.nebular, self.sfh, context,
               tuple((o.band, getattr(o.throughput, 'fingerprint', None) or id(o.throughput))
                     for o in observations))
        if key in self._flux_cache:
            return dict(self._flux_cache[key])
        wave, flam = self.spectrum(values, name, context)
        fluxes = {}
        for o in observations:
            if o.throughput is None:
                raise ValueError(f"Physical SED requires a throughput for {o.band}")
            flux = o.throughput.microjy(wave, flam)
            if o.band in fluxes and not np.isclose(fluxes[o.band], flux, rtol=1e-6):
                raise ValueError("Repeated filter has inconsistent throughput; use distinct band identifiers")
            fluxes[o.band] = flux
        self._flux_cache[key] = dict(fluxes)
        if len(self._flux_cache) > self.cache_size:
            self._flux_cache.popitem(last=False)
        return fluxes

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
