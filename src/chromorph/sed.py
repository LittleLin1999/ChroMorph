"""Each component owns its own parameterized spectrum and normalization."""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from functools import lru_cache
import multiprocessing
import numpy as np
from astropy import units as u
from astropy.constants import L_sun
from astropy.cosmology import Planck18
from .parameters import Parameter
from .cache import ExactLRU

_SOLAR_LUMINOSITY = L_sun.cgs.value


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

    @lru_cache(maxsize=128)
    def distance_cm(self):
        if self.distance_mpc is not None:
            return (self.distance_mpc*u.Mpc).to_value(u.cm)
        if self.redshift == 0:
            raise ValueError("Stellar SED at z=0 requires an explicit positive luminosity distance")
        return Planck18.luminosity_distance(self.redshift).to_value(u.cm)

    @lru_cache(maxsize=128)
    def age_limit_gyr(self):
        return Planck18.age(self.redshift).to_value(u.Gyr)

    @lru_cache(maxsize=128)
    def luminosity_to_observed_flam(self):
        return _SOLAR_LUMINOSITY/(4*np.pi*self.distance_cm()**2*(1+self.redshift))


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


def _evaluate_native_age_grid(parameters):
    global _FSPS
    parameters=dict(parameters);parameters.pop('tage',None)
    for key,value in parameters.items():_FSPS.params[key]=value
    wave,spectra=_FSPS.get_spectrum(tage=0.,peraa=True)
    age=10**(np.asarray(_FSPS.log_age)-9)
    mass=np.asarray(_FSPS.formed_mass)
    if mass.ndim==0:mass=np.full(len(age),mass)
    return np.array(wave,copy=True),np.array(spectra,copy=True),np.array(age,copy=True),np.array(mass,copy=True)


class FSPSStellarSED:
    """Independent FSPS model in a dedicated process, mass = formed Msun.

    'young' and 'old' select starting ages only. They do not fix age or tie
    populations. SFH family and attenuation law are independent per instance.
    """
    def __init__(self, sfh="constant", nebular=False, dust=True, dust_type=2,
                 parameters=None, cache_size=8, cache_bytes=32*1024**2):
        families = {"ssp": 0, "young": 0, "old": 0, "constant": 1, "exponential": 1, "delayed": 4}
        if sfh not in families:
            raise ValueError(f"SFH must be one of {list(families)}")
        self.sfh = sfh
        self.nebular = bool(nebular)
        self.dust = bool(dust)
        self.configuration = dict(sfh=families[sfh], dust_type=int(dust_type),
                                  add_neb_emission=self.nebular, add_neb_continuum=self.nebular)
        if self.nebular:
            self.configuration['nebemlineinspec'] = True
        if sfh == 'constant':
            # const is the fraction of formed mass in continuous constant SFR.
            # No exponential or burst contribution, and no irrelevant free tau.
            self.configuration.update(const=1., fburst=0., sf_start=0., sf_trunc=0.)
        elif sfh in {'exponential', 'delayed'}:
            self.configuration.update(const=0., fburst=0., sf_start=0., sf_trunc=0.)
        age = .01 if sfh == "young" else 3. if sfh == "old" else 1.
        defaults = dict(log_mass=Parameter(8., 2, 13), age=Parameter(age, .001, 13.5),
                        metallicity=Parameter(-.3, -2., .3),
                        av=Parameter(.2 if dust else 0., 0., 5., vary=dust))
        if sfh in {'exponential', 'delayed'}:
            defaults["tau"] = Parameter(1., .01, 20.)
        if self.nebular:
            defaults["gas_logu"] = Parameter(-2., -4., -1., vary=False)
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
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.cache_bytes = int(cache_bytes)
        self._cache = ExactLRU(self.cache_bytes//3, self.cache_size)
        self._observed_cache = ExactLRU(self.cache_bytes//3, self.cache_size)
        self._flux_cache = ExactLRU(self.cache_bytes//3, self.cache_size)
        self.metrics = dict(fsps_calls=0, photometry_calls=0)

    def prepared_copy(self):
        """Process-local execution state; immutable libraries may be shared."""
        from copy import copy
        other=copy(self)
        other.parameters=dict(self.parameters);other.configuration=dict(self.configuration)
        other._executor=None;other._executor_configuration=None
        for name in ['_cache','_observed_cache','_flux_cache']:
            setattr(other,name,getattr(self,name).empty_like())
        other.metrics={key:0 for key in self.metrics}
        return other

    def describe(self):
        return dict(model='fsps', sfh=self.sfh, nebular=self.nebular, dust=self.dust,
                    configuration=dict(self.configuration), mass_definition='formed_solar_mass',
                    age_interpretation=('duration_of_continuous_star_formation_gyr'
                                        if self.sfh == 'constant' else 'population_age_gyr'),
                    cache_size=self.cache_size, cache_bytes=self.cache_bytes,
                    parameters={n: asdict(p) for n, p in self.parameters.items()})

    def __getstate__(self):
        # A spawned optimizer worker must never inherit an executor or a live
        # mutable FSPS model. Immutable cached spectra/fluxes may be copied.
        state = dict(self.__dict__)
        state['_executor'] = None
        state['_executor_configuration'] = None
        return state

    def register(self, registry, name, bands):
        for key, p in self.parameters.items():
            registry.add(f"{name}.sed.{key}", p)

    def _ensure_executor(self):
        configuration_key = tuple(sorted(self.configuration.items()))
        if self._executor is not None and self._executor_configuration != configuration_key:
            self.close()
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_fsps, initargs=(dict(self.configuration),))
            self._executor_configuration = configuration_key

    def _spectrum(self, parameters):
        import time
        key = tuple(sorted(parameters.items())) + tuple(sorted(self.configuration.items()))
        cached = self._cache.get(key)
        if cached is None:
            self._ensure_executor()
            started=time.perf_counter()
            wave, spec, formed_mass = self._executor.submit(_evaluate_fsps, dict(parameters)).result()
            self.metrics['fsps_ipc_seconds']=self.metrics.get('fsps_ipc_seconds',0.)+time.perf_counter()-started
            self.metrics['fsps_calls'] += 1
            if formed_mass <= 0 or not np.isfinite(spec).all():
                raise ValueError("FSPS returned invalid spectrum/mass")
            cached = (wave, spec/formed_mass)
            for array in cached: array.setflags(write=False)
            self._cache[key] = cached
        return cached

    def _shape_key(self, values, name, context):
        return (tuple((k, values[f'{name}.sed.{k}']) for k in self.parameters if k != 'log_mass'),
                tuple(sorted(self.configuration.items())), self.dust, self.nebular, self.sfh, context)

    def _physical_parameters(self,values,name,context):
        p = {k: values[f"{name}.sed.{k}"] for k in self.parameters if k != 'log_mass'}
        if context.redshift > 0 and p['age'] > context.age_limit_gyr():
            raise ValueError("Stellar age exceeds the age of the Universe at source redshift")
        fsps_parameters = dict(tage=p['age'], logzsol=p['metallicity'],
                               dust2=p['av']/1.0857362047581296 if self.dust else 0.,
                               dust1=0., gas_logz=p.get('gas_metallicity', p['metallicity']),
                               gas_logu=p.get('gas_logu', -2.))
        if 'tau' in p:
            fsps_parameters['tau'] = p['tau']
        return fsps_parameters

    def _observed_spectrum(self,wave,luminosity_per_mass,context):
        observed_wave = wave*(1+context.redshift)
        flam = luminosity_per_mass*context.luminosity_to_observed_flam()
        if context.foreground_av:
            from dust_extinction.parameter_averages import CCM89
            law = CCM89(Rv=context.foreground_rv)
            inverse_um = 1e4/observed_wave
            covered = (inverse_um >= law.x_range[0]) & (inverse_um <= law.x_range[1])
            observed_wave, flam = observed_wave[covered], flam[...,covered]
            flam = flam * law.extinguish(observed_wave*u.AA, Av=context.foreground_av)
        return observed_wave,flam

    def spectrum_per_mass(self, values, name, context):
        key = self._shape_key(values, name, context)
        cached = self._observed_cache.get(key)
        if cached is not None:
            return cached
        fsps_parameters=self._physical_parameters(values,name,context)
        wave, luminosity_per_mass = self._spectrum(fsps_parameters)
        observed_wave,flam=self._observed_spectrum(wave,luminosity_per_mass,context)
        observed_wave.setflags(write=False);flam.setflags(write=False)
        cached = (observed_wave, flam)
        self._observed_cache[key] = cached
        return cached

    def native_ssp_grid(self,values,name,context):
        """Public FSPS full-age output, normalized to formed stellar mass."""
        if self.sfh not in {'ssp','young','old'}:raise ValueError('Native SSP grids require an SSP reference')
        self._ensure_executor()
        wave,spectra,ages,mass=self._executor.submit(_evaluate_native_age_grid,
            self._physical_parameters(values,name,context)).result()
        self.metrics['fsps_calls']+=1
        if spectra.shape!=(len(ages),len(wave)) or np.any(mass<=0):raise ValueError('Invalid native FSPS age grid')
        observed,flam=self._observed_spectrum(wave,spectra/mass[:,None],context)
        return observed,flam,ages

    def spectrum(self, values, name, context):
        wave, flam = self.spectrum_per_mass(values, name, context)
        return wave, flam*10**values[f'{name}.sed.log_mass']

    def unit_fluxes(self, values, name, observations, context):
        key = (name, self._shape_key(values, name, context),
               tuple((o.band, getattr(o.throughput, 'fingerprint', None) or id(o.throughput))
                     for o in observations))
        cached = self._flux_cache.get(key)
        if cached is not None:
            return dict(cached)
        wave, flam = self.spectrum_per_mass(values, name, context)
        fluxes = {}
        for o in observations:
            if o.throughput is None:
                raise ValueError(f"Physical SED requires a throughput for {o.band}")
            flux = o.throughput.microjy(wave, flam)
            self.metrics['photometry_calls'] += 1
            if o.band in fluxes and not np.isclose(fluxes[o.band], flux, rtol=1e-6):
                raise ValueError("Repeated filter has inconsistent throughput; use distinct band identifiers")
            fluxes[o.band] = flux
        self._flux_cache[key] = dict(fluxes)
        return fluxes

    def fluxes(self, values, name, observations, context):
        mass = 10**values[f'{name}.sed.log_mass']
        return {b: mass*f for b, f in self.unit_fluxes(values, name, observations, context).items()}

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
