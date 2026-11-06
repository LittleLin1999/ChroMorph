"""Optional differentiable SSP synthesis with the same FSPS band library.

DSPS age weights are used explicitly, rather than assuming that its stored
line-luminosity fields reproduce FSPS nebular continuum and gas dependence.
All dust, lines and continuum are already included in the reference SSP nodes.
"""
from itertools import product
import numpy as np
from ..templates import SSPBandLibrary,PrecomputedFSPSSED
from ..sed import FSPSStellarSED


class DSPSBackend(PrecomputedFSPSSED):
    def __init__(self,library,*,parameters=None,experimental=False,method='dsps',fallback='reference'):
        if not isinstance(library,SSPBandLibrary):raise ValueError('Use an FSPS-generated SSPBandLibrary')
        if method not in {'dsps','piecewise_linear'}:raise ValueError('Unknown SSP integration method')
        super().__init__(library,parameters=parameters,experimental=experimental,fallback=fallback)
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64',True)
        self.jax,self.jnp=jax,jnp;self.method=method
        ages=jnp.asarray(library.ages);ssp=jnp.asarray(library._ssp);cumulative=jnp.asarray(library._integral)
        other={k:jnp.asarray(v) for k,v in library._other.items()}
        from ..sed import SEDContext
        cosmic_age=SEDContext(**library.metadata['signature']['context']).age_limit_gyr()
        active_fraction=jnp.geomspace(1e-6,1.,1024)
        def interpolate(table,shape):
            answer=0.
            brackets=[]
            for k,axis in other.items():
                if len(axis)==1:brackets.append((0,0,0.));continue
                i=jnp.clip(jnp.searchsorted(axis,shape[k],side='right')-1,0,len(axis)-2)
                fraction=(shape[k]-axis[i])/(axis[i+1]-axis[i]);brackets.append((i,i+1,fraction))
            for corner in product([0,1],repeat=len(other)):
                weight=1.;indices=[]
                for bit,(lo,hi,f) in zip(corner,brackets):
                    indices.append(hi if bit else lo);weight*=f if bit else 1-f
                answer=answer+weight*table[(slice(None),*indices,slice(None))]
            return answer
        if method=='dsps':
            from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
            from dsps.constants import T_TABLE_MIN
            self.max_duration=cosmic_age-float(T_TABLE_MIN)
        def prediction(shape):
            spectra=interpolate(ssp,shape);duration=shape['age']
            if method=='dsps':
                # DSPS expects cosmic formation time. Its minimum birth time
                # must not shorten a young population's requested SF duration.
                onset=cosmic_age-duration
                edge=jnp.maximum(1e-8,duration*1e-6)
                time_grid=jnp.concatenate((jnp.array([T_TABLE_MIN,onset-edge]),onset+duration*active_fraction))
                # Unit SFR avoids contaminating old zero-SFR intervals with
                # the library's absolute SFR floor; PDF weights normalize mass.
                sfr=jnp.concatenate((jnp.zeros(2),jnp.ones_like(active_fraction)))
                weights=calc_age_weights_from_sfh_table(time_grid,sfr,jnp.log10(ages),cosmic_age)
                return weights@spectra
            integral=interpolate(cumulative,shape)
            i=jnp.clip(jnp.searchsorted(ages,duration,side='right')-1,0,len(ages)-2)
            dt=duration-ages[i]
            if library.age_interpolation=='log_age_linear_flux':
                value=(integral[i]+dt*spectra[i]+(spectra[i+1]-spectra[i])/jnp.log(ages[i+1]/ages[i])*(duration*jnp.log(duration/ages[i])-dt))/duration
            else:
                value=(integral[i]+dt*spectra[i]+.5*dt**2*(spectra[i+1]-spectra[i])/(ages[i+1]-ages[i]))/duration
            return jnp.where(duration<=ages[0],spectra[0],value)
        self.predict_jax=jax.jit(prediction)
        self.metrics.update(dsps_calls=0)

    def unit_fluxes(self,values,name,observations,context):
        # A NumPy/PolyChord adapter. Differentiable samplers call predict_jax
        # directly after independently validating the model/prior domain.
        if not self.experimental:
            validation=self.library.metadata.get('validation_'+self.method) or {}
            if not validation.get('production_ready'):
                if self.fallback=='raise':raise ValueError('Differentiable SSP accuracy gate not passed')
                self.metrics['reference_fallbacks']+=1
                return FSPSStellarSED.unit_fluxes(self,values,name,observations,context)
        from ..templates import observational_signature
        import json
        if json.dumps(observational_signature(observations,context),sort_keys=True)!=json.dumps(self.library.metadata['signature'],sort_keys=True):
            raise ValueError('Differentiable SSP observational context changed')
        shape={k:values[name+'.sed.'+k] for k in self.parameters if k!='log_mass'}
        # This checks bounds and fixed dust/gas/metallicity configurations.
        self.library.predict(shape)
        predicted=np.asarray(self.predict_jax(shape))
        if not np.isfinite(predicted).all():raise ValueError('Nonfinite differentiable SSP prediction; accuracy gate failed')
        self.metrics['dsps_calls']+=1
        return dict(zip(self.library.metadata['bands'],map(float,predicted)))

    def describe(self):
        result=super().describe();result.update(backend='differentiable_ssp',integration=self.method)
        return result
