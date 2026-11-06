"""Immutable, accuracy-gated FSPS band libraries with independent SED instances."""
from dataclasses import asdict
from itertools import product
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
from .sed import FSPSStellarSED
from .parameters import Parameter
from .cache import ExactLRU

_LOADED_LIBRARIES=ExactLRU(256*1024**2,max_entries=4)


def observational_signature(observations,context):
    return dict(context=asdict(context),bands=[(o.band,getattr(o.throughput,'fingerprint',None)) for o in observations])


def _reference_metadata(reference,values,name,observations,context,axes,elapsed):
    bands=list(dict.fromkeys(o.band for o in observations))
    import importlib.metadata,importlib.util,os,subprocess
    spec=importlib.util.find_spec('fsps');extension=next(Path(spec.origin).parent.glob('_fsps*.so'))
    reference_root=os.environ.get('SPS_HOME')
    data_revision=None
    if reference_root:
        q=subprocess.run(['git','-C',reference_root,'rev-parse','HEAD'],capture_output=True,text=True)
        if q.returncode==0:data_revision=q.stdout.strip()
    metadata=dict(bands=bands,signature=observational_signature(observations,context),
        reference=reference.describe(),fixed_shape={k:values[name+'.sed.'+k] for k in reference.parameters
            if k!='log_mass' and k not in axes and not (k=='age' and 'log_age' in axes)},
        fsps_version=importlib.metadata.version('fsps'),extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        fsps_data_revision=data_revision,mass_definition='formed_solar_mass',build_seconds=elapsed,
        validation=None)
    return metadata


class FrozenDict(dict):
    """JSON-compatible immutable metadata, including nested physical settings."""
    def __init__(self,value):
        dict.__init__(self,((k,FrozenDict(v) if isinstance(v,dict) else
            tuple(FrozenDict(x) if isinstance(x,dict) else x for x in v) if isinstance(v,list) else v)
            for k,v in value.items()))
    def _immutable(self,*args,**kwargs):raise TypeError('Template metadata is immutable; use validate() to replace its report')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_immutable
    def __reduce__(self):return (FrozenDict,(dict(self),))


class BandFluxLibrary:
    """Full filter-integrated spectra; no effective-wavelength dust shortcut.

    Every node is a direct reference FSPS spectrum with that node's dust and
    configured nebular continuum/lines, integrated through the actual filters.
    Log-age is an interpolation coordinate, never a stellar-mass coordinate.
    """
    def __init__(self,axes,flux,metadata):
        self.axes=FrozenDict({k:np.array(v,dtype=float,copy=True) for k,v in axes.items()})
        self.flux=np.array(flux,dtype=float,copy=True);self.metadata=FrozenDict(metadata)
        expected=tuple(len(a) for a in self.axes.values())+(len(metadata['bands']),)
        if self.flux.shape!=expected or not np.isfinite(self.flux).all() or np.any(self.flux<=0):
            raise ValueError('Invalid positive band-flux grid')
        for a in self.axes.values():
            if a.ndim!=1 or len(a)<1 or not np.isfinite(a).all() or np.any(np.diff(a)<=0):raise ValueError('Grid axes must increase')
            a.setflags(write=False)
        self.flux.setflags(write=False)
        self._interpolator=RegularGridInterpolator(tuple(self.axes.values()),np.log(self.flux),bounds_error=True)
        from .provenance import content_signature,array_signature
        self.fingerprint=content_signature(dict(axes={k:array_signature(v) for k,v in self.axes.items()},
            flux=array_signature(self.flux),metadata={k:v for k,v in self.metadata.items() if k not in {'validation','build_seconds'}}))
        self.kind='composite'

    @classmethod
    def build(cls,reference,values,name,observations,context,axes):
        if not axes or any(k=='log_mass' or k not in set(reference.parameters)|{'log_age'} for k in axes):
            raise ValueError('Specify independent SED-shape axes; mass is an exact amplitude')
        bands=list(dict.fromkeys(o.band for o in observations));shape=tuple(len(a) for a in axes.values())+(len(bands),)
        flux=np.empty(shape);started=time.perf_counter()
        for index in product(*(range(len(a)) for a in axes.values())):
            v=dict(values)
            for key,i in zip(axes,index):v[name+'.sed.'+('age' if key=='log_age' else key)]=10**axes[key][i] if key=='log_age' else axes[key][i]
            predicted=reference.unit_fluxes(v,name,observations,context)
            flux[index]=[predicted[b] for b in bands]
        metadata=_reference_metadata(reference,values,name,observations,context,axes,time.perf_counter()-started)
        return cls(axes,flux,metadata)

    def authorizes(self,values,name,observations):
        """Explicit finite-tested image/noise and normalization domain."""
        from .provenance import observation_signature
        domain=(self.metadata.get('validation') or {}).get('validation_domain') or {}
        if tuple(domain.get('observation_signatures',()))!=tuple(observation_signature(o) for o in observations):return False
        scene_bounds=domain.get('scene_parameter_bounds',{})
        if not set(values).issubset(scene_bounds):return False
        def within(x,bounds):
            lo,hi=bounds
            return (-np.inf if lo is None else lo)<=x<=(np.inf if hi is None else hi)
        if not all(within(values[k],bounds) for k,bounds in scene_bounds.items()):return False
        bounds=domain.get('parameter_bounds',{})
        required=['sed.log_mass']+['morphology.'+k for k in ['x','y','re','n','q','pa']]
        if not all(k in bounds for k in required):return False
        return all(np.isfinite(values[name+'.'+k]) and within(values[name+'.'+k],interval) for k,interval in bounds.items())

    def coordinates(self,shape):
        return [np.log10(shape['age']) if k=='log_age' else shape[k] for k in self.axes]

    def predict(self,shape):
        for key,value in self.metadata['fixed_shape'].items():
            if shape[key]!=value:raise ValueError('A fixed template configuration changed: '+key)
        return np.exp(self._interpolator(np.array(self.coordinates(shape)))[0])

    def validate(self,reference,values,name,observations,context,points,*,relative_tolerance=1e-3,
                 likelihood_error=None,delta_chi2_tolerance=.01,loglikelihood_error=None,
                 delta_loglikelihood_tolerance=.005,validation_domain=None):
        """Check off-grid reference calls; image-noise validation is explicit.

        `likelihood_error(predicted,reference,shape)` must bound the image-level
        delta-chi2 for the intended normalization/morphology domain. Without
        this additional gate the library stays experimental for inference.
        """
        if likelihood_error is not None and loglikelihood_error is not None:
            raise ValueError('Choose a nonnegative chi2 bound or signed loglikelihood discrepancy')
        rows=[]
        for shape in points:
            v=dict(values)
            for k,x in shape.items():v[name+'.sed.'+k]=x
            full_shape={k:v[name+'.sed.'+k] for k in reference.parameters if k!='log_mass'}
            exact=reference.unit_fluxes(v,name,observations,context)
            truth=np.array([exact[b] for b in self.metadata['bands']]);predicted=self.predict(full_shape)
            bound=None if likelihood_error is None else float(likelihood_error(predicted,truth,full_shape))
            signed=None if loglikelihood_error is None else float(loglikelihood_error(predicted,truth,full_shape))
            if bound is not None and (not np.isfinite(bound) or bound<0):
                raise ValueError('likelihood_error must return a finite nonnegative chi2 error bound')
            if signed is not None and not np.isfinite(signed):
                raise ValueError('Nonfinite signed loglikelihood discrepancy')
            rows.append(dict(shape=full_shape,relative_error=(predicted/truth-1).tolist(),
                delta_chi2=bound,delta_loglikelihood=signed))
        maximum=max((max(abs(np.array(r['relative_error']))) for r in rows),default=np.inf)
        worst=max((r['delta_chi2'] for r in rows if r['delta_chi2'] is not None),default=None)
        worst_signed=max((abs(r['delta_loglikelihood']) for r in rows if r['delta_loglikelihood'] is not None),default=None)
        image_passed=(worst is not None and worst<=delta_chi2_tolerance) or (worst_signed is not None and worst_signed<=delta_loglikelihood_tolerance)
        passed=bool(rows and np.isfinite(maximum) and maximum<=relative_tolerance and image_passed and validation_domain)

        report=dict(production_ready=passed,photometry_passed=bool(maximum<=relative_tolerance),
            maximum_relative_error=float(maximum),relative_tolerance=relative_tolerance,maximum_delta_chi2=worst,
            delta_chi2_tolerance=delta_chi2_tolerance,finite_validation_is_not_a_proof_over_the_whole_prior=True,
            image_domain_validation_supplied=likelihood_error is not None or loglikelihood_error is not None,
            maximum_absolute_delta_loglikelihood=worst_signed,delta_loglikelihood_tolerance=delta_loglikelihood_tolerance,
            validation_domain=validation_domain,points=rows)
        self.metadata=FrozenDict({**self.metadata,'validation':report})
        return report

    def save(self,path):
        payload={"axis_"+k:v for k,v in self.axes.items()}
        np.savez_compressed(path,flux=self.flux,metadata=json.dumps(__import__('chromorph.provenance',fromlist=['json_value']).json_value(self.metadata),sort_keys=True,allow_nan=False),
            axis_names=np.array(list(self.axes)),kind=self.kind,**payload)

    @property
    def nbytes(self):
        # Include log-flux interpolation storage and SSP cumulative arrays.
        total=2*self.flux.nbytes+sum(a.nbytes for a in self.axes.values())
        if hasattr(self,'_integral'):total+=self._integral.nbytes
        return total

    @classmethod
    def load(cls,path):
        path=Path(path).resolve();stat=path.stat();key=(str(path),stat.st_mtime_ns,stat.st_size)
        cached=_LOADED_LIBRARIES.get(key)
        if cached is not None:return cached
        with np.load(path,allow_pickle=False) as z:
            kind=str(z['kind']);typ=SSPBandLibrary if kind=='ssp_integral' else cls
            result=typ({str(k):z['axis_'+str(k)] for k in z['axis_names']},z['flux'],json.loads(str(z['metadata'])))
        _LOADED_LIBRARIES[key]=result
        return result


class SSPBandLibrary(BandFluxLibrary):
    """Constant SFR from an explicitly normalized SSP age integral.

    The integral uses actual age (Gyr), not log age. Dust and the entire
    configured nebular spectrum are included before filter integration.
    The tiny interval below the youngest SSP is held at that SSP; validation
    against native continuous-SFR FSPS is required before inference.
    """
    def __init__(self,axes,flux,metadata):
        super().__init__(axes,flux,metadata)
        if metadata['reference']['sfh'] not in {'ssp','young','old'} or 'log_age' not in axes:
            raise ValueError('SSP age integration requires a direct SSP reference grid')
        self.kind='ssp_integral';self.ages=10**self.axes['log_age']
        age_axis=list(self.axes).index('log_age')
        self._ssp=np.moveaxis(self.flux,age_axis,0)
        self.age_interpolation=metadata.get('age_interpolation','linear_age_linear_flux')
        if self.age_interpolation=='log_age_linear_flux':
            reshape=(-1,)+(1,)*(self._ssp.ndim-1)
            dt=np.diff(self.ages).reshape(reshape)
            ratio=np.log(self.ages[1:]/self.ages[:-1]).reshape(reshape)
            term=(self.ages[1:]*np.log(self.ages[1:]/self.ages[:-1])-np.diff(self.ages)).reshape(reshape)
            increments=self._ssp[:-1]*dt+(self._ssp[1:]-self._ssp[:-1])/ratio*term
            self._integral=np.concatenate((np.zeros_like(self._ssp[:1]),np.cumsum(increments,axis=0)))+self.ages[0]*self._ssp[0]
        else:
            self._integral=cumulative_trapezoid(self._ssp,x=self.ages,axis=0,initial=0)+self.ages[0]*self._ssp[0]
        self._other={k:a for k,a in self.axes.items() if k!='log_age'}
        self._ssp_interpolation={};self._cumulative_interpolation={}

    @classmethod
    def build_native(cls,reference,values,name,observations,context,axes=None):
        """One public full-age FSPS call per metallicity/dust/gas node.

        FSPS interpolates SSP flux linearly in log age. Integrate that rule
        analytically in actual time, instead of densely resampling spectra.
        Native age-grid photometry is compared with independent FSPS calls.
        """
        axes=dict(axes or {})
        if any(k in {'age','log_age','log_mass'} for k in axes):raise ValueError('Native SSP ages and linear mass are not external grid axes')
        started=time.perf_counter();bands=list(dict.fromkeys(o.band for o in observations));selected=[]
        for b in bands:selected.append(next(o for o in observations if o.band==b))
        output=None;ages=None
        for indices in product(*(range(len(a)) for a in axes.values())):
            v=dict(values)
            for key,i in zip(axes,indices):v[name+'.sed.'+key]=float(axes[key][i])
            wave,spectra,current_age=reference.native_ssp_grid(v,name,context)
            if ages is None:
                ages=current_age;output=np.empty((len(ages),)+tuple(len(a) for a in axes.values())+(len(bands),))
            elif not np.array_equal(ages,current_age):raise ValueError('FSPS changed its age sampling')
            weights=np.stack([o.throughput.integration_weights(wave) for o in selected],axis=1)
            output[(slice(None),*indices,slice(None))]=spectra@weights
        all_axes={'log_age':np.log10(ages),**axes}
        metadata=_reference_metadata(reference,values,name,observations,context,all_axes,time.perf_counter()-started)
        metadata.update(age_interpolation='log_age_linear_flux',native_full_age_output=True)
        return cls(all_axes,output,metadata)

    def _at(self,table,index,shape):
        if not self._other:return table[index]
        axes=tuple(self._other.values())
        cache=self._ssp_interpolation if table is self._ssp else self._cumulative_interpolation
        if index not in cache:cache[index]=RegularGridInterpolator(axes,table[index],bounds_error=True)
        return cache[index]([[shape[k] for k in self._other]])[0]

    def predict(self,shape):
        for key,value in self.metadata['fixed_shape'].items():
            if key!='age' and shape[key]!=value:raise ValueError('A fixed SSP configuration changed: '+key)
        t=shape['age']
        if t<=0 or t>self.ages[-1]:raise ValueError('SF duration outside the SSP age domain')
        if t<=self.ages[0]:return self._at(self._ssp,0,shape)
        i=min(np.searchsorted(self.ages,t,side='right')-1,len(self.ages)-2)
        dt=t-self.ages[i];a=self._at(self._ssp,i,shape);b=self._at(self._ssp,i+1,shape)
        if self.age_interpolation=='log_age_linear_flux':
            integral=self._at(self._integral,i,shape)+dt*a+(b-a)/np.log(self.ages[i+1]/self.ages[i])*(t*np.log(t/self.ages[i])-dt)
        else:
            integral=self._at(self._integral,i,shape)+dt*a+.5*dt**2*(b-a)/(self.ages[i+1]-self.ages[i])
        return integral/t


class PrecomputedFSPSSED(FSPSStellarSED):
    """Independent component shape/cache, with an immutable shared library.

    Invalid/unvalidated/out-of-domain proposals use the exact reference by
    default. Enabling an experimental interpolator is always explicit.
    """
    def __init__(self,library,*,parameters=None,experimental=False,fallback='reference'):
        if isinstance(library,(str,Path)):library=BandFluxLibrary.load(library)
        self.library=library;self.experimental=bool(experimental);self.fallback=fallback
        if fallback not in {'reference','raise'}:raise ValueError('Unknown fallback policy')
        definition=library.metadata['reference']
        sfh='constant' if library.kind=='ssp_integral' else definition['sfh']
        restored={}
        for key,row in definition.get('parameters',{}).items():
            row=dict(row)
            if row.get('lower') is None:row['lower']=-np.inf
            if row.get('upper') is None:row['upper']=np.inf
            restored[key]=Parameter(**row)
        restored.update(parameters or {})
        super().__init__(sfh=sfh,nebular=definition['nebular'],dust=definition['dust'],
            dust_type=definition['configuration']['dust_type'],parameters=restored)
        self.configuration.update(definition['configuration'])
        if library.kind=='ssp_integral':
            self.configuration.update(sfh=1,const=1.,fburst=0.,sf_start=0.,sf_trunc=0.)
        self.metrics.update(template_calls=0,reference_fallbacks=0)

    def _shape_key(self,values,name,context):
        return (super()._shape_key(values,name,context),id(self.library),self.experimental,self.fallback,
                getattr(self,'method',None),id(self.library.metadata.get('validation')),getattr(self,'_domain_authorized',False))

    def describe(self):
        result=super().describe();result.update(backend='precomputed_fsps',library_kind=self.library.kind,
            experimental=self.experimental,library_fingerprint=self.library.fingerprint,validation=self.library.metadata.get('validation'))
        return result

    def unit_fluxes(self,values,name,observations,context):
        try:
            validation=self.library.metadata.get('validation') or {}
            if not self.experimental and not validation.get('production_ready'):raise ValueError('Template accuracy gate not passed')
            # An SED-only query cannot certify an image/mass/morphology domain.
            # The evaluator explicitly authorizes a validated domain; standalone
            # queries remain reference-only unless experimental=True.
            if not self.experimental and not getattr(self,'_domain_authorized',False):
                raise ValueError('Template image-domain authorization is absent')
            current=observational_signature(observations,context)
            # JSON normalizes tuple/list representation after library round trips.
            if json.dumps(current,sort_keys=True)!=json.dumps(self.library.metadata['signature'],sort_keys=True):
                raise ValueError('Template redshift/foreground/bandpass context changed')
            shape={k:values[name+'.sed.'+k] for k in self.parameters if k!='log_mass'}
            if context.redshift>0 and shape['age']>context.age_limit_gyr():raise ValueError('Age exceeds cosmic age')
            key=(name,tuple(shape.items()),context)
            cached=self._flux_cache.get(key)
            if cached is not None:return dict(cached)
            array=self.library.predict(shape)
            result=dict(zip(self.library.metadata['bands'],map(float,array)))
            self._flux_cache[key]=result;self.metrics['template_calls']+=1
            return dict(result)
        except ValueError:
            if self.fallback=='raise':raise
            self.metrics['reference_fallbacks']+=1
            return super().unit_fluxes(values,name,observations,context)


def image_loglikelihood_discrepancy(scene,values,component_name,predicted,reference,bands):
    """Signed exact Gaussian discrepancy for one held-out table query.

    Inputs are band fluxes per formed solar mass. Other components remain in
    the reference full image; covariance whitening is the prepared reference.
    This validation helper is never called by the proposal hot path.
    """
    evaluator=scene.prepare();models,_,fluxes=scene.predict(values)
    mass=10**values[component_name+'.sed.log_mass'];predicted=dict(zip(bands,predicted));reference=dict(zip(bands,reference))
    total=0.
    for o in scene.observations:
        calibration=np.exp(values['observation.'+o.name+'.log_calibration'])/o.ujy_per_rate
        unit=evaluator.unit_template(evaluator.component_named(component_name),o,values)*calibration
        reference_image=models[o.name]+(mass*reference[o.band]-fluxes[component_name][o.band])*unit
        delta=mass*(predicted[o.band]-reference[o.band])*unit
        residual=evaluator.whitened_residual(o,reference_image)
        whitened=evaluator.whiten(o,delta[o.mask])
        total+=float(residual@whitened-.5*(whitened@whitened))
    return total
