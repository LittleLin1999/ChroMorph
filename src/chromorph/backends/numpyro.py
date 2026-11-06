"""Optional NUTS for an explicitly validated differentiable joint model."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,time
import numpy as np


class NumPyroNUTSSampler:
    """Same component definitions and independent priors as the CPU model.

    The experimental FFT renderer must pass likelihood and gradient checks
    over the intended prior domain before run(). No profile likelihood is
    substituted for marginalization. NUTS does not estimate Bayesian evidence.
    """
    def __init__(self,scene,renderer,*,nuclear_mode='sample',reference_scene=None):
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import log_ndtr
        from ..inference import PolyChordSampler,PolyChordPosterior
        from ..sed import FreeBandSED
        from ..spatial import PointSource
        jax.config.update('jax_enable_x64',True)
        self.scene,self.renderer=scene,renderer;self.jax,self.jnp=jax,jnp
        self.nuclear_mode=nuclear_mode
        if nuclear_mode=='sample':self.posterior=PolyChordSampler(scene,likelihood='full')
        elif nuclear_mode=='flat_marginal':self.posterior=PolyChordPosterior(scene,likelihood='full')
        else:raise ValueError('Unknown nuclear mode')
        reference_scene=scene if reference_scene is None else reference_scene
        self.reference_posterior=(PolyChordSampler(reference_scene,likelihood='full') if nuclear_mode=='sample' else PolyChordPosterior(reference_scene,likelihood='full'))
        self.names=self.posterior.names;self.linear=self.posterior.linear
        if self.reference_posterior.names!=self.names:raise ValueError('Reference and accelerated parameter definitions differ')
        self.lower,self.upper=scene.registry.bounds(self.names)
        self._validated=None;self.preparation_seconds=0.
        started=time.perf_counter();self._data={}
        for o in scene.observations:
            if o.covariance is not None:raise ValueError('The optional NUTS adapter currently requires fixed diagonal noise')
            renderer.prepare(o)
            included=np.flatnonzero(o.mask.ravel())
            self._data[o.name]=(jnp.asarray(included),jnp.asarray(o.data.ravel()[included]),jnp.asarray(o.sigma.ravel()[included]),
                -.5*float(len(included)*np.log(2*np.pi)+2*np.log(o.sigma.ravel()[included]).sum()))
        for c in scene.components:
            if isinstance(c.sed,FreeBandSED):continue
            if not hasattr(c.sed,'predict_jax'):raise ValueError('Every physical SED needs a validated differentiable backend')
            axes=c.sed.library.axes
            for n,lo,hi in zip(self.names,self.lower,self.upper):
                if not n.startswith(c.name+'.sed.') or n.endswith('.log_mass'):continue
                key=n.split('.')[-1]
                axis='log_age' if key=='age' else key
                if axis not in axes:raise ValueError('A varying SED parameter is absent from the template axes: '+n)
                a,b=axes[axis][0],axes[axis][-1]
                if axis=='log_age':a,b=10**a,10**b
                if lo<a or hi>b:raise ValueError('Prior exceeds validated template support: '+n)
                if key=='age' and hasattr(c.sed,'max_duration') and hi>=c.sed.max_duration:
                    raise ValueError('SF-duration prior exceeds DSPS cosmic-time support')
        self.preparation_seconds=time.perf_counter()-started
        band_parameters={n.split('.')[-1]:n for n in self.linear}
        points=jnp.asarray([isinstance(c,PointSource) for c in scene.components])
        indices={c.name:i for i,c in enumerate(scene.components)}
        def evaluate(vector):
            values={n:p.value for n,p in scene.registry.parameters.items()}
            values.update(zip(self.names,vector))
            for alias in scene.registry.ties:values[alias]=values[scene.registry.resolve(alias)]
            for name in self.linear:values[name]=0.
            morph=jnp.stack([jnp.stack([jnp.asarray(values[c.name+'.morphology.'+key]) if key in c.parameters else jnp.asarray(default)
                for key,default in [('x',0.),('y',0.),('re',1.),('n',1.),('q',1.),('pa',0.)]]) for c in scene.components])
            fluxes={}
            for c in scene.components:
                if isinstance(c.sed,FreeBandSED):fluxes[c.name]={b:values[c.name+'.flux.'+b] for b in scene.bands}
                else:
                    shape={k:values[c.name+'.sed.'+k] for k in c.sed.parameters if k!='log_mass'}
                    unit=c.sed.predict_jax(shape)*10**values[c.name+'.sed.log_mass']
                    fluxes[c.name]=dict(zip(c.sed.library.metadata['bands'],unit))
            total=0.;bands={};conditional={}
            for o in scene.observations:
                shift=jnp.stack([jnp.asarray(values[f'observation.{o.name}.{a}']) for a in ['dx','dy']])
                templates=renderer.render_jax(morph,points,o,shift)
                coefficient=jnp.exp(values[f'observation.{o.name}.log_calibration'])/o.ujy_per_rate
                amplitudes=jnp.stack([jnp.asarray(fluxes[c.name][o.band])*coefficient for c in scene.components])
                model=jnp.einsum('i,iyx->yx',amplitudes,templates)+values[f'observation.{o.name}.background']
                included,data,sigma,normalization=self._data[o.name]
                residual=(data-model.ravel()[included])/sigma;total+=normalization
                parameter=band_parameters.get(o.band)
                if parameter is None:total-=.5*jnp.dot(residual,residual)
                else:
                    column=templates[indices[parameter.split('.')[0]]].ravel()[included]*coefficient/sigma
                    rr,cc=bands.setdefault(o.band,([],[]));rr.append(residual);cc.append(column)
            for band,(residuals,columns) in bands.items():
                residual=jnp.concatenate(residuals);column=jnp.concatenate(columns)
                precision=jnp.dot(column,column);sigma=1/jnp.sqrt(precision);mean=jnp.dot(column,residual)/precision
                remaining=residual-column*mean
                total+=-.5*jnp.dot(remaining,remaining)+.5*jnp.log(2*jnp.pi)+jnp.log(sigma)+log_ndtr(mean/sigma)
                conditional[band_parameters[band]]=jnp.stack([mean,sigma])
            return total,conditional
        self._evaluate=jax.jit(evaluate)

    def signature(self):
        from ..result import _json
        return hashlib.sha256(json.dumps(_json(dict(names=self.names,parameters={n:asdict(p) for n,p in self.scene.registry.parameters.items()},
            renderer=self.renderer.cache_token(),context=asdict(self.scene.context),ties=self.scene.registry.ties,
            components=[(c.name,c.sed.describe()) for c in self.scene.components],inputs=[(o.name,hashlib.sha256(o.data.tobytes()+o.sigma.tobytes()+o.mask.tobytes()+o.psf.image.tobytes()).hexdigest(),o.psf.center,o.psf.pixel_scale,o.pixel_scale,o.ujy_per_rate,o.matrix.tolist(),o.offset.tolist()) for o in self.scene.observations])),sort_keys=True).encode()).hexdigest()

    def validate(self,*,samples=16,seed=71,loglike_tolerance=.05,gradient_tolerance=.03):
        rng=np.random.default_rng(seed);vectors=[self.scene.registry.initial(self.names)]
        vectors.extend(self.posterior.prior(cube) for cube in rng.uniform(.001,.999,size=(samples,len(self.names))))
        start=time.perf_counter();self._evaluate(self.jnp.asarray(vectors[0]))[0].block_until_ready();compile_seconds=time.perf_counter()-start
        differences=[]
        for vector in vectors:
            reference=self.reference_posterior.evaluate(vector)[0];value=float(self._evaluate(self.jnp.asarray(vector))[0])
            differences.append(value-reference)
        vector=vectors[0];_,gradient=self.jax.value_and_grad(lambda x:self._evaluate(x)[0])(self.jnp.asarray(vector))
        finite=[]
        for i,value in enumerate(vector):
            step=min(1e-4*max(abs(value),1.),(value-self.lower[i])/2,(self.upper[i]-value)/2)
            if not np.isfinite(step) or step<=0:finite.append(np.nan);continue
            plus=vector.copy();minus=vector.copy();plus[i]+=step;minus[i]-=step
            finite.append((self.reference_posterior.evaluate(plus)[0]-self.reference_posterior.evaluate(minus)[0])/(2*step))
        from ..optimize import parameter_scales
        scale=parameter_scales(self.names,vector);finite=np.array(finite)
        gradient_error=np.abs(np.asarray(gradient)-finite)*scale/(1+np.abs(finite)*scale)
        passed=bool(np.isfinite(differences).all() and np.isfinite(gradient_error).all()
            and max(abs(np.array(differences)))<=loglike_tolerance and max(gradient_error)<=gradient_tolerance)
        report=dict(passed=passed,signature=self.signature(),prior_predictive_checks=len(vectors),
            maximum_loglikelihood_difference=float(max(abs(np.array(differences)))),maximum_scaled_gradient_error=float(max(gradient_error)),
            loglike_tolerance=loglike_tolerance,gradient_tolerance=gradient_tolerance,first_compile_seconds=compile_seconds,
            finite_checks_do_not_prove_global_accuracy=True)
        self._validated=report
        return report

    def run(self,directory,*,num_warmup=200,num_samples=400,num_chains=2,seed=41,target_accept_prob=.85,
            max_tree_depth=8,preprocessing_seconds=0.,restore_samples=200):
        if not self._validated or not self._validated['passed'] or self._validated['signature']!=self.signature():
            raise ValueError('Differentiable likelihood/gradient accuracy gate must pass before NUTS')
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC,NUTS,init_to_value
        from numpyro.diagnostics import summary
        from ..priors import prior_for
        from ..result import _json
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        priors=[]
        for name,lo,hi in zip(self.names,self.lower,self.upper):
            p=prior_for(self.scene.registry.parameters[name],lo,hi);spec=p.spec
            if spec.distribution=='uniform':d=dist.Uniform(p.lower,p.upper)
            else:
                low,high=p.lower,p.upper
                if spec.distribution=='lognormal':low=-np.inf if low<=0 else np.log(low);high=np.log(high)
                base=(dist.Normal(spec.mean,spec.sigma) if not np.isfinite([low,high]).any()
                      else dist.TruncatedNormal(spec.mean,spec.sigma,low=low,high=high))
                d=dist.TransformedDistribution(base,dist.transforms.ExpTransform()) if spec.distribution=='lognormal' else base
            priors.append(d)
        def model():
            vector=self.jnp.stack([numpyro.sample(name,p) for name,p in zip(self.names,priors)])
            numpyro.factor('joint_images',self._evaluate(vector)[0])
        initial={n:float(self.scene.registry.parameters[n].value) for n in self.names}
        sampler=MCMC(NUTS(model,init_strategy=init_to_value(values=initial),target_accept_prob=target_accept_prob,max_tree_depth=max_tree_depth),
            num_warmup=num_warmup,num_samples=num_samples,num_chains=num_chains,chain_method='sequential',progress_bar=False)
        key1,key2=self.jax.random.split(self.jax.random.PRNGKey(seed))
        start=time.perf_counter();cpu_start=time.process_time();t=time.perf_counter()
        sampler.warmup(key1,extra_fields=('num_steps','potential_energy'))
        warmup=time.perf_counter()-t;t=time.perf_counter()
        sampler.run(key2,extra_fields=('num_steps','potential_energy'))
        sampling=time.perf_counter()-t
        samples={n:np.asarray(x) for n,x in sampler.get_samples(group_by_chain=True).items()}
        diagnostics=summary(samples,prob=.68,group_by_chain=True)
        flat=np.column_stack([samples[n].reshape(-1) for n in self.names])
        from astropy.table import Table
        Table(flat,names=self.names).write(directory/'posterior.ecsv',format='ascii.ecsv',overwrite=True)
        np.savez_compressed(directory/'chains.npz',**samples)
        if self.linear and restore_samples:
            from scipy.stats import truncnorm
            rng=np.random.default_rng(seed);selected=flat[rng.choice(len(flat),size=min(restore_samples,len(flat)),replace=False)]
            output=np.empty((len(selected),len(self.names)+len(self.linear)));output[:,:len(self.names)]=selected
            for i,row in enumerate(selected):
                _,conditional=self._evaluate(self.jnp.asarray(row))
                for j,name in enumerate(self.linear):
                    mean,sigma=map(float,conditional[name]);output[i,len(self.names)+j]=truncnorm.rvs(-mean/sigma,np.inf,loc=mean,scale=sigma,random_state=rng)
            Table(output,names=self.names+self.linear).write(directory/'posterior_with_nucleus.ecsv',format='ascii.ecsv',overwrite=True)
        elapsed=time.perf_counter()-start;extra=sampler.get_extra_fields(group_by_chain=True)
        minimum_ess=min(float(x['n_eff']) for x in diagnostics.values());max_rhat=max(float(x['r_hat']) for x in diagnostics.values())
        total=elapsed+self.preparation_seconds+preprocessing_seconds+self._validated['first_compile_seconds']
        report=dict(sampler='NumPyro NUTS',seed=seed,parameters=self.names,marginalized_nuclear_fluxes=self.linear,
            diagnostics=diagnostics,warmup_seconds=warmup,sampling_seconds=sampling,postprocessing_seconds=elapsed-warmup-sampling,
            cpu_seconds=time.process_time()-cpu_start,total_seconds_including_preprocessing=total,
            preprocessing_seconds=preprocessing_seconds,renderer_preparation_seconds=self.preparation_seconds,
            accuracy_gate=self._validated,minimum_ess=minimum_ess,maximum_rhat=max_rhat,
            effective_samples_per_second=minimum_ess/total,divergences=int(np.asarray(extra['diverging']).sum()),
            leapfrog_steps=int(np.asarray(extra['num_steps']).sum()),log_evidence=None,
            note='NUTS is not an evidence estimator; it may miss separated modes and component exchange.')
        (directory/'inference.json').write_text(json.dumps(_json(report),indent=2,allow_nan=False)+'\n')
        return report
