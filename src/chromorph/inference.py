"""PolyChord inference with a physically unconstrained nuclear SED.

PolyChordSampler samples independent band amplitudes with normalized scalar
priors and retains Bayesian evidence. PolyChordPosterior optionally integrates
improper flat nonnegative amplitudes exactly; only this improper-prior case
lacks an absolute evidence. Neither mode imposes a nuclear spectral shape.
"""
from pathlib import Path
import json
import time
from dataclasses import asdict
import numpy as np
from .gaussian import marginal_linear_flux
from scipy.stats import truncnorm
from .provenance import model_signature, content_signature, POSTERIOR_SCHEMA_VERSION, MODEL_SCHEMA_VERSION


def nuclear_prior_metadata(scene, marginalized=()):
    """Record scalar prior measures separately from the absence of an SED model."""
    from .priors import prior_for, prior_spec
    rows = {}
    for name in scene.nuclear_flux_names():
        parameter = scene.registry.parameters[name]
        try:
            dist = prior_for(parameter)
        except ValueError:
            proper = False
            support = [parameter.lower, parameter.upper]
        else:
            proper = True
            support = [dist.lower, dist.upper]
        rows[name] = dict(units='microJy', proper=proper, support=support,
                          prior=asdict(prior_spec(parameter.prior)),
                          treatment='analytically_integrated' if name in marginalized else 'sampled')
    return dict(spectral_model=None, factorization='product of independent per-filter scalar priors',
                cross_filter_prior_correlation=False, fluxes=rows)


class PolyChordPosterior:
    """Optional exact flat-amplitude integration for posterior-only inference."""
    evidence_defined = False
    def __init__(self, scene, *, likelihood='auto'):
        scene.validate_nuclear_seds()
        if likelihood not in {'auto','full','compressed'}:raise ValueError('Unknown likelihood path')
        self.likelihood=likelihood
        self.evaluator=scene.prepare()
        from .spatial import PointSource
        self.scene = scene
        self.linear = [n for c in scene.components
                       if isinstance(c, PointSource) and (c.nuclear or c.name in {'nucleus', 'nuclear'})
                       for n in scene._eligible_linear() if n.startswith(c.name+'.flux.')]
        if len({n.split('.')[-1] for n in self.linear}) != len(self.linear):
            raise ValueError('Analytic nuclear marginalization requires one nuclear amplitude per band')
        self.derived_names=[name+suffix for name in self.linear for suffix in ['.conditional_location','.conditional_scale']]
        self.metrics=dict(restoration_replay_evaluations=0,restored_flux_draws=0)
        self.speed_groups=scene.registry.cost_groups(scene.registry.names(exclude=self.linear))
        self.names=[n for group in self.speed_groups.values() for n in group]
        self.grade_dims=[len(group) for group in self.speed_groups.values() if group]
        self.lower, self.upper = scene.registry.bounds(self.names)
        for name, lo, hi in zip(self.names, self.lower, self.upper):
            if scene.registry.parameters[name].prior is None and not np.isfinite([lo, hi]).all():
                raise ValueError('PolyChord needs a proper prior for '+name)
        from .priors import CubePriorTransform
        self._prior_transform=CubePriorTransform(scene.registry,self.names)
        self.evidence_defined=not bool(self.linear)
        self.flux_measure = ('independent flat F>=0, dF in microJy; no finite upper limit'
                             if self.linear else 'independent normalized scalar priors')

    def prior(self, unit_cube):
        return self._prior_transform(unit_cube)

    def evaluate(self,vector):
        values=self.scene.registry.unpack(vector,self.names)
        return self.evaluator.marginalized_loglikelihood(values,self.linear,mode=self.likelihood)

    def loglikelihood(self, vector):
        value, conditional = self.evaluate(vector)
        return value, [float(x) for name in self.linear for x in conditional[name]]

    def restore_fluxes(self, samples, *, derived=None, seed=0):
        """One exact conditional truncated-normal draw per posterior sample."""
        samples=np.asarray(samples,dtype=float)
        if derived is not None:
            derived=np.asarray(derived,dtype=float)
            if derived.shape!=(len(samples),2*len(self.linear)) or not np.isfinite(derived).all():
                raise ValueError('Invalid stored nuclear conditional columns')
            if np.any(derived[:,1::2]<=0):raise ValueError('Conditional Gaussian scales must be positive')
        rng = np.random.default_rng(seed)
        output = np.empty((len(samples), len(self.names)+len(self.linear)))
        output[:, :len(self.names)] = samples
        for i, row in enumerate(samples):
            if derived is None and self.linear:
                _, conditional = self.evaluate(row)
                self.metrics['restoration_replay_evaluations']+=1
            for j, name in enumerate(self.linear):
                mean, sigma = conditional[name] if derived is None else derived[i,2*j:2*j+2]
                self.metrics['restored_flux_draws']+=1
                output[i, len(self.names)+j] = truncnorm.rvs(-mean/sigma, np.inf,
                    loc=mean, scale=sigma, random_state=rng)
        return self.names+self.linear, output

    def _run_analytic(self,directory,*,restore_samples=2000,**options):
        """No nested sampler is needed when all remaining quantities are conditional."""
        from types import SimpleNamespace
        from astropy.table import Table
        from .provenance import json_value
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        score,conditional=self.evaluate(np.empty(0));count=max(1,int(restore_samples))
        derived=np.tile([x for name in self.linear for x in conditional[name]],(count,1))
        names,draws=self.restore_fluxes(np.empty((count,0)),derived=derived,seed=options.get('seed',0))
        np.savetxt(directory/'posterior.txt',np.column_stack([np.full(count,1/count),np.full(count,-2*score),derived]))
        if names:Table(draws,names=names).write(directory/'posterior_with_nucleus.ecsv',format='ascii.ecsv',overwrite=True)
        report=dict(sampler='analytic_conditional',posterior_schema_version=POSTERIOR_SCHEMA_VERSION,
            model_signature=self.signature(),parameter_names=[],derived_parameter_names=self.derived_names,
            evidence_defined=self.evidence_defined,log_evidence=score if self.evidence_defined else None,
            log_integral_reference_measure=score,likelihood_calls=1,restoration=dict(self.metrics),
            nuclear_sed=nuclear_prior_metadata(self.scene,self.linear))
        (directory/'inference.json').write_text(json.dumps(json_value(report),indent=2)+'\n')
        return SimpleNamespace(logZ=report['log_evidence'],logZerr=0. if self.evidence_defined else None,nlike=1)

    def signature(self):
        return content_signature(dict(model=model_signature(self.scene),sampled_names=self.names,
            derived_names=self.derived_names,linear_measure=self.linear,posterior_schema_version=POSTERIOR_SCHEMA_VERSION))

    def run(self, directory, *, restore_samples=2000, **options):
        """Run public PyPolyChord; each MPI rank owns a separate Scene.

        A best-fit initialization is deliberately not used to truncate priors.
        Pass nlive/num_repeats explicitly for reproducible production settings.
        """
        if not self.names:return self._run_analytic(directory,restore_samples=restore_samples,**options)
        try:
            import pypolychord
            from pypolychord.settings import PolyChordSettings
        except ImportError as exc:
            raise ImportError('Install public PolyChordLite/PyPolyChord with its compiled library') from exc
        if hasattr(self,'grade_dims'):
            options.setdefault('grade_dims',self.grade_dims)
            options.setdefault('grade_frac',[1.]*len(self.grade_dims))
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        metadata = dict(sampler='PolyChord', parameter_names=self.names,
            posterior_schema_version=POSTERIOR_SCHEMA_VERSION,model_schema_version=MODEL_SCHEMA_VERSION,
            derived_parameter_names=self.derived_names,derived_convention='untruncated Gaussian location and scale, microJy',
            marginalized_parameters=self.linear, nuclear_flux_measure=self.flux_measure,
            nuclear_sed=nuclear_prior_metadata(self.scene,self.linear),
            inference_mode='posterior_and_evidence' if self.evidence_defined else 'posterior_only',
            parameter_definitions={n: asdict(p) for n, p in self.scene.registry.parameters.items()},
            ties=self.scene.registry.ties, context=asdict(self.scene.context),
            log_evidence=None, evidence_defined=self.evidence_defined,
            evidence_warning=(None if self.evidence_defined else 'Absolute Bayesian evidence is undefined with flat unbounded nuclear flux priors. PolyChord logZ is only a reference-measure integral.'),
            speed_groups=getattr(self,'speed_groups',None), settings=options)
        from .result import _json
        signature=self.signature()
        metadata['model_signature']=signature
        previous=directory/'inference.json'
        if options.get('read_resume',True) and previous.exists() and list(directory.glob('*.resume')):
            if json.loads(previous.read_text()).get('model_signature')!=signature:
                raise ValueError('Resume model/prior/input signature mismatch; use a new output directory')
        # Every MPI rank receives identical input, but only rank zero writes metadata.
        rank = 0
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.rank
        except ImportError:
            pass
        if rank == 0:
            from .result import _json
            (directory/'inference.json').write_text(json.dumps(_json(metadata), indent=2)+'\n')
        settings = PolyChordSettings(len(self.names), len(self.derived_names), base_dir=str(directory),
                                     file_root='posterior', **options)
        started = time.monotonic()
        def progress(live, dead, logweights, logZ, logZerr):
            if rank == 0:
                status = dict(state='sampling', n_dead=len(dead), n_live=len(live),
                    elapsed_seconds=time.monotonic()-started, log_evidence=float(logZ) if self.evidence_defined else None,
                    log_integral_reference_measure=float(logZ), log_integral_error=float(logZerr))
                (directory/'progress.json').write_text(json.dumps(_json(status), indent=2)+'\n')
        output = pypolychord.run_polychord(self.loglikelihood, len(self.names), len(self.derived_names),
                                          settings, self.prior, progress)
        if rank == 0:
            from astropy.table import Table
            raw = np.atleast_2d(np.loadtxt(directory/'posterior.txt'))
            weights = raw[:, 0]/raw[:, 0].sum()
            expected=2+len(self.names)+len(self.derived_names)
            if raw.shape[1]!=expected:raise ValueError('Posterior column/schema mismatch; use explicit legacy replay reading for old files')
            sampled=raw[:,2:2+len(self.names)]
            derived=raw[:,2+len(self.names):]
            table = Table(sampled, names=self.names)
            for j,name in enumerate(self.derived_names):table[name]=derived[:,j]
            table['weight'] = weights
            table.write(directory/'posterior_weighted.ecsv', format='ascii.ecsv', overwrite=True)
            if restore_samples:
                rng = np.random.default_rng(options.get('seed', 0))
                indices=rng.choice(len(raw),size=restore_samples,p=weights)
                selected=sampled[indices]
                names, restored = (self.restore_fluxes(selected,derived=derived[indices],seed=options.get('seed',0))
                                   if self.linear else (self.names,selected))
                Table(restored, names=names).write(directory/'posterior_with_nucleus.ecsv',
                    format='ascii.ecsv', overwrite=True)
                quantiles = np.quantile(restored, [.16, .5, .84], axis=0)
                (directory/'posterior_summary.json').write_text(json.dumps({n:
                    dict(p16=float(q[0]), median=float(q[1]), p84=float(q[2]))
                    for n, q in zip(names, quantiles.T)}, indent=2)+'\n')
            summary = dict(state='sampling_complete', elapsed_seconds=time.monotonic()-started,
                effective_sample_size=float(1/(weights @ weights)), samples=len(raw),
                log_evidence=float(output.logZ) if self.evidence_defined else None,
                log_evidence_error=float(output.logZerr) if self.evidence_defined else None,
                likelihood_calls=output.nlike, restoration=dict(self.metrics), convergence_validated=False,
                note='Check independent seeds, slice lengths, component exchange and prior sensitivity before scientific interpretation.')
            (directory/'run_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            metadata.update(log_evidence=summary['log_evidence'],log_evidence_error=summary['log_evidence_error'])
            (directory/'inference.json').write_text(json.dumps(_json(metadata),indent=2)+'\n')
        return output


class PolyChordSampler(PolyChordPosterior):
    """Full normalized joint sampling, including every nuisance amplitude.

    No variable-projection objective is used. Independent proper amplitude
    priors must be explicitly configured for Bayesian evidence; the default
    unconstrained nucleus is not assigned an arbitrary normalization here.
    """
    evidence_defined = True

    def __init__(self,scene,*,likelihood='auto',fast_slow=True):
        from .priors import prior_for
        scene.validate_nuclear_seds()
        if likelihood not in {'auto','full','compressed'}:raise ValueError('Unknown likelihood path')
        self.scene=scene;self.linear=[];self.derived_names=[];self.likelihood=likelihood
        self.metrics=dict(restoration_replay_evaluations=0,restored_flux_draws=0)
        self.speed_groups=scene.registry.cost_groups()
        self.names=[n for group in self.speed_groups.values() for n in group] if fast_slow else scene.registry.names()
        self.grade_dims=([len(group) for group in self.speed_groups.values() if group] if fast_slow else [len(self.names)])
        self.lower,self.upper=scene.registry.bounds(self.names)
        for name,lo,hi in zip(self.names,self.lower,self.upper):
            try:prior_for(scene.registry.parameters[name],lo,hi)
            except ValueError as e:
                raise ValueError('Proper independent prior required for '+name+'; no spectral shape is imposed. '+str(e)) from e
        from .priors import CubePriorTransform
        self._prior_transform=CubePriorTransform(scene.registry,self.names)
        self.flux_measure='explicitly configured, independently normalized per-parameter priors'

    def evaluate(self,vector):
        values=self.scene.registry.unpack(vector,self.names)
        return self.scene.loglikelihood(values,mode=self.likelihood),{}

    def measure_speed_hierarchy(self,*,repeats=3):
        """Probe all parameters of each grade together, including all shapes."""
        report={};center=np.full(len(self.names),.5)
        start=time.perf_counter();self.evaluate(self.prior(center))
        report['initialization_seconds']=time.perf_counter()-start
        for group,names in self.speed_groups.items():
            if not names:continue
            indices=[self.names.index(n) for n in names];elapsed=[]
            before=self.scene.metrics['render_calls']
            backends=getattr(self.scene.prepare()._spectral_engine,'backends',{})
            synthesis={c.name:getattr(backends.get(c.name,c.sed),'metrics',{}).get('fsps_calls',0) for c in self.scene.components}
            for i in range(repeats):
                cube=center.copy();cube[indices]+=(i+1)*1e-4
                start=time.perf_counter();self.evaluate(self.prior(cube));elapsed.append(time.perf_counter()-start)
            report[group]=dict(parameters=names,median_seconds=float(np.median(elapsed)),
                render_calls=self.scene.metrics['render_calls']-before,
                fsps_calls={c.name:getattr(backends.get(c.name,c.sed),'metrics',{}).get('fsps_calls',0)-synthesis[c.name] for c in self.scene.components})
            self.evaluate(self.prior(center))
        return report


def sample_emcee(scene, values, *, steps=1000, walkers=None, seed=0):
    """Optional emcee posterior, retaining free nuclear band amplitudes."""
    import emcee
    names = scene.registry.names()
    lo, hi = scene.registry.bounds(names)
    center = np.array([values[n] for n in names])
    ndim = len(names)
    if ndim == 0:
        raise ValueError("No varying parameters")
    walkers = max(2*ndim+2, 32) if walkers is None else walkers
    rng = np.random.default_rng(seed)
    start = np.clip(center + rng.normal(size=(walkers, ndim))*1e-4*np.maximum(abs(center), 1), lo+1e-10, hi-1e-10)
    def logp(x):
        if not ((x >= lo).all() and (x <= hi).all()):
            return -np.inf
        try:
            v = scene.registry.unpack(x, names)
            r = scene.residual(v)
            return -.5*float(r@r)
        except ValueError:
            return -np.inf
    sampler = emcee.EnsembleSampler(walkers, ndim, logp)
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(start, steps, progress=False)
    return sampler
