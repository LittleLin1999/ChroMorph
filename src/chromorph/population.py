"""Optional pocoMC adapter over the exact common posterior target.

Sampling in the prior cube preserves every configured normalized prior,
including lognormal and tied-root measures. No profile likelihood is used.
"""
import time
import numpy as np


class SamplingBudgetExceeded(RuntimeError):pass


class PocoMCPosterior:
    def __init__(self,target,*,max_wall_seconds=None):
        self.target=target;self.max_wall_seconds=max_wall_seconds
        self.calls=0;self.started=None;self.likelihood_seconds=0.

    def loglikelihood(self,cubes):
        cubes=np.atleast_2d(cubes);answers=[]
        for cube in cubes:
            if self.max_wall_seconds is not None and time.perf_counter()-self.started>self.max_wall_seconds:
                raise SamplingBudgetExceeded('Explicit sampler pilot wall budget exhausted')
            if np.any(cube<0) or np.any(cube>1):answers.append(-np.inf);continue
            vector=self.target.prior(cube);start=time.perf_counter()
            score=self.target.evaluate(vector)[0]
            self.likelihood_seconds+=time.perf_counter()-start;self.calls+=1;answers.append(score)
        return np.asarray(answers)

    def loglikelihood_with_conditionals(self,cube):
        if self.max_wall_seconds is not None and time.perf_counter()-self.started>self.max_wall_seconds:
            raise SamplingBudgetExceeded('Explicit sampler pilot wall budget exhausted')
        vector=self.target.prior(cube);start=time.perf_counter()
        score,conditional=self.target.evaluate(vector)
        self.likelihood_seconds+=time.perf_counter()-start;self.calls+=1
        return (score,*(x for name in self.target.linear for x in conditional[name]))

    def run(self,directory,*,n_total=512,n_effective=128,n_active=64,seed=0,restore_samples=1000,**options):
        from pathlib import Path
        import json
        import pocomc
        from scipy.stats import uniform
        from .provenance import json_value
        if options.get('pool') is not None:raise NotImplementedError('Use isolated outer processes; live evaluators are not shared with pocoMC pools')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        self.started=time.perf_counter()
        blobs=bool(self.target.linear)
        sampler=pocomc.Sampler(prior=pocomc.Prior([uniform(0.,1.) for _ in self.target.names]),
            likelihood=self.loglikelihood_with_conditionals if blobs else self.loglikelihood,
            vectorize=not blobs,blobs_dtype=[(n,float) for n in self.target.derived_names] if blobs else None,
            n_effective=n_effective,n_active=n_active,
            random_state=seed,pytorch_threads=1,output_dir=str(directory/'states'),**options)
        sampler.run(n_total=n_total,n_evidence=0,progress=False)
        # Do not trim weights or discard low-weight posterior modes.
        posterior=sampler.posterior(resample=False,trim_importance_weights=False,return_blobs=blobs)
        cubes,weights,logl,logp_cube=posterior[:4]
        derived=np.column_stack([posterior[4][n] for n in self.target.derived_names]) if blobs else np.empty((len(cubes),0))
        samples=np.asarray([self.target.prior(c) for c in cubes]);weights=np.asarray(weights);weights/=weights.sum()
        np.savez_compressed(directory/'population.npz',samples=samples,weights=weights,
            loglikelihood=logl,names=np.asarray(self.target.names),derived=derived,derived_names=np.asarray(self.target.derived_names))
        if restore_samples and blobs:
            ids=np.random.default_rng(seed).choice(len(samples),size=restore_samples,p=weights)
            names,restored=self.target.restore_fluxes(samples[ids],derived=derived[ids],seed=seed)
            np.savez_compressed(directory/'posterior_with_nucleus.npz',names=np.asarray(names),samples=restored)
        report=dict(sampler='pocomc',version=pocomc.__version__,model_signature=self.target.signature(),
            derived_names=self.target.derived_names,restoration=dict(self.target.metrics),n_total=n_total,n_effective=n_effective,n_active=n_active,seed=seed,options=options,
            wall_seconds=time.perf_counter()-self.started,likelihood_seconds=self.likelihood_seconds,
            likelihood_calls=self.calls,importance_weight_ess=float(1/(weights@weights)),
            evidence_defined=self.target.evidence_defined,evidence_estimated=False,convergence_validated=False,
            note='Particle weight ESS is not autocorrelation ESS or independent-run/mode validation. No nuclear-prior normalization is invented.')
        (directory/'population.json').write_text(json.dumps(json_value(report),indent=2,allow_nan=False)+'\n')
        return samples,weights,report
