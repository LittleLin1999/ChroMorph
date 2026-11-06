"""Sampler-independent posterior with exact nuclear-amplitude integration.

Density is in registered physical coordinates, never a profile objective.
An improper flat nuclear measure leaves absolute evidence undefined.
"""
import numpy as np
from scipy.stats import truncnorm
from .provenance import model_signature,content_signature,POSTERIOR_SCHEMA_VERSION


class ExactPosteriorTarget:
    evidence_defined=False
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
        self.lower, self.upper = scene.registry.bounds(self.names)
        for name, lo, hi in zip(self.names, self.lower, self.upper):
            if scene.registry.parameters[name].prior is None and not np.isfinite([lo, hi]).all():
                raise ValueError('A proper remaining prior is required for '+name)
        from .priors import CubePriorTransform
        self._prior_transform=CubePriorTransform(scene.registry,self.names)
        self._prepare_prior_density()
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
        if samples.shape!=(len(samples),len(self.names)):raise ValueError('Sample columns do not match the target')
        output=np.empty((len(samples),len(self.names)+len(self.linear)))
        output[:,:len(self.names)]=samples
        if self.linear:
            if derived is None:
                derived=np.empty((len(samples),2*len(self.linear)))
                for i,row in enumerate(samples):
                    _,conditional=self.evaluate(row)
                    derived[i]=[x for name in self.linear for x in conditional[name]]
                    self.metrics['restoration_replay_evaluations']+=1
            mean=derived[:,0::2];sigma=derived[:,1::2]
            # One row-major uniform sequence, matching the former scalar loop.
            output[:,len(self.names):]=truncnorm.rvs(-mean/sigma,np.inf,loc=mean,scale=sigma,random_state=rng)
            self.metrics['restored_flux_draws']+=len(samples)*len(self.linear)
        return self.names+self.linear,output

    def signature(self):
        return content_signature(dict(model=model_signature(self.scene),sampled_names=self.names,
            derived_names=self.derived_names,linear_measure=self.linear,posterior_schema_version=POSTERIOR_SCHEMA_VERSION))

    def _prepare_prior_density(self):
        self._other_priors=[(i,d) for i,d in enumerate(self._prior_transform.priors) if d.spec.distribution!='uniform']
        self._flat_logdensity=-sum(np.log(d.upper-d.lower) for d in self._prior_transform.priors if d.spec.distribution=='uniform')

    def logprior(self, vector):
        vector=np.asarray(vector,dtype=float)
        if vector.shape!=(len(self.names),) or not np.isfinite(vector).all():return -np.inf
        if np.any(vector<self.lower) or np.any(vector>self.upper):return -np.inf
        return float(self._flat_logdensity+sum(d.logpdf(vector[i]) for i,d in self._other_priors))

    def logpdf(self, vector):
        prior=self.logprior(vector)
        if not np.isfinite(prior):return -np.inf
        return prior+self.evaluate(vector)[0]
