"""Version-aware saved results and posterior columns, without live resources."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .parameters import Parameter
from .provenance import RESULT_SCHEMA_VERSION,POSTERIOR_SCHEMA_VERSION,json_value


@dataclass(frozen=True)
class SavedResult:
    document: dict

    @classmethod
    def read(cls,path):
        path=Path(path);path=path/'result.json' if path.is_dir() else path
        document=json.loads(path.read_text());version=document.get('result_schema_version',1)
        if version not in {1,RESULT_SCHEMA_VERSION}:raise ValueError('Unsupported result schema version '+str(version))
        if not isinstance(document.get('values'),dict):raise ValueError('Result lacks named parameter values')
        return cls(document)

    @property
    def values(self):return dict(self.document['values'])

    def parameters(self):
        result={}
        for name,row in self.document.get('parameter_definitions',{}).items():
            row=dict(row)
            for key,default in [('lower',-np.inf),('upper',np.inf)]:
                if row.get(key) is None:row[key]=default
            if isinstance(row.get('prior'),dict):
                prior=dict(row['prior'])
                for key,default in [('lower',-np.inf),('upper',np.inf)]:
                    if prior.get(key) is None:prior[key]=default
                row['prior']=prior
            result[name]=Parameter(**row)
        return result

    def write(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(json_value(self.document),indent=2,allow_nan=False)+'\n')


@dataclass(frozen=True)
class SavedPosterior:
    names: tuple
    samples: np.ndarray
    weights: np.ndarray
    derived_names: tuple
    derived: object
    metadata: dict

    @classmethod
    def read(cls,directory):
        directory=Path(directory);metadata=json.loads((directory/'inference.json').read_text())
        version=metadata.get('posterior_schema_version',1)
        if version not in {1,POSTERIOR_SCHEMA_VERSION}:raise ValueError('Unsupported posterior schema version '+str(version))
        names=tuple(metadata['parameter_names']);derived_names=tuple(metadata.get('derived_parameter_names',()))
        raw=np.atleast_2d(np.loadtxt(directory/'posterior.txt'))
        if raw.shape[1]!=2+len(names)+len(derived_names):raise ValueError('Posterior column/schema mismatch')
        weights=raw[:,0]
        if not np.isfinite(raw).all() or np.any(weights<0) or weights.sum()<=0:raise ValueError('Invalid posterior samples/weights')
        return cls(names,raw[:,2:2+len(names)],weights/weights.sum(),derived_names,
                   raw[:,2+len(names):] if derived_names else None,metadata)

    def restore(self,sampler,*,count=2000,seed=0,allow_legacy_replay=False):
        if set(self.names)!=set(sampler.names):raise ValueError('Posterior and model sampled parameters differ')
        if self.derived is None and sampler.linear and not allow_legacy_replay:
            raise ValueError('Legacy conditional moments are absent; enable explicit replay with the matching model/data')
        if self.derived is not None and self.derived_names!=tuple(sampler.derived_names):raise ValueError('Conditional column order differs')
        if self.metadata.get('posterior_schema_version',1)>=2 and self.metadata.get('model_signature')!=sampler.signature():
            raise ValueError('Posterior model/prior/input signature mismatch')
        indices=np.random.default_rng(seed).choice(len(self.weights),size=count,p=self.weights)
        samples=self.samples[indices][:,[self.names.index(n) for n in sampler.names]]
        return sampler.restore_fluxes(samples,derived=None if self.derived is None else self.derived[indices],seed=seed)
