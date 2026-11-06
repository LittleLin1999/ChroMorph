"""Versioned scientific signatures and JSON conversion, outside hot loops."""
from dataclasses import asdict,is_dataclass
import hashlib,json
import numpy as np

MODEL_SCHEMA_VERSION=2
CONFIG_SCHEMA_VERSION=2
RESULT_SCHEMA_VERSION=2
POSTERIOR_SCHEMA_VERSION=2


def json_value(value):
    if is_dataclass(value):return json_value(asdict(value))
    if isinstance(value,np.ndarray):return json_value(value.tolist())
    if isinstance(value,np.generic):return json_value(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_value(v) for v in value]
    return value


def content_signature(value):
    return hashlib.sha256(json.dumps(json_value(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def array_signature(array):
    array=np.asarray(array);digest=hashlib.sha256()
    digest.update(str(array.shape).encode());digest.update(array.dtype.str.encode())
    digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return digest.hexdigest()


def observation_signature(observation):
    """Cache stable content signatures only for frozen inputs, never hash per proposal."""
    o=observation
    runtime=(o.likelihood_token(),o.psf.cache_token(),tuple(o.matrix.ravel()),tuple(o.offset),
        o.pixel_scale,o.ujy_per_rate,getattr(o.throughput,'fingerprint',None))
    previous=getattr(o,'_content_signature',None)
    immutable=not any(a.flags.writeable for a in [o.data,o.sigma,o.mask])
    if immutable and previous is not None and previous[0]==runtime:return previous[1]
    covariance=o.covariance
    if covariance is None:noise=None
    elif isinstance(covariance,np.ndarray):noise=array_signature(covariance)
    elif hasattr(covariance,'matrix'):noise=array_signature(covariance.matrix)
    elif getattr(covariance,'signature',None) is not None:noise=covariance.signature
    elif hasattr(getattr(covariance,'operator',None),'A'):
        noise=dict(operator=array_signature(covariance.operator.A),logdet=covariance.logdet_covariance,
                   mask=covariance.mask_indices)
    else:raise ValueError('A custom fixed covariance/whitening operator needs a stable signature for persisted results or checkpoints')
    payload=dict(name=o.name,band=o.band,data=array_signature(o.data),sigma=array_signature(o.sigma),
        mask=array_signature(o.mask),covariance=noise,psf=dict(image=array_signature(o.psf.image),
            center=o.psf.center,scale=o.psf.pixel_scale,pixel_integrated=o.psf.pixel_integrated),
        matrix=o.matrix,offset=o.offset,pixel_scale=o.pixel_scale,ujy_per_rate=o.ujy_per_rate,
        throughput=getattr(o.throughput,'fingerprint',None))
    result=content_signature(payload)
    if immutable:o._content_signature=(runtime,result)
    return result


def dependency_versions():
    from importlib.metadata import version,PackageNotFoundError
    result={}
    for package in ['numpy','scipy','astropy','GalSim','fsps']:
        try:result[package]=version(package)
        except PackageNotFoundError:result[package]=None
    return result


def fsps_reference_signature():
    """Synthesis binary/data identity; called for persistence, never per proposal."""
    from importlib.util import find_spec
    from pathlib import Path
    from functools import lru_cache
    import os,subprocess
    spec=find_spec('fsps')
    if spec is None:return dict(available=False)
    root=os.environ.get('SPS_HOME');revision=None
    if root:
        q=subprocess.run(['git','-C',root,'rev-parse','HEAD'],capture_output=True,text=True)
        if q.returncode==0:revision=q.stdout.strip()
    extensions=list(Path(spec.origin).parent.glob('_fsps*.so'))
    binary=None if not extensions else hashlib.sha256(extensions[0].read_bytes()).hexdigest()
    return dict(available=True,extension_sha256=binary,data_git_revision=revision)


def model_signature(scene):
    evaluator=scene.prepare();evaluator._ensure_renderer()
    for o in scene.observations:evaluator.gaussian_data(o)
    parameters=[]
    for name,p in scene.registry.parameters.items():
        spec=asdict(p);spec.pop('value')
        if not p.vary:spec['fixed_value']=p.value
        parameters.append((name,spec))
    components=[]
    from .spatial import PointSource
    for c in scene.components:
        description=c.sed.describe() if hasattr(c.sed,'describe') else dict(model=type(c.sed).__name__)
        description={k:v for k,v in description.items() if k not in {'parameters','initial_fluxes','cache_size','cache_bytes'}}
        components.append(dict(name=c.name,spatial='point' if isinstance(c,PointSource) else 'sersic',
            nuclear=getattr(c,'nuclear',False),sed=description))
    renderer=evaluator.renderer
    rendering=dict(configuration=getattr(renderer,'configuration',{}),
        gsparams=repr(renderer.gsparams) if hasattr(renderer,'gsparams') else None,
        implementation='galsim' if hasattr(renderer,'gsparams') else type(renderer).__name__)
    payload=dict(model_schema_version=MODEL_SCHEMA_VERSION,dependencies=dependency_versions(),
        fsps_reference=fsps_reference_signature() if any(isinstance(c.sed,__import__('chromorph.sed',fromlist=['FSPSStellarSED']).FSPSStellarSED) for c in scene.components) else None,parameter_order=parameters,ties=dict(scene.registry.ties),
        context=asdict(scene.context),components=components,renderer=rendering,
        observations=[observation_signature(o) for o in scene.observations],
        density_parameterization='registered coordinates; log_mass is log10 formed mass')
    return content_signature(payload)
