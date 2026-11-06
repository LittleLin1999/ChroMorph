#!/usr/bin/env python3
"""Bounded direct-composite FSPS table validation on an explicit input manifest.

A deliberately small 3^3 pilot covers age, metallicity and attenuation bounds.
It is allowed to fail; failed tables remain reference-fallback only. No global
refinement campaign or production interpolation is launched by this command.
"""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from chromorph.config import load_config,build_scene
from chromorph.cli import observations_for
from chromorph.sed import FSPSStellarSED
from chromorph.templates import BandFluxLibrary,image_loglikelihood_discrepancy
from chromorph.provenance import observation_signature,json_value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--result',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    config=load_config(a.config);obs,qa,ctx=observations_for(config);started=time.perf_counter();cpu=time.process_time()
    with build_scene(config,obs,context=ctx) as scene:
        values=json.loads(Path(a.result).read_text())['values']
        component=next(c for c in scene.components if isinstance(c.sed,FSPSStellarSED));name=component.name
        # Prepare process-local reference states, leaving declarative SEDs inert.
        scene.component_fluxes(values);backend=scene.prepare()._spectral_engine.backends[name]
        axes={}
        for key in ['age','metallicity','av']:
            par=scene.registry.parameters[name+'.sed.'+key]
            if par.vary:
                axes['log_age' if key=='age' else key]=np.linspace(np.log10(par.lower) if key=='age' else par.lower,
                    np.log10(par.upper) if key=='age' else par.upper,3)
        if set(axes)!={'log_age','metallicity','av'}:raise ValueError('This bounded driver requires free age, metallicity and av')
        library=BandFluxLibrary.build(backend,values,name,obs,ctx,axes)
        # Held-out locations, never grid nodes or used to adapt the pilot grid.
        points=[]
        for fractions in [(.02,.13,.07),(.23,.41,.37),(.68,.77,.81),(.97,.96,.93)]:
            point={}
            for key,f in zip(axes,fractions):
                x=axes[key][0]+f*(axes[key][-1]-axes[key][0]);point['age' if key=='log_age' else key]=10**x if key=='log_age' else x
            points.append(point)
        for o in obs:o.ensure_frozen()
        bounds={k[len(name)+1:]:[p.lower,p.upper] for k,p in scene.registry.parameters.items() if k.startswith(name+'.')}
        domain=dict(observation_signatures=[observation_signature(o) for o in obs],parameter_bounds=bounds,
                    scene_parameter_bounds={k:[p.lower,p.upper] if p.vary else [p.value,p.value] for k,p in scene.registry.parameters.items()},
                    scope='Finite pilot: compact and saved morphology, three formed masses; not a proof over these bounds')
        image_rows=[]
        def discrepancy(predicted,truth,shape):
            trial=dict(values)
            for key,x in shape.items():trial[name+'.sed.'+key]=x
            candidates=[]
            for log_mass in [6.,8.,9.]:
                for radius in [values[name+'.morphology.re'],.1]:
                    trial[name+'.sed.log_mass']=log_mass;trial[name+'.morphology.re']=radius
                    error=image_loglikelihood_discrepancy(scene,trial,name,predicted,truth,library.metadata['bands'])
                    candidates.append(error);image_rows.append(dict(shape=shape,log_mass=log_mass,re=radius,delta_loglikelihood=error))
            return max(candidates,key=abs)
        report=library.validate(backend,values,name,obs,ctx,points,loglikelihood_error=discrepancy,validation_domain=domain,
                                relative_tolerance=.001,delta_loglikelihood_tolerance=.005)
        library.save(out/'pilot_library.npz')
        report.update(wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                      peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      axes={k:v.tolist() for k,v in axes.items()},component=name,image_checks=image_rows,
                      reference_metrics=dict(backend.metrics),fingerprint=library.fingerprint,
                      deployment='reference_fallback' if not report['production_ready'] else 'finite_domain_gate_only')
        (out/'validation.json').write_text(json.dumps(json_value(report),indent=2,allow_nan=False)+'\n')
    report['including_shutdown_wall_seconds']=time.perf_counter()-started
    report['children_cpu_seconds']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    report['total_cpu_hours']=(time.process_time()-cpu+report['children_cpu_seconds'])/3600
    (out/'validation.json').write_text(json.dumps(json_value(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['production_ready','maximum_relative_error','maximum_absolute_delta_loglikelihood','including_shutdown_wall_seconds','total_cpu_hours']}))


if __name__=='__main__':main()
