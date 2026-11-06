#!/usr/bin/env python3
"""Bounded CPU execution benchmark; each strategy uses a fresh process.

Run: python scripts/benchmark_cpu_execution.py --output REPORT.json --repeats 4
Each strategy includes process startup/import, input preparation and cold calls.
Warm proposal timings are reported separately, never called end-to-end speedup.
"""
import argparse,json,os,platform,resource,subprocess,sys,time
from pathlib import Path


def child(case,mode,repeats,fit_workers):
    started=time.perf_counter();cpu=time.process_time()
    import numpy as np
    from chromorph.benchmarks import synthetic_scene,proposals
    from chromorph.direct import FullPixelLikelihood
    from chromorph.policy import ExecutionPolicy
    from chromorph.likelihood import CompressedGaussianLikelihood
    size,filters,clumps=case
    scene,truth=synthetic_scene(size,filters,clumps,overlap=True)
    if fit_workers:
        for c in scene.components[1:]:
            for name in ['n','q','pa']:
                from dataclasses import replace
                key=c.name+'.morphology.'+name;scene.registry.parameters[key]=replace(scene.registry.parameters[key],vary=False)
        result=scene.fit(max_nfev=3,max_wall_seconds=40,workers=fit_workers,multistart=1,estimate_uncertainty=False)
        report=dict(fit=result.performance,objective=result.runs[0]['objective'],success=result.runs[0]['success'])
    else:
        engine=(FullPixelLikelihood(scene) if mode=='full' else
                CompressedGaussianLikelihood(scene) if mode=='compressed' else
                ExecutionPolicy(scene,rendering='aggregate' if mode=='aggregate' else 'components'))
        prep=time.perf_counter();initial=engine.loglikelihood(truth);cold=time.perf_counter()-prep
        report=dict(initial_score=initial,cold_likelihood_seconds=cold,updates={})
        mixed=[]
        for kind in ['all_morphology','one_morphology','nuclear_flux','extended_band_flux','nuclear_centroid','repeated_n']:
            values=proposals(scene,truth,kind if kind not in {'nuclear_centroid','repeated_n'} else 'all_morphology',repeats)
            if kind=='nuclear_centroid':
                values=[dict(truth,**{'nucleus.morphology.x':truth['nucleus.morphology.x']+.013*(i+1)}) for i in range(repeats)]
            if kind=='repeated_n':
                for v in values:
                    for c in scene.components[1:]:v[c.name+'.morphology.n']=truth[c.name+'.morphology.n']
            mixed.extend(values)
            t=time.perf_counter();engine.loglikelihood(truth);preparation=time.perf_counter()-t
            times=[];scores=[];before=dict(scene.renderer.metrics);ct=time.process_time()
            for v in values:
                t=time.perf_counter();scores.append(engine.loglikelihood(v));times.append(time.perf_counter()-t)
            report['updates'][kind]=dict(seconds=sum(times),median_seconds=float(np.median(times)),scores=scores,
                cpu_seconds=time.process_time()-ct,preparation_seconds=preparation,
                renderer_calls={k:scene.renderer.metrics[k]-before[k] for k in before})
        # Identical mixed trace includes revisits/rejected states and all-shape changes.
        mixed=mixed[::2]+list(reversed(mixed[1::2]))+[truth]
        t=time.perf_counter();scores=[engine.loglikelihood(v) for v in mixed]
        report['mixed']=dict(seconds=time.perf_counter()-t,calls=len(mixed),scores=scores)
        report['engine_metrics']=dict(engine.metrics)
    scene.close()
    report.update(case=case,mode=mode,fit_workers=fit_workers,wall_seconds=time.perf_counter()-started,
        cpu_seconds=time.process_time()-cpu,children_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        renderer_metrics=dict(scene.renderer.metrics),threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--child',choices=['full','components','aggregate','compressed']);p.add_argument('--case',default='64,2,1');p.add_argument('--fit-workers',type=int,default=0)
    p.add_argument('--optimization',action='store_true');args=p.parse_args()
    if args.child:
        print(json.dumps(child(tuple(map(int,args.case.split(','))),args.child,args.repeats,args.fit_workers)));return
    from chromorph.provenance import dependency_versions
    rows=[];started=time.perf_counter()
    for case in [(64,2,1),(64,4,4),(128,2,4)]:
        matching=[]
        for mode in ['full','components','aggregate','compressed']:
            t=time.perf_counter();q=subprocess.run([sys.executable,__file__,'--child',mode,'--case',','.join(map(str,case)),'--repeats',str(args.repeats)],capture_output=True,text=True,check=True)
            row=json.loads(q.stdout);row['process_wall_seconds']=time.perf_counter()-t;rows.append(row);matching.append(row)
        reference=matching[0]
        for row in matching[1:]:
            errors=[abs(a-b) for kind in reference['updates'] for a,b in zip(reference['updates'][kind]['scores'],row['updates'][kind]['scores'])]
            errors.extend(abs(a-b) for a,b in zip(reference['mixed']['scores'],row['mixed']['scores']))
            row['maximum_absolute_delta_loglikelihood']=max(errors)
            if max(errors)>(.01 if row['mode']=='aggregate' else 1e-7):raise RuntimeError('Accuracy gate failed: '+str(row))
    if args.optimization:
        for workers in [1,2,4]:
            t=time.perf_counter();q=subprocess.run([sys.executable,__file__,'--child','components','--case','64,2,1','--fit-workers',str(workers)],capture_output=True,text=True,check=True)
            row=json.loads(q.stdout);row['process_wall_seconds']=time.perf_counter()-t;rows.append(row)
    report=dict(schema_version=2,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),versions=dependency_versions(),
        hardware=dict(platform=platform.platform(),affinity=list(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,cpu_count=os.cpu_count()),
        peak_rss_units='KiB on Linux; bytes on macOS',end_to_end_wall_seconds=time.perf_counter()-started,
        cases=rows,limitations=['Bounded optimization is not convergence validation.','No posterior ESS or evidence claim follows from likelihood throughput.','No physical SED proposals in this free-band renderer workload.'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
