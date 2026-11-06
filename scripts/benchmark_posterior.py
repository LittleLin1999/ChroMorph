#!/usr/bin/env python3
"""Bounded posterior pilots with complete cost accounting and no sampler selection.

Each child owns one scene and its FSPS state. An outer watchdog includes cold
initialization, training, shutdown, peak process-family RSS and CPU cost. Use
explicit configs/results for private workloads. No morphology prior is changed.
"""
from pathlib import Path
import argparse,json,os,sys,time,resource,subprocess,signal
import numpy as np


def child(args):
    from chromorph.posterior import ExactPosteriorTarget
    from chromorph.provenance import json_value
    if args.config:
        from chromorph.config import load_config,build_scene
        from chromorph.cli import observations_for
        config=load_config(args.config);obs,qa,ctx=observations_for(config)
        scene=build_scene(config,obs,context=ctx);truth=None
        initial_values=json.loads(Path(args.saved_result).read_text())['values']
    else:
        from posterior_benchmark_cases import synthetic_case
        scene,truth=synthetic_case(args.data_seed,overlap=not args.separated,full_morphology=args.full_morphology)
        initial_values=truth
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    target=ExactPosteriorTarget(scene,likelihood='full' if args.method=='full_blocked' else 'auto')
    signature=target.signature()
    (out/'target.json').write_text(json.dumps(dict(signature=signature,names=target.names,linear=target.linear,
        lower=target.lower.tolist(),upper=target.upper.tolist(),truth=truth,config=args.config,method=args.method),default=str,indent=2)+'\n')
    initial=np.array([initial_values[n] for n in target.names]);rng=np.random.default_rng(args.seed)
    if args.prior_initial or (args.config and args.seed%2):initial=target.prior(rng.uniform(size=len(initial)))
    elif args.seed%2:
        # Independent label-permuted starting modes. Priors/bounds never change.
        from chromorph.blocked import component_blocks
        exchanges=[b for b in component_blocks(target) if b.exchange]
        if exchanges:
            b=exchanges[0];old=initial.copy();initial[list(b.indices)]=old[list(b.exchange)];initial[list(b.exchange)]=old[list(b.indices)]
            if not np.isfinite(target.logprior(initial)):initial=np.array([initial_values[n] for n in target.names])
    np.save(out/'initial.npy',initial)
    start=time.perf_counter();cpu=time.process_time();status='complete';error=None
    try:
        if args.method in ['blocked','full_blocked']:
            from chromorph.blocked import BlockedMetropolis
            result=BlockedMetropolis(target,joint_component_weight=args.joint_component_weight).run(initial,draws=args.draws,warmup=args.warmup,seed=args.seed,
                max_wall_seconds=args.seconds*.9,transactional=args.method=='blocked')
            result.save(out)
            status='complete' if len(result.samples)==args.draws else 'budget_exhausted'
        elif args.method=='pocomc':
            from chromorph.population import PocoMCPosterior
            PocoMCPosterior(target,max_wall_seconds=args.seconds*.9).run(out,n_total=args.population_draws,
                n_effective=args.particles,n_active=args.particles//2,seed=args.seed,n_prior=args.particles*2,
                flow='nsf3',train_config=dict(epochs=50,patience=10),n_steps=2,n_max_steps=20)
        elif args.method=='polychord':
            from chromorph.inference import PolyChordPosterior
            PolyChordPosterior(scene,likelihood=target.likelihood).run(out,nlive=args.particles,num_repeats=2*len(target.names),
                max_ndead=args.max_dead,precision_criterion=.05,seed=args.seed,read_resume=False,feedback=0,restore_samples=0)
    except Exception as exc:
        status='budget_exhausted' if 'budget' in str(exc).lower() else 'failed';error=repr(exc)
    finally:
        metrics=dict(scene.prepare().metrics)
        spectral={n:dict(b.metrics) for n,b in getattr(scene.prepare()._spectral_engine,'backends',{}).items()}
        renderer=dict(getattr(scene.renderer,'metrics',{}));scene.close()
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    report=dict(status=status,error=error,method=args.method,seed=args.seed,data_seed=args.data_seed,model_signature=signature,
        wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu+usage.ru_utime+usage.ru_stime,
        peak_parent_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_child_rss_native=usage.ru_maxrss,
        spatial=metrics,spectral=spectral,renderer=renderer,convergence_validated=False,
        note='Timing includes target execution and shutdown; outer cost.json additionally includes imports/preparation/training.')
    (out/'run.json').write_text(json.dumps(json_value(report),indent=2,allow_nan=False)+'\n')
    if error:print(error,file=sys.stderr)


def launch(command,directory,seconds):
    import psutil
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN);peak=0;cpu_by_pid={};timed_out=False
    with (directory/'stdout.log').open('w') as stdout,(directory/'stderr.log').open('w') as stderr:
        process=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
        root=psutil.Process(process.pid)
        while process.poll() is None:
            try:family=[root]+root.children(recursive=True)
            except psutil.Error:family=[]
            total=0
            for member in family:
                try:
                    total+=member.memory_info().rss;t=member.cpu_times();cpu_by_pid[member.pid]=t.user+t.system
                except psutil.Error:pass
            peak=max(peak,total)
            if time.perf_counter()-start>seconds:
                timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(.1)
        code=process.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)
    if timed_out:cpu=max(cpu,sum(cpu_by_pid.values()))
    report=dict(command=command,wall_seconds=time.perf_counter()-start,cpu_seconds=cpu,cpu_hours=cpu/3600,
        peak_family_rss_bytes=peak,exit_code=code,watchdog_timeout=timed_out,
        sampled_family_cpu_seconds=sum(cpu_by_pid.values()),cpu_accounting='Complete: waited child/descendants. Watchdog timeout: max(waited, sampled family); sampled aborted costs are lower-bound estimates.',
        reliable_ess_per_cpu_hour=None)
    (directory/'cost.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--child',action='store_true')
    p.add_argument('--method',choices=['blocked','full_blocked','pocomc','polychord'],default='blocked')
    p.add_argument('--methods',nargs='+',default=['blocked','full_blocked','pocomc','polychord'])
    p.add_argument('--seeds',nargs='+',type=int,default=[20,21,22,23]);p.add_argument('--seed',type=int,default=20)
    p.add_argument('--data-seed',type=int,default=101);p.add_argument('--draws',type=int,default=2000);p.add_argument('--warmup',type=int,default=1000)
    p.add_argument('--seconds',type=float,default=180);p.add_argument('--joint-component-weight',type=float,default=0.);p.add_argument('--population-draws',type=int,default=512);p.add_argument('--particles',type=int,default=64);p.add_argument('--max-dead',type=int,default=300)
    p.add_argument('--config');p.add_argument('--saved-result');p.add_argument('--prior-initial',action='store_true')
    p.add_argument('--separated',action='store_true');p.add_argument('--full-morphology',action='store_true')
    args=p.parse_args()
    if args.child:child(args);return
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);reports=[]
    for method in args.methods:
        for seed in args.seeds:
            output=root/(method+'_seed'+str(seed));command=[sys.executable,__file__,'--child','--method',method,'--seed',str(seed),
                '--output',str(output),'--draws',str(args.draws),'--warmup',str(args.warmup),'--seconds',str(args.seconds),
                '--joint-component-weight',str(args.joint_component_weight),'--particles',str(args.particles),'--population-draws',str(args.population_draws),'--max-dead',str(args.max_dead),'--data-seed',str(args.data_seed)]
            if args.config:command+=['--config',args.config,'--saved-result',args.saved_result]
            for option,flag in [('prior_initial',args.prior_initial),('separated',args.separated),('full_morphology',args.full_morphology)]:
                if flag:command+=['--'+option.replace('_','-')]
            report=launch(command,output,args.seconds);reports.append(report)
            (root/'costs.json').write_text(json.dumps(reports,indent=2)+'\n')
            print(method,seed,round(report['wall_seconds'],2),report['exit_code'],flush=True)


if __name__=='__main__':main()
