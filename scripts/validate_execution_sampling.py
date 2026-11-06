#!/usr/bin/env python3
"""Bounded public PolyChord execution/derived-column validation, not convergence.

Synthetic priors are declared here only; no real-target defaults are changed.
Run with scripts/run_with_env.py, optionally -n 2 for MPI.
"""
from pathlib import Path
import argparse,json,time,resource
import numpy as np
from dataclasses import replace
from chromorph.benchmarks import synthetic_scene
from chromorph.inference import PolyChordPosterior
from chromorph.records import SavedPosterior


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['full','components','aggregate'],default='components')
    p.add_argument('--nlive',type=int,default=32);p.add_argument('--max-dead',type=int,default=100);p.add_argument('--seed',type=int,default=71)
    a=p.parse_args();root=Path(a.output)
    try:
        from mpi4py import MPI
        rank=MPI.COMM_WORLD.rank
    except ImportError:rank=0
    t=time.perf_counter();cpu=time.process_time()
    with synthetic_scene(24,2,1,overlap=True)[0] as scene:
        # Host fluxes have explicit uniform simulation priors. Nuclear fluxes
        # retain the already-selected independent improper flat measure.
        for name in scene.registry.names():
            if '.flux.' in name and not name.startswith('nucleus.'):
                scene.registry.parameters[name]=replace(scene.registry.parameters[name],upper=12.)
        # Predeclared compact synthetic validation domain, not a MAP-derived
        # prior and never applied to a target configuration.
        limits={'re':(.3,3.),'n':(.3,1.3),'q':(.4,1.)}
        for c in scene.components[1:]:
            for key,(lo,hi) in limits.items():
                name=c.name+'.morphology.'+key
                scene.registry.parameters[name]=replace(scene.registry.parameters[name],lower=lo,upper=hi)
        scene.policy_options=dict(rendering='aggregate' if a.mode=='aggregate' else 'components')
        sampler=PolyChordPosterior(scene,likelihood='full' if a.mode=='full' else 'auto')
        sampler.run(root,restore_samples=1000,nlive=a.nlive,max_ndead=a.max_dead,num_repeats=2*len(sampler.names),
                    seed=a.seed,read_resume=False,write_resume=True,feedback=0,precision_criterion=.05)
        if rank==0:
            saved=SavedPosterior.read(root);before=scene.metrics['render_calls']
            names,draws=saved.restore(sampler,count=500,seed=11)
            assert scene.metrics['render_calls']==before
            assert sampler.metrics['restoration_replay_evaluations']==0
            report=dict(synthetic_prior_domain=limits,mode=a.mode,nlive=a.nlive,max_ndead=a.max_dead,seed=a.seed,ndim=len(sampler.names),
                nderived=len(sampler.derived_names),wall_seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu,
                peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                renderer_metrics=dict(scene.renderer.metrics),workspace_metrics=dict(scene.metrics),restoration=dict(sampler.metrics),
                nuclear_nonnegative=bool(np.all(draws[:,-len(sampler.linear):]>=0)),evidence_defined=False,
                convergence_validated=False,note='Stopped at a deliberately small dead-point budget; weight ESS is not a convergence or recovery claim.')
            (root/'execution_validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
