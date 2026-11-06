#!/usr/bin/env python3
"""Conditional one-dimensional kernel/coverage audit against numerical quadrature.

This declared synthetic regression model has known geometry. It validates
posterior mathematics and the transition kernel, not full-morphology speed or
J1042 coverage. Real and multicomponent benchmarks retain their original priors.
"""
from pathlib import Path
import argparse,json,time,resource
import numpy as np
from scipy.integrate import cumulative_trapezoid,trapezoid
from chromorph import Scene,PointSource,Sersic,FreeBandSED,Parameter,Observation,EmpiricalPSF
from chromorph.posterior import ExactPosteriorTarget
from chromorph.blocked import BlockedMetropolis
from chromorph.sampling_diagnostics import chain_diagnostics,coverage_report
from posterior_benchmark_cases import IntegratedGaussianRenderer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--datasets',type=int,default=12)
    p.add_argument('--draws',type=int,default=4000);p.add_argument('--warmup',type=int,default=1000);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);rows=[];intervals=[];truths=[];started=time.perf_counter();cpu=time.process_time()
    for data_seed in range(a.datasets):
        yy,xx=np.indices((15,15));kernel=np.exp(-((xx-7)**2+(yy-7)**2)/2);kernel/=kernel.sum()
        obs=[Observation(b,b,np.zeros((15,15)),np.full((15,15),.04),EmpiricalPSF(kernel,.04),.04,1.) for b in ['blue','red']]
        with Scene(obs,renderer=IntegratedGaussianRenderer()) as scene:
            scene.add_component(PointSource('nucleus',FreeBandSED({'blue':1.5,'red':2.}),parameters={k:Parameter(7.,vary=False) for k in ['x','y']}))
            scene.add_component(Sersic('host',FreeBandSED(parameters={'blue':Parameter(1.2,0.,3.),'red':Parameter(1.,0.,np.inf,vary=False)}),parameters={
                'x':Parameter(7.4,vary=False),'y':Parameter(7.,vary=False),'re':Parameter(1.5,.1,4.,vary=False),
                'n':Parameter(.5,.3,6.2,vary=False),'q':Parameter(1.,.1,1.,vary=False),'pa':Parameter(0.,vary=False)}))
            v=scene.registry.unpack();images=scene.predict(v)[0];rng=np.random.default_rng(700+data_seed)
            for o in obs:o.update(data=images[o.name]+rng.normal(0,o.sigma,o.data.shape))
            target=ExactPosteriorTarget(scene,likelihood='full');grid=np.linspace(0,3,1501)
            scores=np.array([target.logpdf([x]) for x in grid]);density=np.exp(scores-scores.max())
            cdf=cumulative_trapezoid(density,grid,initial=0);cdf/=cdf[-1]
            exact_interval=np.interp([.05,.95],cdf,grid);exact_mean=float(trapezoid(grid*density,grid)/trapezoid(density,grid))
            chains=[];errors=[];runs=[]
            for seed,start in enumerate([.1,.8,2.,2.9]):
                result=BlockedMetropolis(target).run([start],draws=a.draws,warmup=a.warmup,seed=100+seed)
                chains.append(result.samples);runs.append(result.metadata)
                errors.append(float(np.mean(result.samples[:,0])-exact_mean))
            diag=chain_diagnostics(np.asarray(chains),mode_validation_passed=True)
            combined=np.concatenate(chains)[:,0];ci=np.quantile(combined,[.05,.95]);intervals.append([ci]);truths.append([1.2])
            rows.append(dict(data_seed=data_seed,diagnostics=diag,mean_errors=errors,exact_mean=exact_mean,
                exact_interval=exact_interval.tolist(),mcmc_interval=ci.tolist(),
                maximum_interval_error=float(np.max(abs(ci-exact_interval))),runs=runs))
            np.savez_compressed(root/('dataset'+str(data_seed)+'.npz'),chains=np.asarray(chains),grid=grid,cdf=cdf)
            (root/'validation.json').write_text(json.dumps(dict(rows=rows,coverage=coverage_report(np.asarray(intervals),np.asarray(truths)),
                wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                scope='Known-geometry one-dimensional exact kernel audit; not full-morphology or J1042 coverage'),indent=2)+'\n')
    print('Completed conditional kernel audit:',len(rows),'datasets',flush=True)


if __name__=='__main__':main()
