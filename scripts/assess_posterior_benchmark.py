#!/usr/bin/env python3
"""Compare exact-target runs without treating weight ESS as MCMC independence."""
from pathlib import Path
import argparse,json
import numpy as np
from chromorph.sampling_diagnostics import chain_diagnostics
from chromorph.provenance import json_value


def load(path):
    header=json.loads((path/'target.json').read_text());names=header['names']
    if (path/'chain.npz').exists():
        with np.load(path/'chain.npz') as z:samples=z['samples'].copy()
        weights=np.ones(len(samples))/max(1,len(samples));kind='mcmc'
    elif (path/'population.npz').exists():
        with np.load(path/'population.npz') as z:samples=z['samples'].copy();weights=z['weights'].copy()
        kind='population'
    elif (path/'posterior.txt').exists():
        raw=np.atleast_2d(np.loadtxt(path/'posterior.txt'));samples=raw[:,2:2+len(names)];weights=raw[:,0]/raw[:,0].sum();kind='nested'
    else:return header,None,None,None
    return header,samples,weights,kind


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--exchangeable',action='store_true');p.add_argument('--output')
    a=p.parse_args();root=Path(a.directory);methods={};signatures=set()
    for path in sorted(root.glob('*_seed*')):
        if not (path/'target.json').exists():continue
        header,samples,weights,kind=load(path);signatures.add(header['signature']);method=header['method']
        cost=json.loads((path/'cost.json').read_text())
        execution=json.loads((path/'run.json').read_text()) if (path/'run.json').exists() else {}
        completed=cost['exit_code']==0 and not cost['watchdog_timeout'] and execution.get('status')=='complete'
        row=dict(directory=path.name,cost=cost,samples_available=0 if samples is None else len(samples),fixed_count_run_completed=completed)
        if samples is not None and len(samples):
            order=np.argsort(samples,axis=0);quant=[]
            for j in range(samples.shape[1]):
                ids=order[:,j];cdf=np.cumsum(weights[ids]);quant.append(np.interp([.05,.5,.95],cdf,samples[ids,j]).tolist())
            mean=weights@samples;centered=samples-mean
            row.update(names=header['names'],mean=mean.tolist(),variance=(weights@centered**2).tolist(),quantiles=quant,
                importance_weight_ess=float(1/(weights@weights)),kind=kind)
            if a.exchangeable:
                i=header['names'].index('clump1.morphology.x');j=header['names'].index('clump2.morphology.x')
                mode=samples[:,i]<samples[:,j];row['left_mode_probability']=float(weights@mode)
                row['mode_transitions']=int(np.sum(mode[1:]!=mode[:-1])) if kind=='mcmc' else None
            row['_samples']=samples
        methods.setdefault(method,[]).append(row)
    if len(signatures)>1:raise ValueError('Runs do not have identical scientific targets/priors/data')
    results={}
    for method,rows in methods.items():
        complete=[r for r in rows if r.get('samples_available',0)>3 and r['fixed_count_run_completed']]
        record=dict(runs=rows,total_cpu_hours=sum(r['cost']['cpu_hours'] for r in rows),
                    total_wall_seconds=sum(r['cost']['wall_seconds'] for r in rows),diagnostics=None,reliable_ess_per_cpu_hour=None)
        if complete and complete[0]['kind']=='mcmc' and len(complete)>=2:
            n=min(len(r['_samples']) for r in complete);chains=np.asarray([r['_samples'][:n] for r in complete])
            mode_ok=bool(a.exchangeable and all(.4<r['left_mode_probability']<.6 and r['mode_transitions']>=20 for r in complete))
            diag=chain_diagnostics(chains,cpu_hours=record['total_cpu_hours'],mode_validation_passed=mode_ok)
            record['diagnostics']=diag;record['reliable_ess_per_cpu_hour']=diag['reliable_ess_per_cpu_hour']
        if len(complete)>=4:
            means=np.array([r['mean'] for r in complete]);variances=np.array([r['variance'] for r in complete])
            between=np.var(means,axis=0,ddof=1);posterior_var=np.mean(variances,axis=0)
            estimate=np.divide(posterior_var,between,out=np.full_like(between,np.nan),where=between>0)
            from scipy.stats import chi2
            df=len(complete)-1
            interval=np.stack([estimate*chi2.ppf(.025,df)/df,estimate*chi2.ppf(.975,df)/df],axis=-1)
            record['replicate_mean_ess_estimate']=dict(minimum=float(np.nanmin(estimate)),parameters=estimate.tolist(),interval_95=interval.tolist(),
                minimum_per_mean_run_cpu_hour=float(np.nanmin(estimate))/(record['total_cpu_hours']/len(rows)),
                replicates=len(complete),validated=False,
                note='Between independent-run means, assuming negligible bias. Very uncertain with four replicates; not a production-sampler selection metric.')
        for r in rows:r.pop('_samples',None)
        results[method]=record
    report=dict(model_signatures=list(signatures),methods=results,production_sampler_selected=None,
        note='Known synthetic label-mode symmetry is tested only with --exchangeable. Importance-weight ESS is separate from autocorrelation ESS. Full J1042 mode coverage remains unvalidated.')
    output=Path(a.output) if a.output else root/'assessment.json';output.write_text(json.dumps(json_value(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({m:dict(cpu_hours=x['total_cpu_hours'],reliable_ess_per_cpu_hour=x['reliable_ess_per_cpu_hour']) for m,x in results.items()}))


if __name__=='__main__':main()
