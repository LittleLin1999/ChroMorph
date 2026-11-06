"""Independent-chain diagnostics; efficiency is gated by mixing and mode checks."""
import numpy as np


def chain_diagnostics(chains,*,cpu_hours=None,mode_validation_passed=False,min_ess=400,max_rhat=1.01):
    import arviz as az
    chains=np.asarray(chains,dtype=float)
    if chains.ndim!=3 or chains.shape[0]<2 or chains.shape[1]<4:raise ValueError('Need independent chains x draws x parameters')
    rows=[]
    for j in range(chains.shape[-1]):
        data=chains[:,:,j]
        rows.append(dict(rhat=float(az.rhat(data,method='rank')),bulk_ess=float(az.ess(data,method='bulk')),
            tail_ess=float(az.ess(data,method='tail')),mcse_mean=float(az.mcse(data,method='mean'))))
    minimum=min(min(x['bulk_ess'],x['tail_ess']) for x in rows);worst=max(x['rhat'] for x in rows)
    finite=all(np.isfinite([r['rhat'],r['bulk_ess'],r['tail_ess']]).all() for r in rows)
    mixed=bool(finite and chains.shape[0]>=4 and np.isfinite(worst) and worst<=max_rhat and minimum>=min_ess)
    reliable=bool(mixed and mode_validation_passed)
    return dict(parameters=rows,chains=chains.shape[0],draws=chains.shape[1],minimum_ess=minimum,maximum_rhat=worst,
        mixing_gate_passed=mixed,mode_validation_passed=bool(mode_validation_passed),
        reliable_ess_per_cpu_hour=(minimum/cpu_hours if reliable and cpu_hours and cpu_hours>0 else None),
        note='Convergence diagnostics cannot prove that an undiscovered mode is absent.')


def coverage_report(intervals,truth,probability=.9):
    from scipy.stats import beta
    intervals=np.asarray(intervals);truth=np.asarray(truth)
    if intervals.shape!=truth.shape+(2,):raise ValueError('Expected truth.shape + lower/upper interval axis')
    hit=(intervals[...,0]<=truth)&(truth<=intervals[...,1]);n=hit.shape[0];counts=hit.sum(axis=0)
    lo=np.where(counts==0,0,beta.ppf(.025,counts,n-counts+1));hi=np.where(counts==n,1,beta.ppf(.975,counts+1,n-counts))
    return dict(n_datasets=n,nominal_probability=probability,covered=counts.tolist(),fraction=(counts/n).tolist(),
        binomial_95_interval=np.stack([lo,hi],axis=-1).tolist(),
        note='Finite injection counts have binomial uncertainty; in-sample recovery is not posterior coverage.')
