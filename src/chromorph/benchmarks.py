"""Reproducible CPU benchmarks, including simultaneous changes to every shape."""
from pathlib import Path
import argparse,json,platform,resource,time,itertools
import numpy as np
from . import Scene,Observation,EmpiricalPSF,PointSource,Sersic,FreeBandSED,Parameter
from .rendering import GalSimRenderer
from .direct import FullPixelLikelihood
from .likelihood import CompressedGaussianLikelihood


def synthetic_scene(size=64,n_filters=2,n_clumps=1,*,overlap=False,seed=17):
    yy,xx=np.indices((25,25));kernel=np.exp(-((xx-12.2)**2+(yy-11.85)**2)/(2*1.15**2));kernel/=kernel.sum()
    observations=[Observation('band'+str(b),'band'+str(b),np.zeros((size,size)),np.full((size,size),.01),
        EmpiricalPSF(kernel,.04,(12.2,11.85)),.04,1.,pivot_angstrom=3000+1000*b) for b in range(n_filters)]
    scene=Scene(observations,renderer=GalSimRenderer())
    scene.add_component(PointSource('nucleus',FreeBandSED({b:2.+.2*j for j,b in enumerate(scene.bands)}),
        parameters={a:Parameter((size-1)/2,(size-1)/2-4,(size-1)/2+4) for a in ['x','y']}))
    for j in range(n_clumps):
        angle=2*np.pi*j/max(n_clumps,1);radius=2. if overlap else 6.+j
        x=(size-1)/2+radius*np.cos(angle);y=(size-1)/2+radius*np.sin(angle)
        scene.add_component(Sersic('clump'+str(j+1),FreeBandSED({b:3.+j+.25*k for k,b in enumerate(scene.bands)}),
            parameters=dict(x=Parameter(x,x-4,x+4),y=Parameter(y,y-4,y+4),
                re=Parameter(1.3+.5*(j%3),.1,8.),n=Parameter([.7,1.,2.,3.5][j%4],.3,6.2),
                q=Parameter(.65,.1,1.),pa=Parameter(20.*j,-180,180))))
    truth=scene.registry.unpack();images,_,_=scene.predict(truth);rng=np.random.default_rng(seed)
    for o in observations:o.data[:]=images[o.name]+rng.normal(0,o.sigma,o.data.shape)
    scene.clear_caches();scene.metrics['render_calls']=0
    return scene,truth


def proposals(scene,truth,kind,repeats):
    output=[]
    for i in range(repeats):
        v=dict(truth);step=(i+1)*.003
        if kind in {'all_morphology','one_morphology'}:
            for c in (scene.components if kind=='all_morphology' else scene.components[1:2]):
                for key in c.parameters:
                    name=c.name+'.morphology.'+key
                    if key in {'x','y'}:v[name]+=step
                    elif key in {'re','n'}:v[name]*=1+step
                    elif key=='q':v[name]-=step
                    elif key=='pa':v[name]+=step*20
        elif kind=='nuclear_flux':
            for b in scene.bands:v['nucleus.flux.'+b]*=1+step
        elif kind=='extended_band_flux':
            for c in scene.components[1:]:
                for b in scene.bands:v[c.name+'.flux.'+b]*=1+step
        output.append(v)
    return output


def benchmark_case(*,size=64,n_filters=2,n_clumps=1,repeats=3,overlap=False,map_nfev=0):
    started=time.perf_counter();timings={};scores={};setup={}
    for mode in ['full','compressed_rebuild','compressed_incremental']:
        t=time.perf_counter();scene,truth=synthetic_scene(size,n_filters,n_clumps,overlap=overlap)
        setup[mode]=time.perf_counter()-t
        engine=(FullPixelLikelihood(scene) if mode=='full' else CompressedGaussianLikelihood(scene,incremental=mode.endswith('incremental')))
        scene._compressed=None if mode=='full' else engine
        t=time.perf_counter();engine.loglikelihood(truth);cold=time.perf_counter()-t
        timings[mode]=dict(initial_likelihood_seconds=cold,updates={});scores[mode]={}
        for kind in ['all_morphology','one_morphology','nuclear_flux','extended_band_flux']:
            # Restore the reference state before each proposal class, separately
            # accounting for its preparation rather than timing a cache hit.
            t=time.perf_counter();engine.loglikelihood(truth);prepare=time.perf_counter()-t
            values=proposals(scene,truth,kind,repeats);seconds=[];evaluated=[]
            calls=scene.metrics['render_calls'];cput=time.process_time()
            for v in values:
                t=time.perf_counter();evaluated.append(engine.loglikelihood(v));seconds.append(time.perf_counter()-t)
            timings[mode]['updates'][kind]=dict(median_seconds=float(np.median(seconds)),total_seconds=sum(seconds),
                cpu_seconds=time.process_time()-cput,preparation_seconds=prepare,
                render_calls=scene.metrics['render_calls']-calls,likelihood_calls=repeats)
            scores[mode][kind]=evaluated
        timings[mode]['metrics']=dict(engine.metrics)
        timings[mode]['component_cache_bytes']=scene._basis_cache.nbytes
        scene.close()
    errors={}
    for mode in list(scores)[1:]:
        errors[mode]={kind:float(np.max(np.abs(np.array(scores[mode][kind])-scores['full'][kind]))) for kind in scores['full']}
        for kind in scores['full']:
            np.testing.assert_allclose(scores[mode][kind],scores['full'][kind],rtol=1e-12,atol=1e-7)
    map_results={}
    if map_nfev:
        for method in ['trf','L-BFGS-B']:
            scene,truth=synthetic_scene(size,n_filters,n_clumps,overlap=overlap)
            t=time.perf_counter();fit=scene.fit(optimizer=method,multistart=1,max_nfev=map_nfev,estimate_uncertainty=False)
            map_results[method]=dict(seconds=time.perf_counter()-t,chi2=fit.chi2,success=fit.success,
                recovery={n:fit.values[n]-v for n,v in truth.items() if '.morphology.' in n},performance=fit.performance)
            scene.close()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024)
    return dict(stamp=size,filters=n_filters,sersic_components=n_clumps,overlap=overlap,repeats=repeats,
        setup_seconds=setup,paths=timings,maximum_loglikelihood_difference=errors,map=map_results,
        process_peak_rss_mb=rss,peak_memory_scope='process high-water mark',end_to_end_seconds=time.perf_counter()-started,
        production_morphology_path=min(timings,key=lambda m:timings[m]['updates']['all_morphology']['total_seconds']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stamps',type=int,nargs='+',default=[64,128,256])
    parser.add_argument('--filters',type=int,nargs='+',default=[2,4,6])
    parser.add_argument('--clumps',type=int,nargs='+',default=[1,2,4])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--overlap',action='store_true')
    parser.add_argument('--map-nfev',type=int,default=0)
    args=parser.parse_args(argv);args.output.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();results=[]
    report=dict(seed=17,platform=platform.platform(),numpy=np.__version__,
        primary_workload='all morphology parameters in all components change simultaneously',
        warm_cache_speedups_are_not_total_inference_speedups=True,cases=results)
    for size,bands,clumps in itertools.product(args.stamps,args.filters,args.clumps):
        result=benchmark_case(size=size,n_filters=bands,n_clumps=clumps,repeats=args.repeats,overlap=args.overlap,map_nfev=args.map_nfev)
        results.append(result);report['total_seconds']=time.perf_counter()-start
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(stamp=size,filters=bands,clumps=clumps,path=result['production_morphology_path'],seconds=result['end_to_end_seconds'])),flush=True)
    return report


if __name__=='__main__':main()
