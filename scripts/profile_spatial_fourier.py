#!/usr/bin/env python3
"""Profile lazy GalSim work and test a gated, reusable-PSF Fourier grid.

The NumPy inverse-FFT experiment is a diagnostic only. It is never an enabled
renderer. Finite padding and Nyquist truncation must pass image-level gates.
"""
import argparse,cProfile,pstats,io,json,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--saved-result');a=p.parse_args()
    from chromorph.rendering import GalSimRenderer
    from chromorph.spatial import Sersic
    if a.config:
        from chromorph.config import load_config,build_scene
        from chromorph.io import load_observations
        config=load_config(a.config);manifest=config.get('observations') or str(Path(config['data_root'])/'observations.yaml')
        obs,_,_=load_observations(manifest,throughputs=False);scene=build_scene(config,obs)
        values=json.loads(Path(a.saved_result).read_text())['values']
    else:
        from chromorph.benchmarks import synthetic_scene
        scene,values=synthetic_scene(64,2,2)
    r=GalSimRenderer();o=scene.observations[0];psf=r._psf(o);rows=[];kcache={};profiler=cProfile.Profile()
    for c in [c for c in scene.components if isinstance(c,Sersic)]:
        morph=scene.morphology(c,values)
        # Fresh, continuous n; no rounding/quantization. Preserve allowed range.
        morph['n']+=1e-5 if morph['n']+1e-5<=6.2 else -1e-5
        t=time.perf_counter();intrinsic=r._intrinsic(morph,o);construction=time.perf_counter()-t
        t=time.perf_counter();intrinsic.kValue(r.gs.PositionD(.73,.43));first_k=time.perf_counter()-t
        t=time.perf_counter();intrinsic.kValue(r.gs.PositionD(.73,.43));repeat_k=time.perf_counter()-t
        profiler.enable();t=time.perf_counter();reference=r.render(c,morph,o);native=time.perf_counter()-t;profiler.disable()
        row=dict(component=c.name,morphology=morph,intrinsic_construction_seconds=construction,first_kvalue_seconds=first_k,
                 repeated_kvalue_seconds=repeat_k,native_draw_after_first_kvalue_seconds=native,grids=[])
        for ngrid in [256,512]:
            dk=2*np.pi/ngrid
            if ngrid not in kcache:
                t=time.perf_counter();kernel=psf.drawKImage(nx=ngrid,ny=ngrid,scale=dk).array.copy()
                kcache[ngrid]=(kernel,time.perf_counter()-t)
            kernel,preparation=kcache[ngrid]
            offset=r._position(morph,o,(0,0));offset-=np.array([.5 if o.data.shape[1]%2==0 else 0,.5 if o.data.shape[0]%2==0 else 0])
            t=time.perf_counter();kprofile=intrinsic.shift(*offset).drawKImage(nx=ngrid,ny=ngrid,scale=dk).array;kt=time.perf_counter()-t
            t=time.perf_counter();image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kprofile*kernel))).real;fft=time.perf_counter()-t
            ny,nx=o.data.shape;y=ngrid//2-ny//2;x=ngrid//2-nx//2;cut=image[y:y+ny,x:x+nx]
            amplitude=values.get(c.name+'.flux.'+o.band,1.)/o.ujy_per_rate
            delta=amplitude*(cut-reference);residual=(o.data-amplitude*reference)/o.sigma
            dll=float(np.sum(residual[o.mask]*delta[o.mask]/o.sigma[o.mask])-.5*np.sum((delta[o.mask]/o.sigma[o.mask])**2))
            maximum=float(np.max(abs(delta[o.mask])/o.sigma[o.mask]))
            row['grids'].append(dict(size=ngrid,psf_preparation_seconds=preparation,profile_kgrid_seconds=kt,inverse_fft_seconds=fft,
                maximum_pixel_error_sigma=maximum,delta_loglikelihood=dll,absolute_flux_sum_error=float(abs(cut.sum()-reference.sum())),
                finite_state_gate_passed=bool(maximum<.01 and abs(dll)<.01)))
        rows.append(row)
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(35)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'cprofile.txt').write_text(stream.getvalue())
    (out/'fourier.json').write_text(json.dumps(dict(rows=rows,production_enabled=False,
        note='First kValue includes lazy Hankel work; draw includes FFT/setup/copies. Python profiling does not isolate every internal C++ FFT/Hankel call. Finite state tests do not validate a Fourier renderer over the prior.'),indent=2)+'\n')
    scene.close()


if __name__=='__main__':main()
