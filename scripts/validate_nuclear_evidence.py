"""Independent nuclear-flux priors: analytic evidence and width sensitivity.

Run on an MPI installation of public PolyChordLite, for example:
  mpiexec -n 4 python scripts/validate_nuclear_evidence.py --output /path/to/results
All images are generated synthetically; no target data are used.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.special import ndtr
from scipy.stats import truncnorm
from astropy.table import Table
from chromorph import Scene, Observation, EmpiricalPSF, PointSource, FreeBandSED, Parameter, Prior
from chromorph.inference import PolyChordSampler
from chromorph.rendering import GalSimRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', type=int, nargs='+', default=[64, 192])
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 29])
    parser.add_argument('--widths', type=float, nargs='+', default=[5., 10., 20.])
    parser.add_argument('--num-repeats', type=int, default=10)
    parser.add_argument('--precision', type=float, default=.03)
    parser.add_argument('--second-filter-width', type=float,
                        help='Hold red fixed while changing blue; default changes both independent supports')
    args = parser.parse_args()
    if any(w <= 0 for w in args.widths) or (args.second_filter_width is not None and args.second_filter_width <= 0):
        parser.error('Uniform widths must be positive')
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    reports = []
    started = time.perf_counter()
    for width in args.widths:
        widths = dict(blue=width, red=args.second_filter_width or width)
        for live in args.live:
            for seed in args.seeds:
                yy, xx = np.indices((19, 19))
                psf = np.exp(-((xx-9)**2+(yy-9)**2)/2)
                psf /= psf.sum()
                obs = [Observation(b, b, np.zeros((13, 13)), np.full((13, 13), .03),
                                   EmpiricalPSF(psf, .04), .04, 1.) for b in widths]
                with Scene(obs, renderer=GalSimRenderer()) as scene:
                    scene.add_component(PointSource('nucleus', FreeBandSED(parameters={
                        b: Parameter(min(1., w/2), 0., np.inf, prior=Prior('uniform', lower=0., upper=w))
                        for b, w in widths.items()}), parameters={k: Parameter(6., vary=False) for k in ['x', 'y']}))
                    values = scene.registry.unpack()
                    values.update({'nucleus.flux.blue': 1.2, 'nucleus.flux.red': .8})
                    models, _, _ = scene.predict(values)
                    rng = np.random.default_rng(71)
                    for o in obs:
                        o.update(data=models[o.name]+rng.normal(0, o.sigma, o.data.shape))
                    basis = scene.basis(values)
                    expected = 0.
                    truth = {}
                    for o in obs:
                        column = (basis[o.name]['nucleus']/o.sigma).ravel()
                        data = (o.data/o.sigma).ravel()
                        sigma = 1/np.sqrt(column@column)
                        mean = column@data/(column@column)
                        remaining = data-column*mean
                        a, b = -mean/sigma, (widths[o.band]-mean)/sigma
                        probability = ndtr(b)-ndtr(a)
                        expected += (-.5*(remaining@remaining)+.5*np.log(2*np.pi)
                                     +np.log(sigma)+np.log(probability)-np.log(widths[o.band]))
                        expected += scene.gaussian_data(o).whitening.log_normalization
                        distribution = truncnorm(a, b, loc=mean, scale=sigma)
                        truth[o.band] = dict(mean=float(distribution.mean()),
                                             sigma=float(distribution.std()),
                                             p16=float(distribution.ppf(.16)),
                                             p84=float(distribution.ppf(.84)))
                    sampler = PolyChordSampler(scene)
                    label = f'width{width:g}' if args.second_filter_width is None else f'blue{width:g}_red{widths["red"]:g}'
                    path = args.output/f'{label}_live{live}_seed{seed}'
                    output = sampler.run(path, nlive=live, num_repeats=args.num_repeats,
                        precision_criterion=args.precision, seed=seed, read_resume=False,
                        write_resume=True, feedback=0, restore_samples=1000)
                    if rank == 0:
                        run = json.loads((path/'run_summary.json').read_text())
                        table = Table.read(path/'posterior_weighted.ecsv')
                        weights = np.asarray(table['weight'])
                        posterior = {}
                        for band in widths:
                            flux = np.asarray(table['nucleus.flux.'+band])
                            mean = weights@flux
                            variance = weights@(flux-mean)**2
                            posterior[band] = dict(mean=float(mean), sigma=float(np.sqrt(variance)),
                                mean_minus_analytic=float(mean-truth[band]['mean']),
                                near_prior_boundary_probability=float(weights[(flux<.01*widths[band]) |
                                                                              (flux>.99*widths[band])].sum()))
                        fluxes = np.column_stack([table['nucleus.flux.'+b] for b in widths])
                        centered = fluxes-weights@fluxes
                        covariance = centered.T@(weights[:, None]*centered)
                        correlation = covariance/np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
                        row = dict(width=width, widths=widths, nlive=live, seed=seed,
                            analytic_log_evidence=float(expected), measured_log_evidence=float(output.logZ),
                            reported_log_evidence_error=float(output.logZerr), difference=float(output.logZ-expected),
                            difference_over_reported_error=float((output.logZ-expected)/output.logZerr),
                            posterior_reference=truth, posterior_measured=posterior,
                            posterior_correlation=correlation.tolist(), summary=run)
                        reports.append(row)
                        args.output.mkdir(parents=True, exist_ok=True)
                        comparison = []
                        for resolution in args.live:
                            for repeat_seed in args.seeds:
                                selected = [r for r in reports if r['nlive']==resolution and r['seed']==repeat_seed]
                                for first, second in zip(selected, selected[1:]):
                                    comparison.append(dict(nlive=resolution, seed=repeat_seed,
                                        widths_from=first['widths'], widths_to=second['widths'],
                                        analytic_log_evidence_change=second['analytic_log_evidence']-first['analytic_log_evidence'],
                                        measured_log_evidence_change=second['measured_log_evidence']-first['measured_log_evidence'],
                                        posterior_mean_changes={b:second['posterior_measured'][b]['mean']-
                                                                first['posterior_measured'][b]['mean'] for b in widths}))
                        report = dict(runs=reports, prior_width_sensitivity=comparison,
                            settings=dict(live=args.live, seeds=args.seeds, widths=args.widths,
                                          second_filter_width=args.second_filter_width,
                                          num_repeats=args.num_repeats, precision=args.precision),
                            total_seconds=time.perf_counter()-started, independent_flux_priors=True,
                            nuclear_spectral_model=None, variable_projection_used=False,
                            gaussian_normalization_retained=True,
                            validation_scope='fixed-geometry two-filter synthetic source; no arbitrary galaxy convergence claim')
                        (args.output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
                        print(json.dumps({k:row[k] for k in ['widths','nlive','seed','difference','reported_log_evidence_error']}), flush=True)


if __name__ == '__main__':
    main()
