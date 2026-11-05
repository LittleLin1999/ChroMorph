"""PolyChord posterior inference without imposing a nuclear spectral prior.

Independent nonnegative nuclear amplitudes have an improper flat measure. They
are integrated analytically, not profiled. This gives a normalized posterior
when the likelihood is integrable, but does NOT define a Bayesian evidence.
"""
from pathlib import Path
import json
import time
from dataclasses import asdict
import numpy as np
from scipy.special import log_ndtr
from scipy.stats import truncnorm


def marginal_linear_flux(residual, column):
    """Log integral over F>=0 and the underlying Gaussian mean/sigma.

    residual and column are already divided by the observational sigma.
    The reference integration measure is dF with F expressed in microJy.
    """
    precision = float(column @ column)
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError('Unobserved nuclear flux: flat-prior posterior is improper')
    sigma = precision**-.5
    mean = float(column @ residual)/precision
    # Subtract the unconstrained optimum explicitly to avoid catastrophic
    # cancellation between two large quadratic forms.
    remaining = residual-column*mean
    log_integral = (-.5*float(remaining @ remaining) + .5*np.log(2*np.pi)
                    + np.log(sigma) + log_ndtr(mean/sigma))
    return float(log_integral), mean, sigma


class PolyChordPosterior:
    def __init__(self, scene):
        from .spatial import PointSource
        self.scene = scene
        self.linear = [n for c in scene.components
                       if isinstance(c, PointSource) and (c.nuclear or c.name in {'nucleus', 'nuclear'})
                       for n in scene._eligible_linear() if n.startswith(c.name+'.flux.')]
        if len({n.split('.')[-1] for n in self.linear}) != len(self.linear):
            raise ValueError('Analytic nuclear marginalization requires one nuclear amplitude per band')
        self.names = scene.registry.names(exclude=self.linear)
        self.lower, self.upper = scene.registry.bounds(self.names)
        for name, lo, hi in zip(self.names, self.lower, self.upper):
            if scene.registry.parameters[name].prior is None and not np.isfinite([lo, hi]).all():
                raise ValueError('PolyChord needs a proper prior for '+name)
        self.flux_measure = 'independent flat F>=0, dF in microJy; no finite upper limit'

    def prior(self, unit_cube):
        cube = np.asarray(unit_cube, dtype=float)
        if cube.shape != (len(self.names),) or np.any((cube < 0) | (cube > 1)):
            raise ValueError('Invalid prior unit cube')
        cube = np.clip(cube, np.finfo(float).eps, 1-np.finfo(float).eps)
        values = np.empty_like(cube)
        for i, (name, lo, hi) in enumerate(zip(self.names, self.lower, self.upper)):
            prior = self.scene.registry.parameters[name].prior
            if prior is None:
                values[i] = lo+cube[i]*(hi-lo)
            else:
                mean, sigma = prior
                values[i] = truncnorm.ppf(cube[i], (lo-mean)/sigma, (hi-mean)/sigma,
                                          loc=mean, scale=sigma)
        return values

    def evaluate(self, vector):
        scene = self.scene
        values = scene.registry.unpack(vector, self.names)
        for name in self.linear:
            values[name] = 0.
        statistics = scene.compressed.statistics(values)
        coefficients = scene.compressed.coefficients(values)
        bands = scene.bands
        conditional, total = {}, 0.
        for band in bands:
            observations = [o for o in scene.observations if o.band == band]
            residual = np.concatenate([statistics[o.name].z-statistics[o.name].R@coefficients[o.name]
                                       for o in observations])
            perpendicular = sum(statistics[o.name].perpendicular_chi2 for o in observations)
            parameter = next((n for n in self.linear if n.endswith('.'+band)), None)
            if parameter is None:
                total -= .5*float(residual @ residual)
            else:
                component = parameter.split('.')[0]
                index = next(i for i, c in enumerate(scene.components) if c.name == component)
                column = np.concatenate([statistics[o.name].R[:, index]
                    *np.exp(values[f'observation.{o.name}.log_calibration'])/o.ujy_per_rate
                    for o in observations])
                log_integral, mean, sigma = marginal_linear_flux(residual, column)
                total += log_integral
                conditional[parameter] = (mean, sigma)
            total -= .5*perpendicular
            total += sum(statistics[o.name].log_normalization for o in observations)
        return float(total), conditional

    def loglikelihood(self, vector):
        value, _ = self.evaluate(vector)
        return value, []

    def restore_fluxes(self, samples, *, seed=0):
        """One exact conditional truncated-normal draw per posterior sample."""
        rng = np.random.default_rng(seed)
        output = np.empty((len(samples), len(self.names)+len(self.linear)))
        output[:, :len(self.names)] = samples
        for i, row in enumerate(samples):
            _, conditional = self.evaluate(row)
            for j, name in enumerate(self.linear):
                mean, sigma = conditional[name]
                output[i, len(self.names)+j] = truncnorm.rvs(-mean/sigma, np.inf,
                    loc=mean, scale=sigma, random_state=rng)
        return self.names+self.linear, output

    def run(self, directory, *, restore_samples=2000, **options):
        """Run public PyPolyChord; each MPI rank owns a separate Scene.

        A best-fit initialization is deliberately not used to truncate priors.
        Pass nlive/num_repeats explicitly for reproducible production settings.
        """
        try:
            import pypolychord
            from pypolychord.settings import PolyChordSettings
        except ImportError as exc:
            raise ImportError('Install public PolyChordLite/PyPolyChord with its compiled library') from exc
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        metadata = dict(sampler='PolyChord', parameter_names=self.names,
            marginalized_parameters=self.linear, nuclear_flux_measure=self.flux_measure,
            parameter_definitions={n: asdict(p) for n, p in self.scene.registry.parameters.items()},
            ties=self.scene.registry.ties, context=asdict(self.scene.context),
            log_evidence=None,
            evidence_warning='Absolute Bayesian evidence is undefined with flat unbounded nuclear flux priors. PolyChord logZ is only a reference-measure integral.',
            settings=options)
        # Every MPI rank receives identical input, but only rank zero writes metadata.
        rank = 0
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.rank
        except ImportError:
            pass
        if rank == 0:
            from .result import _json
            (directory/'inference.json').write_text(json.dumps(_json(metadata), indent=2)+'\n')
        settings = PolyChordSettings(len(self.names), 0, base_dir=str(directory),
                                     file_root='posterior', **options)
        started = time.monotonic()
        def progress(live, dead, logweights, logZ, logZerr):
            if rank == 0:
                status = dict(state='sampling', n_dead=len(dead), n_live=len(live),
                    elapsed_seconds=time.monotonic()-started, log_evidence=None,
                    log_integral_reference_measure=float(logZ), log_integral_error=float(logZerr))
                (directory/'progress.json').write_text(json.dumps(_json(status), indent=2)+'\n')
        output = pypolychord.run_polychord(self.loglikelihood, len(self.names), 0,
                                          settings, self.prior, progress)
        if rank == 0:
            from astropy.table import Table
            raw = np.atleast_2d(np.loadtxt(directory/'posterior.txt'))
            weights = raw[:, 0]/raw[:, 0].sum()
            table = Table(raw[:, 2:], names=self.names)
            table['weight'] = weights
            table.write(directory/'posterior_weighted.ecsv', format='ascii.ecsv', overwrite=True)
            if restore_samples:
                rng = np.random.default_rng(options.get('seed', 0))
                selected = raw[rng.choice(len(raw), size=restore_samples, p=weights), 2:]
                names, restored = self.restore_fluxes(selected, seed=options.get('seed', 0))
                Table(restored, names=names).write(directory/'posterior_with_nucleus.ecsv',
                    format='ascii.ecsv', overwrite=True)
                quantiles = np.quantile(restored, [.16, .5, .84], axis=0)
                (directory/'posterior_summary.json').write_text(json.dumps({n:
                    dict(p16=float(q[0]), median=float(q[1]), p84=float(q[2]))
                    for n, q in zip(names, quantiles.T)}, indent=2)+'\n')
            summary = dict(state='sampling_complete', elapsed_seconds=time.monotonic()-started,
                effective_sample_size=float(1/(weights @ weights)), samples=len(raw),
                log_evidence=None, convergence_validated=False,
                note='Check independent seeds, slice lengths, component exchange and prior sensitivity before scientific interpretation.')
            (directory/'run_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return output
