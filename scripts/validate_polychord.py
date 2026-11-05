"""MPI end-to-end recovery of independent clump colours and nuclear fluxes.

Run with mpiexec -n 4 python scripts/validate_polychord.py OUTPUT_DIRECTORY.
The fixed-morphology linear problem has an independently calculable covariance.
"""
import json
from pathlib import Path
import sys
import numpy as np
from chromorph import Scene, PointSource, FreeBandSED, Parameter
from chromorph.inference import PolyChordPosterior


def main():
    from mpi4py import MPI
    # Reuse the analytic, exactly pixel-integrated Gaussian test fixture.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
    from conftest import observations, clump, GaussianTestRenderer
    output = Path(sys.argv[1])
    scene = Scene(observations(noise=.005), renderer=GaussianTestRenderer())
    scene.add_component(PointSource('nucleus', FreeBandSED({'blue': 2., 'red': 1.}),
        parameters={'x': Parameter(12., vary=False), 'y': Parameter(12., vary=False)}))
    truth = {'clump1': {'blue': 1., 'red': .5}, 'clump2': {'blue': .4, 'red': 1.5}}
    for name, x in [('clump1', 8.), ('clump2', 16.)]:
        scene.add_component(clump(name, x=x, sed=FreeBandSED(parameters={
            b: Parameter(f, 0., 5.) for b, f in truth[name].items()})))
    values = scene.registry.unpack()
    models, _, _ = scene.predict(values)
    for observation in scene.observations:
        observation.data[:] = models[observation.name]
    posterior = PolyChordPosterior(scene)
    posterior.run(output, nlive=160, num_repeats=16, precision_criterion=.02,
        seed=1042, feedback=0, read_resume=False, restore_samples=1500)
    if MPI.COMM_WORLD.rank == 0:
        from astropy.table import Table
        samples = Table.read(output/'posterior_with_nucleus.ecsv', format='ascii.ecsv')
        expected = {}
        basis = scene.basis(values)
        for observation in scene.observations:
            design = np.stack([(basis[observation.name][c.name]/observation.sigma)[observation.mask]
                               for c in scene.components], axis=1)
            covariance = np.linalg.inv(design.T @ design)
            for j, component in enumerate(scene.components):
                expected[f'{component.name}.flux.{observation.band}'] = covariance[j, j]**.5
        rows = []
        for name in samples.colnames:
            mean, sigma = float(np.mean(samples[name])), float(np.std(samples[name]))
            reference = values[name]
            np.testing.assert_allclose(mean, reference, atol=.3*expected[name])
            np.testing.assert_allclose(sigma, expected[name], rtol=.2)
            rows.append(dict(parameter=name, truth=reference, recovered_mean=mean,
                             recovered_sigma=sigma, expected_sigma=expected[name]))
        report = dict(passed=True, mpi_ranks=MPI.COMM_WORLD.size, rows=rows)
        (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    scene.close()


if __name__ == '__main__':
    main()
