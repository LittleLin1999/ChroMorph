"""Reproducible discovery, fit, comparison and injection/recovery commands."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import sys
import numpy as np
import yaml
from .config import load_config, build_scene
from .discovery import discover_j1042, attach_hst_throughputs, TARGET
from .result import _json
from .sed import SEDContext


def _dump(path, value):
    Path(path).write_text(json.dumps(_json(value), indent=2, allow_nan=False)+'\n')


def doctor():
    names = ['numpy', 'scipy', 'astropy', 'galsim', 'yaml', 'matplotlib', 'fsps', 'synphot', 'stsynphot']
    result = {n: importlib.util.find_spec(n) is not None for n in names}
    result['note'] = 'Importability alone does not validate FSPS data or STSynphot reference tables.'
    return result


def example_configs(report):
    guess = report['initial_guess']
    components = []
    for c in guess['components']:
        if not c['enabled']:
            continue
        pars = {}
        for name, p in c['parameters'].items():
            name = 'pa' if name == 'pa_deg' else name
            pars[name] = dict(value=p['value'], min=p['min'], max=p['max'], vary=p['free'])
            if name == 'n':
                if p['value'] > 6.2:
                    raise ValueError("Initial index outside GalSim's supported range")
                pars[name]['min'] = max(.3, pars[name]['min'])
                pars[name]['max'] = min(6.2, pars[name]['max'])
        components.append(dict(name='nucleus' if c['type'] == 'point' else c['id'], spatial=c['type'],
                               nuclear=c['type'] == 'point', sed=dict(model='free'), parameters=pars))
    base = dict(target=TARGET, data_root=report['root'],
                morphology=dict(wavelength_mode='constant', centroid_mode='independent', renderer='galsim'),
                psf=dict(source='empirical_v2', already_drizzled=True),
                context={} if report['redshift'] is None else dict(redshift=report['redshift']),
                components=components,
                fitting=dict(optimizer='scipy', multistart=3, seed=1042, max_nfev=300,
                             variable_projection=True, background=False, registration=True))
    configs = {'A_free': copy.deepcopy(base), 'B_independent_stellar': copy.deepcopy(base), 'C_hybrid': copy.deepcopy(base)}
    for mode in ['B_independent_stellar', 'C_hybrid']:
        count = 0
        for c in configs[mode]['components']:
            if c['spatial'] != 'sersic':
                continue
            count += 1
            if mode == 'C_hybrid' and count > 2:
                continue
            c['sed'] = dict(model='fsps', sfh='ssp', nebular=False, dust=True,
                            parameters=dict(log_mass=dict(value=8.5, min=4, max=12),
                                            age=dict(value=.3, min=.001, max=6),
                                            metallicity=dict(value=-.3, min=-2, max=.3),
                                            av=dict(value=.2, min=0, max=4)))
    return configs


def observations_for(config):
    if config.get('target', TARGET) not in ['J1042', TARGET]:
        raise ValueError('The built-in discovery CLI currently supports J1042; use Observation/Scene for other targets')
    observations, report = discover_j1042(config['data_root'], metadata=config.get('metadata'))
    if report['blockers']:
        raise ValueError('Input QA failed: '+ '; '.join(report['blockers']))
    stellar = any(c['sed']['model'] == 'fsps' for c in config['components'])
    if stellar:
        attach_hst_throughputs(observations, config.get('throughputs'))
        if report['redshift'] is None and not config.get('context', {}).get('distance_mpc'):
            raise ValueError('Physical SED requires measured redshift or distance')
    context = dict(config.get('context', {}))
    if 'redshift' not in context and report['redshift'] is not None:
        context['redshift'] = report['redshift']
    return observations, report, SEDContext(**context)


def run_fit(config, output):
    observations, report, context = observations_for(config)
    options = config.get('fitting', {})
    with build_scene(config, observations, context=context) as scene:
        from .rendering import GalSimRenderer
        scene.renderer = GalSimRenderer()
        report['galsim_sampling'] = {o.name: scene.renderer.validate_psf_sampling(o) for o in observations}
        if not all(v['pass_same_grid'] for v in report['galsim_sampling'].values()):
            raise ValueError('Empirical PSF interpolation round-trip failed')
        result = scene.fit(**{k: options[k] for k in ['multistart', 'seed', 'max_nfev', 'variable_projection'] if k in options})
        result.write(output, configuration=config)
        _dump(Path(output)/'input_qa.json', report)
        return dict(chi2=result.chi2, bic_like=result.bic_like, uncertainty=result.uncertainty,
                    identifiability=result.identifiability, warnings=result.warnings,
                    success=min(result.runs, key=lambda r: r['objective'])['success'])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='chromorph')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor')
    discover = commands.add_parser('discover-j1042')
    discover.add_argument('root', type=Path)
    discover.add_argument('--metadata', type=Path)
    discover.add_argument('--output', type=Path, required=True)
    discover.add_argument('--config-dir', type=Path)
    fit = commands.add_parser('fit')
    fit.add_argument('config', type=Path); fit.add_argument('--output', type=Path, required=True)
    compare = commands.add_parser('compare')
    compare.add_argument('configs', type=Path, nargs='+'); compare.add_argument('--output', type=Path, required=True)
    validate = commands.add_parser('validate')
    validate.add_argument('config', type=Path); validate.add_argument('--output', type=Path, required=True)
    validate.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        print(json.dumps(doctor(), indent=2)); return
    if args.command == 'discover-j1042':
        _, report = discover_j1042(args.root, metadata=args.metadata)
        args.output.parent.mkdir(parents=True, exist_ok=True); _dump(args.output, report)
        if args.config_dir:
            args.config_dir.mkdir(parents=True, exist_ok=True)
            for mode, config in example_configs(report).items():
                (args.config_dir/(mode+'.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
        print(json.dumps(dict(filters=[f['band'] for f in report['filters']], blockers=report['blockers'], redshift=report['redshift']), indent=2))
        return
    if args.command == 'fit':
        result = run_fit(load_config(args.config), args.output)
        print(json.dumps(_json(result), indent=2)); return
    if args.command == 'compare':
        configs = [load_config(path) for path in args.configs]
        signature = None
        from hashlib import sha256
        for config in configs:
            observations, _, _ = observations_for(config)
            current = [(o.name, sha256(o.data.tobytes()+o.sigma.tobytes()+o.mask.tobytes()+o.psf.image.tobytes()).hexdigest()) for o in observations]
            if signature is not None and current != signature:
                raise ValueError('Comparisons require identical input images, noise, masks and PSFs')
            signature = current
        results = {}
        args.output.mkdir(parents=True, exist_ok=True)
        for path, config in zip(args.configs, configs):
            if path.stem in results:
                raise ValueError('Configuration filenames must be unique')
            results[path.stem] = run_fit(config, args.output/path.stem)
        _dump(args.output/'comparison.json', results)
        rows = ''.join('<tr><td>{}</td><td>{:.3f}</td><td>{:.3f}</td></tr>'.format(k, v['chi2'], v['bic_like']) for k, v in results.items())
        (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SED model comparison</h1><p>Identical image likelihood inputs. Lower BIC-like is not proof of more accurate morphology; inspect recovery and degeneracies.</p><table><tr><th>Model</th><th>chi2</th><th>BIC-like</th></tr>'+rows+'</table>')
        return
    if args.command == 'validate':
        from .simulation import recovery_experiment
        config = load_config(args.config)
        observations, report, context = observations_for(config)
        with build_scene(config, observations, context=context) as injection:
            def factory(data):
                return build_scene(config, data, context=context)
            result = recovery_experiment(injection, factory, repeats=args.repeats,
                                         fit_options={k: v for k, v in config.get('fitting', {}).items()
                                                      if k in ['multistart', 'seed', 'max_nfev', 'variable_projection']})
            args.output.parent.mkdir(parents=True, exist_ok=True); _dump(args.output, result)


if __name__ == '__main__':
    main()
