"""YAML parsing with explicit component settings and rejection of unknown keys."""
from pathlib import Path
from dataclasses import replace
import yaml
from .parameters import Parameter
from .scene import Scene
from .sed import FreeBandSED, FSPSStellarSED, SEDContext
from .spatial import PointSource, Sersic


def _keys(mapping, allowed, where):
    unknown = set(mapping)-set(allowed)
    if unknown:
        raise ValueError(f"Unknown {where} options: {sorted(unknown)}")


def parameter(spec):
    if isinstance(spec, (int, float)):
        return Parameter(float(spec), vary=False)
    _keys(spec, ['value', 'min', 'max', 'vary', 'prior'], 'parameter')
    return Parameter(float(spec['value']), float(spec.get('min', '-inf')),
                     float(spec.get('max', 'inf')), bool(spec.get('vary', True)),
                     None if spec.get('prior') is None else tuple(spec['prior']))


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    _keys(config, ['target', 'data_root', 'observations', 'metadata', 'morphology', 'psf', 'context',
                   'components', 'ties', 'centroid_groups', 'throughputs', 'fitting', 'initial_values'], 'top-level')
    morphology = config.get('morphology', {})
    _keys(morphology, ['wavelength_mode', 'centroid_mode', 'renderer'], 'morphology')
    if morphology.get('wavelength_mode', 'constant') != 'constant' or morphology.get('renderer', 'galsim') != 'galsim':
        raise ValueError("ChroMorph requires constant morphology and the GalSim renderer")
    if morphology.get('centroid_mode', 'independent') not in ['independent', 'grouped', 'tied']:
        raise ValueError("Unknown centroid mode")
    psf = config.get('psf', {})
    _keys(psf, ['source', 'already_drizzled'], 'PSF')
    if 'observations' in config:
        manifest = Path(config['observations'])
        config['observations'] = str(manifest if manifest.is_absolute() else Path(path).resolve().parent/manifest)
    elif 'data_root' in config:
        root = Path(config['data_root'])
        config['data_root'] = str(root if root.is_absolute() else Path(path).resolve().parent/root)
    return config


def build_scene(config, observations, *, renderer=None, context=None):
    fit = config.get('fitting', {})
    _keys(fit, ['optimizer', 'multistart', 'seed', 'max_nfev', 'variable_projection',
                'background', 'registration', 'calibration', 'compare_sed_modes', 'workers',
                'method', 'likelihood', 'project_stellar_mass', 'estimate_uncertainty',
                'checkpoint_directory', 'tolerance'], 'fitting')
    if fit.get('optimizer', 'scipy') not in {'scipy', 'trf', 'L-BFGS-B'}:
        raise ValueError('Use scipy, trf or L-BFGS-B for deterministic fitting')
    if fit.get('compare_sed_modes', False):
        raise ValueError("Use the compare CLI with separate A/B/C files to specify each SED choice explicitly")
    scene = Scene(observations, context=context or SEDContext(**config.get('context', {})), renderer=renderer,
                  fit_background=fit.get('background', False), fit_registration=fit.get('registration', False),
                  fit_calibration=fit.get('calibration', False))
    try:
        for row in config['components']:
            _keys(row, ['name', 'spatial', 'nuclear', 'sed', 'parameters'], 'component')
            sedconfig = dict(row['sed']); model = sedconfig.pop('model')
            overrides = {k: parameter(v) for k, v in sedconfig.pop('parameters', {}).items()}
            if model == 'free':
                _keys(sedconfig, ['fluxes'], 'free SED')
                sed = FreeBandSED(parameters=overrides, **sedconfig)
            elif model == 'fsps':
                _keys(sedconfig, ['sfh', 'nebular', 'dust', 'dust_type', 'cache_size', 'cache_bytes'], 'stellar SED')
                sed = FSPSStellarSED(parameters=overrides, **sedconfig)
            else:
                raise ValueError(f"Unknown SED model {model}")
            pars = {k: parameter(v) for k, v in row.get('parameters', {}).items()}
            if row['spatial'] == 'point':
                component = PointSource(row['name'], sed, nuclear=row.get('nuclear', row['name'] == 'nucleus'), parameters=pars)
            elif row['spatial'] == 'sersic':
                if row.get('nuclear'):
                    raise ValueError("Nucleus must be an unresolved point source")
                component = Sersic(row['name'], sed, parameters=pars)
            else:
                raise ValueError("Spatial model must be point or sersic")
            scene.add_component(component)
        for group in config.get('centroid_groups', []):
            scene.tie_centroids(group)
        for alias, target in config.get('ties', {}).items():
            scene.tie(alias, target)
        mode = config.get('morphology', {}).get('centroid_mode', 'independent')
        if mode == 'tied':
            scene.tie_centroids([c.name for c in scene.components])
        elif mode == 'grouped' and not config.get('centroid_groups'):
            raise ValueError("Grouped centroids require explicit centroid_groups")
        for name, value in config.get('initial_values', {}).items():
            if name not in scene.registry.parameters or name in scene.registry.ties:
                raise ValueError('Initial value requires a registered untied parameter: '+name)
            scene.registry.parameters[name] = replace(scene.registry.parameters[name], value=float(value))
        return scene
    except Exception:
        scene.close()
        raise
