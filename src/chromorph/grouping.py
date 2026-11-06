"""Physical reporting roles are independent of spatial profile type."""
import numpy as np
from .spatial import PointSource


def component_groups(components):
    nucleus = [c.name for c in components if isinstance(c, PointSource)
               and (getattr(c, 'nuclear', False) or c.name in {'nucleus', 'nuclear'})]
    extended = [c.name for c in components if c.name not in nucleus]
    psf_clumps = [c.name for c in components if isinstance(c, PointSource) and c.name in extended]
    return dict(nucleus=nucleus, extended=extended, psf_clumps=psf_clumps,
                definition='Nucleus: declared nuclear point source(s). Extended: all non-nuclear components, including PSF clumps.')


def grouped_images(components, images):
    groups = component_groups(components)
    template = next(iter(images.values()))
    def summed(names):
        return sum((images[name] for name in names), np.zeros_like(template))
    return dict(nucleus=summed(groups['nucleus']), extended=summed(groups['extended']),
                psf_clumps=summed(groups['psf_clumps']))
