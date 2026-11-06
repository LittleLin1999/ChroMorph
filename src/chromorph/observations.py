"""Observations are electron rates; component SED fluxes are calibrated microJy."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import uuid
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class EmpiricalPSF:
    image: np.ndarray
    pixel_scale: float
    center: tuple[float, float] | None = None
    pixel_integrated: bool = True
    provenance: dict = field(default_factory=dict)
    _cache_uid: str = field(default_factory=lambda: uuid.uuid4().hex, init=False, repr=False)

    def __post_init__(self):
        self.image = np.array(self.image, dtype=float, copy=True)
        if self.image.ndim != 2 or not np.isfinite(self.image).all() or self.image.sum() <= 0:
            raise ValueError("PSF must be finite 2D with positive total flux")
        if not np.isfinite(self.pixel_scale) or self.pixel_scale <= 0:
            raise ValueError("PSF pixel scale must be positive")
        if self.center is None:
            self.center = ((self.image.shape[1]-1)/2, (self.image.shape[0]-1)/2)
        if len(self.center) != 2 or not np.isfinite(self.center).all():
            raise ValueError("PSF requires a finite center")
        x, y = self.center
        if not (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]):
            raise ValueError("PSF center outside its stamp")
        self.image.setflags(write=False)
        self._image_reference = self.image
        self._image_version = 0

    def cache_token(self):
        if self.image is not self._image_reference:
            image = np.array(self.image, dtype=float, copy=True)
            if image.ndim != 2 or not np.isfinite(image).all() or image.sum() <= 0:
                raise ValueError('Invalid replacement empirical PSF')
            image.setflags(write=False)
            self.image = self._image_reference = image
            self._image_version += 1
        return (self._cache_uid, self._image_version, tuple(self.center), self.pixel_scale,
                self.pixel_integrated)

    def qa(self):
        positive = np.maximum(self.image, 0)
        yy, xx = np.indices(positive.shape)
        total = positive.sum()
        centroid = [float((positive*xx).sum()/total), float((positive*yy).sum()/total)]
        radius = np.hypot(xx-self.center[0], yy-self.center[1])
        order = np.argsort(radius.ravel())
        cumulative = np.cumsum(positive.ravel()[order])/total
        radii = radius.ravel()[order]
        ee = {str(q): float(radii[np.searchsorted(cumulative, q)]*self.pixel_scale)
              for q in [.5, .8, .9, .95]}
        border = np.zeros_like(positive, dtype=bool)
        border[:2] = True; border[-2:] = True; border[:, :2] = True; border[:, -2:] = True
        support_radius = float(radius[positive > positive.max()*1e-10].max())
        support_edge = radius >= max(0., support_radius-2)
        # Area-equivalent half-maximum diameter, diagnostic for noncircular PSFs.
        fwhm = 2*np.sqrt(np.count_nonzero(positive >= positive.max()/2)/np.pi)*self.pixel_scale
        return dict(sum=float(self.image.sum()), centroid_pixels=centroid,
                    declared_center_pixels=list(self.center),
                    centroid_offset_pixels=float(np.linalg.norm(np.array(centroid)-self.center)),
                    fwhm_area_equivalent_arcsec=float(fwhm), encircled_energy_radii_arcsec=ee,
                    outer_two_pixel_positive_fraction=float(positive[border].sum()/total),
                    outer_support_annulus_positive_fraction=float(positive[support_edge].sum()/total),
                    positive_support_radius_pixels=support_radius,
                    negative_absolute_fraction=float(np.abs(self.image[self.image < 0]).sum()/total),
                    pixel_integrated=self.pixel_integrated, provenance=self.provenance)


@dataclass
class Observation:
    name: str
    band: str
    data: np.ndarray
    sigma: np.ndarray
    psf: EmpiricalPSF
    pixel_scale: float
    ujy_per_rate: float
    mask: np.ndarray | None = None  # True means INCLUDED
    matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    background: float = 0.0
    header: object = None
    throughput: object = None
    pivot_angstrom: float | None = None
    provenance: dict = field(default_factory=dict)
    covariance: object = None
    _likelihood_revision: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self.data = np.array(self.data, dtype=float, copy=True)
        self.sigma = np.array(self.sigma, dtype=float, copy=True)
        if self.data.ndim != 2 or self.sigma.shape != self.data.shape:
            raise ValueError("data and sigma must be equal 2D shapes")
        if self.mask is None:
            self.mask = np.ones(self.data.shape, dtype=bool)
        else:
            self.mask = np.array(self.mask, dtype=bool, copy=True)
        if self.mask.shape != self.data.shape:
            raise ValueError("mask shape mismatch")
        self.mask &= np.isfinite(self.data) & np.isfinite(self.sigma) & (self.sigma > 0)
        if not self.mask.any():
            raise ValueError("Observation has no usable pixels")
        if not np.isfinite([self.pixel_scale, self.ujy_per_rate, self.background]).all() or min(self.pixel_scale, self.ujy_per_rate) <= 0:
            raise ValueError("Invalid scale, calibration or sky")
        self.matrix = np.array(self.matrix, dtype=float, copy=True)
        self.offset = np.array(self.offset, dtype=float, copy=True)
        if self.matrix.shape != (2, 2) or self.offset.shape != (2,) or not np.isfinite(self.matrix).all() or not np.isfinite(self.offset).all() or abs(np.linalg.det(self.matrix)) < 1e-8:
            raise ValueError("Invalid reference-to-observation affine transform")
        if not np.isclose(self.pixel_scale, self.psf.pixel_scale, rtol=.005):
            raise ValueError("Science and empirical PSF pixel scales disagree")

    def likelihood_token(self):
        return (self._likelihood_revision, id(self.data), id(self.sigma), id(self.mask), id(self.covariance), self.data.shape)

    def freeze(self):
        """Make fixed likelihood inputs immutable; use update() for changes."""
        self.__post_init__()
        for array in [self.data, self.sigma, self.mask]:
            array.setflags(write=False)
        if isinstance(self.covariance, np.ndarray):
            from .noise import DenseCovariance
            self.covariance = DenseCovariance(self.covariance)
        self._likelihood_revision += 1

    def update(self, **changes):
        unknown = set(changes)-{'data', 'sigma', 'mask', 'covariance'}
        if unknown:
            raise ValueError('Unsupported observation updates: '+str(unknown))
        for name, value in changes.items():
            setattr(self, name, value)
        self.freeze()

    @classmethod
    def from_fits(cls, path, psf, *, name, band, sigma_ext="ERR", mask_ext=None):
        with fits.open(path) as hd:
            h = hd["SCI"].header.copy()
            # WHT is deliberately not assumed to be a complete noise model.
            if sigma_ext not in hd:
                raise ValueError("A calibrated uncertainty map is required; WHT alone is insufficient")
            if str(h.get("BUNIT", "")).upper() not in {"ELECTRONS/S", "ELECTRON/S"}:
                raise ValueError("Input SCI must be in electrons/s")
            scale = float(np.sqrt(np.prod(proj_plane_pixel_scales(WCS(h).celestial)))*3600)
            pivot = float(h["PHOTPLAM"])
            factor = float(h["PHOTFLAM"])*pivot**2/2.99792458e18*1e29
            return cls(name, band, hd["SCI"].data, hd[sigma_ext].data, psf,
                       scale, factor, mask=None if mask_ext is None else hd[mask_ext].data.astype(bool),
                       header=h, pivot_angstrom=pivot,
                       provenance={"science": str(path), "sha256": sha256(path)})
