"""Resolution-aware spectral subtraction for diagnostic plots, not fitting."""
import numpy as np
from scipy.special import erf


def pixel_edges(wave):
    wave = np.asarray(wave, dtype=float)
    if len(wave) < 2 or np.any(np.diff(wave) <= 0):
        raise ValueError('Wavelengths must increase strictly')
    return np.r_[wave[0]-(wave[1]-wave[0])/2, (wave[1:]+wave[:-1])/2,
                 wave[-1]+(wave[-1]-wave[-2])/2]


def resample_flux(wave, flux, target):
    """Integrate the piecewise-linear f_lambda model over target pixel bins."""
    wave, flux = np.asarray(wave), np.asarray(flux)
    edges = pixel_edges(target)
    cumulative = np.r_[0., np.cumsum(np.diff(wave)*(flux[1:]+flux[:-1])/2)]
    index = np.clip(np.searchsorted(wave, edges, side='right')-1, 0, len(wave)-2)
    dx = edges-wave[index]
    slope = (flux[index+1]-flux[index])/(wave[index+1]-wave[index])
    integral = cumulative[index]+flux[index]*dx+slope*dx**2/2
    result = np.diff(integral)/np.diff(edges)
    covered = (edges[:-1] >= wave[0]) & (edges[1:] <= wave[-1])
    return np.where(covered, result, np.nan)


def desi_resolution_sigma(wave, diagonals):
    """Gaussian-equivalent row second moments of DESI's stored DIA operator.

    The diagonal ordering follows public desispec.Resolution. This preserves
    measured wavelength-dependent widths, but not non-Gaussian LSF structure.
    """
    wave, diagonal = np.asarray(wave), np.asarray(diagonals)
    if diagonal.ndim != 2 or diagonal.shape[1] != len(wave) or diagonal.shape[0] % 2 != 1:
        raise ValueError('DESI resolution requires odd ndiag by nwave')
    half = diagonal.shape[0]//2
    weight, first, second = np.zeros((3, len(wave)))
    for k, offset in enumerate(range(half, -half-1, -1)):
        row = np.arange(max(0, -offset), min(len(wave), len(wave)-offset))
        column = row+offset
        value = np.maximum(diagonal[k, column], 0.)
        delta = wave[column]-wave[row]
        weight[row] += value;first[row] += value*delta;second[row] += value*delta**2
    mean = np.divide(first, weight, out=np.zeros_like(first), where=weight > 0)
    variance = np.divide(second, weight, out=np.zeros_like(second), where=weight > 0)-mean**2
    return np.sqrt(np.maximum(variance, 0.))


def gaussian_smooth(wave, flux, sigma_angstrom, *, good=None, variance=None):
    """Apply normalized, pixel-integrated, variable Gaussian kernels locally."""
    wave, flux, sigma = np.asarray(wave), np.asarray(flux), np.asarray(sigma_angstrom)
    good = np.isfinite(flux) if good is None else np.asarray(good) & np.isfinite(flux)
    edges = pixel_edges(wave)
    output = np.full_like(flux, np.nan);error = np.full_like(flux, np.nan)
    valid = np.zeros(len(wave), dtype=bool)
    for i, width in enumerate(sigma):
        if width < 1e-8:
            if good[i]:
                output[i] = flux[i];valid[i] = True
                if variance is not None:error[i] = np.sqrt(variance[i])
            continue
        lo = max(0, np.searchsorted(edges, wave[i]-5*width)-1)
        hi = min(len(wave), np.searchsorted(edges, wave[i]+5*width))
        weight = np.diff(erf((edges[lo:hi+1]-wave[i])/(np.sqrt(2)*width)))/2
        accepted = weight*good[lo:hi]
        if accepted.sum() < .99*weight.sum() or accepted.sum() <= 0:continue
        accepted /= accepted.sum()
        output[i] = accepted@np.where(good[lo:hi], flux[lo:hi], 0.)
        valid[i] = True
        if variance is not None:
            error[i] = np.sqrt(np.sum(accepted**2*np.where(good[lo:hi], variance[lo:hi], 0.)))
    return output, valid, error


def subtract_matched_extended(wave, total, good, variance, model_wave, extended,
                              instrument_sigma_angstrom, model_sigma_kms):
    """Subtract at a common Gaussian-equivalent resolution without sharpening.

    If data are sharper than the stellar library, broaden data as well. The
    common variance is max(instrument variance, library variance plus model
    bin-integration variance). No model deconvolution or flux renormalization.
    """
    wave = np.asarray(wave)
    model = resample_flux(model_wave, extended, wave)
    sampling_sigma = np.diff(pixel_edges(wave))/np.sqrt(12.)
    template_sigma = np.hypot(wave*np.asarray(model_sigma_kms)/299792.458, sampling_sigma)
    instrument = np.asarray(instrument_sigma_angstrom)
    common = np.maximum(instrument, template_sigma)
    add_model = np.sqrt(np.maximum(common**2-template_sigma**2, 0.))
    add_data = np.sqrt(np.maximum(common**2-instrument**2, 0.))
    matched_model, model_good, _ = gaussian_smooth(wave, model, add_model)
    matched_total, data_good, error = gaussian_smooth(wave, total, add_data, good=good, variance=variance)
    valid = model_good & data_good
    return dict(wave=wave, total=matched_total, extended=matched_model,
                nuclear=np.where(valid, matched_total-matched_model, np.nan),
                sigma_total=error, good=valid, common_sigma_angstrom=common,
                fraction_data_broadened=float(np.mean(add_data > 1e-8)))
