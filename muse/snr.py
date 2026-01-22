import numpy as np

def nai_snr(wave, flux, err):
    """measure the median signal to noise of the NaI continuum"""
    windows = [(5865, 5875), (5915, 5925)]

    wave_window = (wave>=windows[0][0]) & (wave<=windows[0][1]) | (wave>=windows[1][0]) & (wave<=windows[1][1])
    wave_inds = np.where(wave_window)[0]

    flux_select = flux[wave_inds]
    sigma_select = err[wave_inds]
    
    real = np.isfinite(flux_select) & np.isfinite(sigma_select) & (sigma_select > 0)

    if np.sum(real) == 0:
        return 0

    return np.median(flux_select[real] / sigma_select[real])