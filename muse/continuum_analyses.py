import numpy as np

def emline_mask(flux: np.ndarray, wave: np.ndarray, cont_blim: tuple, cont_rlim: tuple, s = 1):

    bind = np.where((wave > cont_blim[0]) & (wave < cont_blim[1]))
    rind = np.where((wave > cont_rlim[0]) & (wave < cont_rlim[1]))

    continuum = np.concatenate((flux[bind], flux[rind]))
    med = np.median(continuum)
    std = np.std(continuum)
    continuum_mask = (continuum < med + s * std) & (continuum > med - s * std)

    median = np.median(continuum[continuum_mask])
    standard_dev = np.std(continuum[continuum_mask])

    mask = flux > median + s * standard_dev

    return mask

def equivalent_width(norm_flux, restwave, integration_lims = (5885, 5905)):

    ## get the indices defining the Na D restwave region
    integration_inds = np.where((restwave >= integration_lims[0]) & (restwave <= integration_lims[1]))[0]

    ## extract Na D values
    normflux_cut = norm_flux[integration_inds]
    restwave_cut = restwave[integration_inds]

    if len(normflux_cut) < 10:
        return -999
    
    ones = np.ones(len(normflux_cut))
    dLambda = np.gradient(restwave_cut)
    EW = np.sum(  (( ones - normflux_cut ) * dLambda)  )

    ew = EW if np.isfinite(EW) else -999

    return ew