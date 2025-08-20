import numpy as np

def emline_mask(flux, wave, cont_blim: tuple, cont_rlim: tuple, datamask = None, s = 1, testrun = False):
    if testrun:
        print("Masking emission lines")

    if datamask is None:
        datamask = np.zeros_like(flux).astype(bool)

    continuum_select = (wave > cont_blim[0]) & (wave < cont_blim[1]) | (wave > cont_rlim[0]) & (wave < cont_rlim[1])
    continuum = flux[continuum_select]
    continuum_mask = datamask[continuum_select]
    median = np.median(continuum[~continuum_mask])
    standard_dev = np.std(continuum[~continuum_mask])

    if testrun:
        print(f"Continuum level: {median:.3f}")

    mask = flux > median + s * standard_dev

    if testrun:
        print(f"Masking {np.sum(mask)} / {len(mask)} values")

    return mask

def equivalent_width(norm_flux, restwave, integration_lims = (5885, 5905), datamask = None, testrun = False):
    if datamask is None:
        datamask = np.zeros_like(norm_flux).astype(bool)

    flux_masked = norm_flux[~datamask]
    restwave_masked = restwave[~datamask]

    ## get the indices defining the Na D restwave region
    integration_select = (restwave_masked > integration_lims[0]) & (restwave_masked < integration_lims[1])

    ## extract Na D values
    normflux_cut = flux_masked[integration_select]
    restwave_cut = restwave_masked[integration_select]

    if len(normflux_cut) < 10:
        return -999
    
    ones = np.ones(len(normflux_cut))
    dLambda = np.gradient(restwave_cut)
    EW = np.sum(  (( ones - normflux_cut ) * dLambda)  )

    ew = EW if np.isfinite(EW) else -999

    if testrun:
        print(f"Equivalent Width = {ew:.3f} measured over {np.min(restwave_cut)} - {np.max(restwave_cut)}")

    return ew