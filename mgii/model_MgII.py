import numpy as np
import scipy.special as sp
from linetools.spectra.xspectrum1d import XSpectrum1D
from astropy import units as u
import matplotlib.pyplot as plt

def transitions():
    # Set up constants for MgII
    # values from Cashman 2017
    lamblu0 = 2796.352
    lamred0 = 2803.53

    fblu0 = 0.613
    fred0 = 0.306
    
    lamfblu0 = lamblu0 * fblu0
    lamfred0 = lamred0 * fred0
    
    return {'lamblu0':lamblu0, 'lamred0':lamred0, 'lamfblu0':lamfblu0, 'lamfred0':lamfred0}


# Set up model line profile
# theta contains lamred, logN, bD, Cf (in that order)
def model_MgII(theta,fwhm,newwv):
    """
    Generate a model MgII absorption profile using the single-component physical model described
    by Rubin et al. 2014

    Computes the model absorption profile based on the input parameters, theta, smooths it by the
    FWHM Gaussian, then rebins it to match observational data.

    Parameters
    ----------
    theta : tuple
        A tuple containing four parameters: 
        - lamred (float): central wavelength of the red (2803) line of the doublet
        - logN (float): log column density
        - bD (float): Doppler parameter (velocity width)
        - Cf (float): covering fraction
    fwhm : numpy.ndarray or array-like
        LSF of the instrument in pixels corresponding ~ to the observed Mg II doublet wavelength;
        used for Gaussian smoothing of the model spectrum.
    newwv : array-like
        Wavelength array (in Angstroms) representing the observational data for rebinning.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'modwv' (array): The rebinned wavelength array (in Angstroms).
        - 'modflx' (array): The rebinned model flux array corresponding to the modeled MgII absorption profile.
    """

    ## First, get info on transitions
    sol = 2.998e5    # speed of light km/s
    transinfo = transitions() # transition wavelength and quantum oscillator values
    velratio = 1.0 + (transinfo['lamblu0'] - transinfo['lamred0'])/transinfo['lamred0'] # constant velocity ratio of the MgII doublet
    dmwv = 0.1   # in Angstroms
    
    ## feature wavelength, column density, doppler param/velocity width, covering fraction
    lamred, logN, bD, Cf = theta 
    
    N = 10.0**logN # convert from log column density
    lamblu = lamred * velratio # central wavelength of the blue (2796) line
    
    # central optical depths of each line
    taured0 = N * 1.497e-15 * transinfo['lamfred0'] / bD
    taublu0 = N * 1.497e-15 * transinfo['lamfblu0'] / bD

    # set up the model wavelength with resolution of .1 angstrom (dmwv)
    wv_unit = u.AA
    modwave = np.arange(int(newwv.min()),int(newwv.max()),dmwv)
    modwave_u = u.Quantity(modwave,unit=wv_unit)
    
    # optical depth as a function of wavelength for each line
    exp_red = -1.0 * (modwave - lamred)**2 / (lamred * bD / sol)**2
    exp_blu = -1.0 * (modwave - lamblu)**2 / (lamblu * bD / sol)**2
    taured = taured0 * np.exp(exp_red)
    taublu = taublu0 * np.exp(exp_blu)

    ## Unsmoothed model profile
    mod_MgII = 1.0 - Cf + (Cf * np.exp(-1.0*(taublu + taured)))

    # initialize model as a linetools.spectra.xspectrum1d.XSpectrum1D
    xspec = XSpectrum1D.from_tuple((modwave,mod_MgII))
    
    ## Now smooth with a Gaussian resolution element
    smxspec = xspec.gauss_smooth(fwhm)
    
    ## Now rebin to match pixel size of observations
    ## Can try XSpectrum1D.rebin, need to input observed wavelength array
    uwave = u.Quantity(newwv,unit=wv_unit)

    # Rebinned spectrum
    rbsmxspec = smxspec.rebin(uwave)
    
    # remove astropy units
    modwv = rbsmxspec.wavelength.value
    modflx = rbsmxspec.flux.value
    
    return {'modwv':modwv, 'modflx':modflx}


## normalize to the continuum around doublet
## returns the region of the spectrum
def continuum_normalize(waves, flux, error):    
    """
    Normalize the flux of a spectrum to the continuum around the MgIIdoublet region.

    The function selects two regions (blue and red) around the doublet for continuum fitting, 
    performs a linear fit to the continuum, and normalizes the flux and error based on the fitted continuum.

    Parameters
    ----------
    waves : numpy.ndarray
        The wavelength array of the spectrum.

    flux : numpy.ndarray
        The flux array of the spectrum.

    error : numpy.ndarray
        The error array corresponding to the flux values.

    Returns
    -------
    dict
        A dictionary containing:
        - 'wavelength' : numpy.ndarray
            The wavelengths of the region around the doublet.
        - 'normflux' : numpy.ndarray
            The normalized flux values.
        - 'normerr' : numpy.ndarray
            The normalized error values.
    """

    # Define continuum regions to normalize to
    continuum_boundaries = [(2765, 2780), (2810, 2825)]  # Wavelength boundaries for the continuum (Angstrom) blueward and redward of Mg II
    
    # Helper function to find indices in the continuum regions
    def get_indices(region, waves, flux):
        return np.where((waves > region[0]) & (waves < region[1]) & (flux != 0))

    # Combine indices from both regions for fitting
    continuum_indices = np.concatenate([get_indices(boundaries, waves, flux)[0] for boundaries in continuum_boundaries])

    # slice flux and wavelength by the continuum indices and
    continuum_flx = flux[continuum_indices]
    continuum_wav = waves[continuum_indices]

    # mask to only include flux values within 1 std of the median in the fit
    w = abs(continuum_flx - np.median(continuum_flx)) < 1 * np.std(continuum_flx)

    # linear model params
    p = np.polyfit(continuum_wav[w], continuum_flx[w], deg=1)


    # Slice the flux and wavelength by the outer boundaries but inlcuding the doublet
    mgii_inds = np.where((waves >= 2765) & (waves <= 2825))
    wav = waves[mgii_inds]
    flx = flux[mgii_inds]
    err = error[mgii_inds]
    
    # generate the linear model continuum
    model_continuum = np.polyval(p,wav)

    # normalize flux and error
    normflx = flx/model_continuum
    normerr = err/model_continuum

    return {'wavelength':wav, 'normflux':normflx, 'normerr':normerr}



##gets the fwhm in pixels for gaussian smoothing
def get_fwhm_MUSE_UDF(wavelength, redshift):
    """
    NOTE:: This was used for MUSE spectra of the MUSE Ultra Deep Field (MUDF).

    Calculates the Pixel Resolution—Full Width at Half Maximum (FWHM) in pixels—for Gaussian smoothing 
    from the MUSE UDF-10 line spread function (LSF) (Bacon et al. 2017).

    Parameters
    ----------
    wavelength : array-like
        The wavelength array (in Angstroms) of the observed spectrum.
    redshift : float
        The redshift of the galaxy to calculate the 'observed' MgII wavelength position
    Returns
    -------
    float
        The FWHM in pixels at the corresponding observed wavelength of the MgII doublet (2803 line)
    """
    # MUSE udf10 line spread function (LSF) in Angstrom from Bacon et al. 2017 Eq 8
    specres = np.genfromtxt('example_data/LSF-Config_MUSE_WFM', names=('wave', 'fwhm'), comments='#')

    # Get the `expected` observed wavelength position of MgII
    transinfo = transitions() # grab MgII info
    lamred = transinfo['lamred0'] * (1+redshift) # Expected observed position of the red abs-line

    # find index the closest corresponding observed wavelength of expected Mg II
    ind = np.argmin(abs(wavelength - lamred)) # index of the closest matching observed wavelength

    # define a 10 angstrom region centered on Mg II and calculate the median pixel scale
    mgii_region = wavelength[ind-5:ind+5]
    wavperpix = np.median(np.diff(mgii_region)) # median resolution (Angstrom / pixel)

    # Get the LSF in Angstrom of the observed Mg II
    i = np.argmin(abs(lamred-specres['wave'])) # find the index of LSF wavelength that corresponds to expected Mg II
    res = specres['fwhm'][i] # LSF in Angstrom

    # Calculate pixel resolution
    fwhm = res / wavperpix # Resolution (FWHM) [pix] = LSF [Angstrom] / pixel scale [Angstrom / pix]
    
    return fwhm