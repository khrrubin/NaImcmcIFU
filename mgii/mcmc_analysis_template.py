import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from model_fitter import modeling
import model_MgII

def get_data(arg):
    """
    Add your code to initialize necessary data!
    """
    pass

def main():
    """
    Basic template
    """
    ## your data
    wavelengths, flux, uncertainty, redshift  = get_data()

    ## initial (somewhat arbitrary) guess of physical parameters
    param_guess = (2803, 14, 90, 0.5)

    ## initialize the modeling class
    model = modeling(wavelengths, flux, uncertainty, redshift, param_guess)
    
    ## run emcee
    model.mcmc()

    ## best-fit parameters from the emcee analysis of the single spectra
    best_fit = tuple(param[0] for param in model.theta_percentiles)

if __name__ == "__main__":
    main()