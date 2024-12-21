# Define likelihood and priors
import numpy as np
import model_MgII

def mcmc_priors():
        """
        Initialize and return the priors for the model parameters.

        This function defines the lower and upper bounds for the model parameters used in the
        Markov Chain Monte Carlo (MCMC) fitting.

        The function returns a dictionary containing the priors for the following parameters:
        - Velocity (vlim) in Angstroms, defined by the blueshift and redshift limits
        - Column density (logN), in the range [10.0, 18.0] (log scale)
        - Doppler parameter (bD), in the range [10.0, 200.0] (km/s)
        - Covering fraction (Cf), in the range [0.0, 1.0]

        Returns:
            dict: A dictionary containing the lower and upper bounds for each model parameter:
                - 'lower_lims': Lower bounds for the parameters
                - 'upper_lims': Upper bounds for the parameters
        """

        sol = 2.998e5    # speed of light km/s
        restwave = model_MgII.transitions()['lamred0'] # restwave of doublet red line center
        vlim = 400 # velocity limit km/s
        
        # modify for needs
        priors_dict = {
            'lower_lims':{
                'lambda': -1.0 * (vlim * restwave / sol) + restwave, # Angstrom blueshift limit
                'logN': 10.0, # log column density lower
                'bD': 10.0, # doppler paremeter lower
                'Cf': 0.0 # covering fraction lower
            },
            'upper_lims':{
                'lambda': (vlim * restwave / sol) + restwave, # Angstrom redshift limit
                'logN': 18.0, # log column density upper
                'bD': 200.0, # doppler parameter upper
                'Cf': 1.0 # covering fraction upper
            }
        }

        return priors_dict


# Define the probability function as likelihood * prior.
def lnprior(theta):
    """
    Evaluate the natural logarithm of the prior probability for the model parameters.
    
    This function checks whether the given model parameters fall within predefined physical 
    and observational bounds (priors). If all parameters are within their respective limits, 
    a logarithmic prior probability of 0.0 is returned (indicating a uniform prior within bounds). 
    If any parameter lies outside its bounds, it returns negative infinity (-np.inf), 
    indicating zero probability.
    
    Parameters:
        theta (tuple): A tuple of model parameters in the order:
            - lamred (float): Redshifted wavelength parameter (Angstroms).
            - logN (float): Logarithm of the column density.
            - bD (float): Doppler broadening parameter.
            - Cf (float): Covering fraction.
    
    Returns:
        float: 
            - 0.0 if all parameters lie within their respective bounds.
            - -np.inf if any parameter lies outside its bounds.
    
    Notes:
        - The parameter bounds are retrieved from the `mcmc_priors()` function.
        - This function assumes uniform priors within the specified parameter ranges.
    """

    lamred, logN, bD, Cf = theta
    priors = mcmc_priors()

    if (priors['lower_lims']['lambda'] < lamred < priors['upper_lims']['lambda'] and
    priors['lower_lims']['logN'] < logN < priors['upper_lims']['logN'] and
    priors['lower_lims']['bD'] < bD < priors['upper_lims']['bD'] and
    priors['lower_lims']['Cf'] < Cf < priors['upper_lims']['Cf']):
        return 0.0
    
    return -np.inf


def lnlike(theta, wave, flux, err, fwhm):
    """
    Compute the natural logarithm of the likelihood for the model parameters.
    
    This function evaluates how well a model, defined by the given parameters, 
    matches the observed data. It calculates the log-likelihood assuming Gaussian 
    uncertainties on the flux measurements.
    
    Parameters:
        theta (tuple): Model parameters in the order:
            - lamred (float): Redshifted wavelength parameter (Angstroms).
            - logN (float): Logarithm of the column density.
            - bD (float): Doppler broadening parameter.
            - Cf (float): Covering fraction.
        wave (array-like): Wavelength array (Angstroms).
        flux (array-like): Observed flux values.
        err (array-like): Uncertainties in the observed flux values.
        fwhm (float): Full width at half maximum (FWHM) of the instrument profile.
    
    Returns:
        float: The natural logarithm of the likelihood, where higher values indicate 
               a better match between the model and the observed data.
    """
    model_dict = model_MgII.model_MgII(theta,fwhm,wave)
    
    ## correct for model droppoff in value at boundaries
    inds = np.where(model_dict['modflx']>=0.5)[0]
    w = slice(inds.min(),inds.max())
    
    return -0.5 * np.sum(((flux[w] - model_dict['modflx'][w])/err[w])**2 - np.log(2 * np.pi / (err[w])**2))



def lnprob(theta, wave, flux, err, fwhm):
    """
    Compute the natural logarithm of the posterior probability for the model parameters.
    
    This function combines the log-prior (`lnprior`) and the log-likelihood (`lnlike`) 
    to compute the total log-posterior probability, which is used in Bayesian inference.
    
    Parameters:
        theta (tuple): Model parameters in the order:
            - lamred (float): Redshifted wavelength parameter (Angstroms).
            - logN (float): Logarithm of the column density.
            - bD (float): Doppler broadening parameter.
            - Cf (float): Covering fraction.
        wave (array-like): Wavelength array (Angstroms).
        flux (array-like): Observed flux values.
        err (array-like): Uncertainties in the observed flux values.
        fwhm (float): Full width at half maximum (FWHM) of the instrument profile.
    
    Returns:
        float: The natural logarithm of the posterior probability. 
               Returns `-np.inf` if the parameters are outside the prior bounds.
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, flux, err, fwhm)