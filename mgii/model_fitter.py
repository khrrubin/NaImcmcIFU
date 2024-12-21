from __future__ import print_function

import emcee
#import corner
import lnlikelihood
import model_MgII
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class modeling:

    def __init__(self, wavelength, flux, error, redshift, guesses, fwhm = None, linetimefil = None, printresult = False):
        """
        Initialize the model with spectral data and setup parameters for analysis.

        This constructor normalizes the input spectrum to a defined continuum, initializes key 
        physical parameters, and sets up configurations for Markov Chain Monte Carlo (MCMC) sampling.

        Parameters:
            wavelength (array-like): 
                Observed wavelength array of the spectrum in Angstroms.
            flux (array-like): 
                Observed flux array corresponding to the wavelength array.
            error (array-like): 
                Uncertainty array for the flux measurements.
            redshift (float): 
                Redshift of the observed source.
            guesses (tuple): 
                Initial guesses for model parameters in the order:
                (lambda_red_guess, logN_guess, bD_guess, Cf_guess), where:
                    - lambda_red_guess (float): Initial guess for the redshifted wavelength.
                    - logN_guess (float): Initial guess for column density (log scale).
                    - bD_guess (float): Initial guess for Doppler parameter.
                    - Cf_guess (float): Initial guess for covering fraction.
            fwhm (float, optional):
                Full-width at half maximum (FWHM) of the spectral resolution. 
                NOTE:: Not optional if your spectrum does NOT use the MUSE UDF-10 LSF config.
            linetimefil (str, optional): 
                Full filepath to save MCMC chain time plots. Defaults to None.
            printresult (bool, optional): 
                Whether to print MCMC fitting results to the console. Defaults to False.

        Attributes:
            wave (array-like): 
                Normalized wavelength array in the rest frame.
            flux (array-like): 
                Normalized flux array.
            err (array-like): 
                Normalized uncertainty array.
            z (float): 
                Redshift of the source.
            fwhm (float): 
                Full-width at half maximum (FWHM) of the spectral resolution.
            sampndim (int): 
                Number of dimensions for the MCMC sampler (default: 4).
            sampnwalk (int): 
                Number of walkers for the MCMC sampler (default: 100).
            nsteps (int): 
                Total number of MCMC steps per walker (default: 600).
            burnin (int): 
                Number of initial MCMC steps to discard as burn-in (default: 500).
            linetimefil (str or None): 
                Path to save MCMC chain time plots, if specified.
            printresult (bool): 
                Flag to print the MCMC fitting results.
            theta_guess (list): 
                Initial guesses for model parameters as [lambda_red_guess, logN_guess, bD_guess, Cf_guess].

        Notes:
            - The input spectrum is normalized using the `model_MgII.continuum_normalize` function.
            - Spectral resolution (FWHM) is estimated using `model_MgII.get_fwhm_MUSE_UDF`.
        """
        restwave = wavelength / (1+redshift)
        normdict = model_MgII.continuum_normalize(restwave, flux, error)

        ###### initialize class attributes
        ## physical parameters
        lamred_guess, logN_guess, bD_guess, Cf_guess = guesses
        self.wave = normdict['wavelength']
        self.flux = normdict['normflux']
        self.err = normdict['normerr']
        self.z = redshift
        if fwhm is None:
            self.fwhm = model_MgII.get_fwhm_MUSE_UDF(wavelength, redshift)
        else:
            self.fwhm = fwhm
        
        ## MCMC setup (modify for needs)
        self.sampndim = 4
        self.sampnwalk = 100
        self.nsteps = 600
        self.burnin = 500

        ## optional for diagnostics
        self.linetimefil = linetimefil
        self.printresult = printresult

        self.theta_guess = [lamred_guess, logN_guess, bD_guess, Cf_guess]
    

    def maxlikelihood(self):
        """
        Calculate the maximum likelihood model using a chi-squared minimization approach.

        This function computes the model parameters that maximize the likelihood by minimizing 
        the chi-squared value. The likelihood is based on the assumption that the probability is 
        proportional to the exponential of the negative half of the chi-squared value (P = exp(-0.5 * chi^2)).

        For more information on Markov Chain Monte Carlo (MCMC) methods used here, see the `emcee` documentation:
        https://emcee.readthedocs.io/en/stable/

        The function minimizes the negative log-likelihood to determine the best-fitting parameters
        for the model. It then stores the resulting model parameters in `self.theta_ml`.
        """      
        priors = lnlikelihood.mcmc_priors()

        # Create the boundaries tuple for minimize
        boundaries = tuple(
            (priors['lower_lims'][param], priors['upper_lims'][param]) 
            for param in ['lambda', 'logN', 'bD', 'Cf']
            )
        
        chi2 = lambda *args: -2 * lnlikelihood.lnlike(*args)       
        result = op.minimize(chi2, self.theta_guess, args=(self.wave, self.flux, self.err, self.fwhm), 
                             bounds=boundaries)

        self.theta_ml = result.x


    def mcmc(self):

        """
        Run a Markov Chain Monte Carlo (MCMC) sampling to estimate model parameters.

        This function sets up and runs an MCMC sampler using the `emcee` library to explore the 
        posterior probability distribution of the model parameters. It also generates diagnostic 
        plots of the sampling chains for inspection.

        Workflow:
        1. Perform maximum likelihood estimation (`self.maxlikelihood`) to initialize the MCMC starting positions.
        2. Set up an ensemble sampler with a given number of walkers and dimensions.
        3. Run the MCMC chain to sample the posterior.
        4. Optionally save diagnostic plots showing the parameter evolution across steps.
        5. Apply a burn-in phase and store the final samples.
        6. Compute parameter percentiles (16th, 50th, 84th) to summarize the posterior distributions.

        Parameters (class attributes used):
            - `self.sampndim` (int): Number of dimensions (free parameters) in the model.
            - `self.sampnwalk` (int): Number of walkers for the MCMC sampler.
            - `self.nsteps` (int): Number of steps for each walker.
            - `self.burnin` (int): Number of initial steps to discard as burn-in.
            - `self.linetimefil` (str, optional): Filepath to save chain diagnostic plots (if provided).
            - `self.printresult` (bool): Whether to print the final parameter estimates to the console.

        Returns:
            None: Updates the following class attributes:
                - `self.samples`: Array of MCMC samples after burn-in.
                - `self.theta_percentiles`: List of tuples containing median parameter values and their uncertainties.

        Outputs (if `self.printresult` is True):
            - Prints the parameter estimates and their uncertainties in the format:
            `par i = median +upper_error -lower_error`

        Diagnostic Plots:
            - If `self.linetimefil` is specified, saves a plot showing the parameter chains 
            over time for each parameter.

        Notes:
            - The initial walker positions are sampled from a Gaussian distribution centered 
            on the maximum likelihood estimates.
            - The posterior distribution is evaluated using the `lnlikelihood.lnprob` function.
            - Quantiles are calculated to represent the central 68% confidence interval.

        References:
            - `emcee` documentation: https://emcee.readthedocs.io/en/stable/
        """

        self.maxlikelihood()
        ndim = self.sampndim
        nwalkers = self.sampnwalk
        pos = [self.theta_ml + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood.lnprob,
                                        args=(self.wave, self.flux, self.err, self.fwhm))

        # Clear and run the production chain.        
        sampler.run_mcmc(pos, self.nsteps, rstate0=np.random.get_state(), progress=True)

        # save time chain plots of params if linetimefil is not None
        if(self.linetimefil != None):
            
            fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(5,6))
            label_list = [r'$\lambda_{red}$', 'log N', '$b_D$', '$C_f$']

            for ind in range(0,ndim):
                axes[ind].plot(sampler.chain[:, :, ind].T, color="k", alpha=0.4)
                axes[ind].yaxis.set_major_locator(MaxNLocator(5))
                axes[ind].axhline(self.theta_ml[ind], color="#888888", lw=2)
                axes[ind].set_ylabel(label_list[ind])
                


            fig.tight_layout(h_pad=0.0)
            fig.savefig(self.linetimefil, dpi=200)
            plt.close(fig)


        burnin = self.burnin
        samples_burnin = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        self.samples = np.transpose(sampler.get_chain(),axes=(1,0,2))

        # Compute the quantiles.
        theta_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_burnin, [16, 50, 84], axis=0))))
        self.theta_percentiles = theta_mcmc            
            
        if self.printresult:
            print("""MCMC result:""")
            for ind in range(0, ndim):
                print(""" par {0} = {1[0]} +{1[1]} -{1[2]}""".format(ind, theta_mcmc[ind]))
        