# Define likelihood and priors
import numpy as np
import scipy.special as sp
import math
import model_NaI

# Define the probability function as likelihood * prior.
def lnprior(theta):

    lamred, logN, bD, Cf = theta

    sol = 2.998e5    # km/s
    transinfo = model_NaI.transitions()

    vlim = 450 # km/s
    #vlim = 700.0     # km/s
    lamlim1 = -1.0 * (vlim * transinfo['lamred0'] / sol) + transinfo['lamred0']
    lamlim2 = (vlim * transinfo['lamred0'] / sol) + transinfo['lamred0']

    #logNlim1 = 10.0
    #logNlim2 = 15.0
    #logNlim1 = 14.0
    #logNlim1 = 13.5

    # 1st run of priors
    # logNlim1 = 11.4
    # logNlim2 = 16.0

    # 2nd run of priors
    # logNlim1 = 14.0
    # logNlim2 = 16.5

    # # 3rd run of priors
    # logNlim1 = 12.0
    # logNlim2 = 20.0

    # 4th run of priors
    logNlim1 = 12.0
    logNlim2 = 16.5

    bDlim1 = 2.0
    #bDlim2 = 200.0
    bDlim2 = 100
    #bDlim2 = 150

    Cflim1 = 0.0
    Cflim2 = 1.0
    
    #if -5.0 < m < 5.0 and -10.0 < b < 10.0 and -10.0 < lnf < 10.0:
    if lamlim1 < lamred < lamlim2 and logNlim1 < logN < logNlim2 and bDlim1 < bD < bDlim2 and Cflim1 < Cf < Cflim2:
        return 0.0
    return -np.inf


def lnlike(theta, wave, flux, err, velres):
    
    lamred, logN, bD, Cf = theta

    model = model_NaI.model_NaI(theta,velres,wave)
    flx_model = model['modflx']

    inv_sigma2 = 1.0/(err**2)
    #model = m * x + b
    #inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    # return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    return -0.5*(np.sum((flux-flx_model)**2*inv_sigma2 - np.log(2.0*math.pi*inv_sigma2)))



def lnprob(theta, wave, flux, err, velres):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, flux, err, velres)


# Debugging
#def isigsq(theta, x, y, yerr):
#    m, b, lnf = theta
#    model = m * x + b
#    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
#    term1 = (y-model)**2*inv_sigma2
#    term2 = np.log(2.0*math.pi*inv_sigma2)
#    return term1-term2

# More debugging
#def term(theta, x, y, yerr):
#    m, b, lnf = theta
#    model = m * x + b
#    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
#    term1 = (y-model)**2*inv_sigma2
#    term2 = np.log(2.0*math.pi*inv_sigma2)
#    return (y-model)**2
