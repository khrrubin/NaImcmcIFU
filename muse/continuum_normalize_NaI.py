
from __future__ import print_function

import math
import numpy as np
import scipy.optimize as op
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import pdb

# Continuum-normalize using a linear fit to the avg
# flux in the windows blim and rlim

def func_lin(x, a, b):
    return (a*x) + b

def norm(wave, flux, err, blim, rlim, FIT_FLG=None, smod=None):

    if((FIT_FLG==None) | (FIT_FLG==0)):
        return_smod = 0
    elif(FIT_FLG==1):
        return_smod = 1
    
    bind = np.where((wave > blim[0]) & (wave < blim[1]))
    rind = np.where((wave > rlim[0]) & (wave < rlim[1]))
    #     bind = np.where((wave > blim[0]) & (wave < blim[1]) & (flux.mask==False))
#     rind = np.where((wave > rlim[0]) & (wave < rlim[1]) & (flux.mask==False))

    if((len(bind[0])==0) | (len(rind[0])==0)):

        mskflg = 1
        continuum = 1.0 + 0.0*wave
        s2n = 0.0
        
    else:
      
        x = np.concatenate([wave[bind],wave[rind]])
#         y = np.ma.concatenate([flux[bind],flux[rind]])
#         sigy = np.ma.concatenate([err[bind],err[rind]])
        y = np.concatenate([flux[bind],flux[rind]])
        sigy = np.concatenate([err[bind],err[rind]])
    
        #z = np.polyfit(x,y,1)
        #cfit = np.poly1d(z)
        #continuum = cfit(wave)

        popt, pcov = curve_fit(func_lin, x, y, sigma=sigy, method='lm', absolute_sigma=True)
        mskflg = 0
        continuum = (popt[0] * wave) + popt[1]

        s2n = np.median(y/sigy)


    if(return_smod==0):
        nflux = flux/continuum

    elif(return_smod==1):
        nflux = smod/continuum
        
    nerr = err/continuum

    return {'nwave':wave, 'nflux':nflux, 'nerr':nerr,
            'cont':continuum, 's2n':s2n, 'mskflg':mskflg}



def smod_norm(wave, flux, err, smod, blim, rlim, emline_mask = True, s = 1):

    ## continuum outside of Na I for S/N
    bind = np.where((wave > blim[0]) & (wave < blim[1]))[0]
    rind = np.where((wave > rlim[0]) & (wave < rlim[1]))[0]

    if((len(bind[0])==0) | (len(rind[0])==0)):
        mskflg = 1
        s2n = 0.0
    else:
        y = np.ma.concatenate([flux[bind],flux[rind]])
        sigy = np.ma.concatenate([err[bind],err[rind]])
        s2n = np.median(y/sigy)
    
    ## normalize
    nflux = flux/smod
    nerr = err/smod
    mskflg = 0

    if emline_mask:
        continuum_inds = np.concatenate([bind, rind])
        continuum = flux[continuum_inds]
        med = np.median(continuum)
        std = np.std(continuum)
        continuum_mask = (continuum < med + s * std) & (continuum > med - s * std)

        median = np.median(continuum[continuum_mask])
        standard_dev = np.std(continuum[continuum_mask])

        mask = flux > median + s * standard_dev

        nflux.mask = mask
        nerr.mask = mask
        smod.mask = mask

    return {'nwave':wave, 'nflux':nflux, 'nerr':nerr,
            'cont':smod, 's2n':s2n, 'mskflg':mskflg}