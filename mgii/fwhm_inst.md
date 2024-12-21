
# Computing FWHM in Pixels

To compute the FWHM for Gaussian smoothing of the model, I used

$$
\mathrm{Resolution\ (pixels)} = R_{\mathrm{pix}} = \frac{\mathrm{LSF_{FWHM}}}{\mathrm{pixel\ scale}}
$$

where
* $\mathrm{LSF_{FWHM}}$ is the value of the line spread function of the instrument in Angstroms. For our MUSE spectra, I used the udf-10 line-spread-function from Bacon et al. 2017 found in `NaImcmcIFU/mgii/LSF-Config_MUSE_WFM`.
* $\mathrm{pixel\ scale}$ is the number of Angstroms per pixel in the spectrograph's wavelength range.

To calculate the pixel resolution for the Mg II doublet $(R_{pix})$, I first computed the *expected* observed wavelength red line of Mg II by 

$$
\lambda_{exp,\ \mathrm{Mg II}} = 2803.53 (1 + z)
$$ 

where $z$ is the redshift of the galaxy. 

I then found the $\mathrm{LSF_{FWHM}}$ of the closest matching observed wavelength to $\lambda_{exp,\ \mathrm{Mg II}}$ from the LSF configuration mentioned above to get the LSF in Angstroms at Mg II $(\mathrm{LSF_{FWHM,\ Mg\ II}})$.

To calculate the pixel scale at Mg II $(\mathrm{pixel\ scale|_{Mg\ II}})$, I took the array of observed wavelength, and computed the median difference in wavelength between each pixel in a $10\ \mathrm{\AA}$ region, centered on $\lambda_{exp,\ \mathrm{Mg II}}$. That is,

$$
\mathrm{pixel\ scale|_{Mg\ II}} = \mathrm{med}\left[\Delta \lambda(\lambda \in [\lambda_l, \lambda_r]) \right]
$$

where $\lambda_l \approx \lambda_{exp,\ \mathrm{Mg II}} - 5$ and $\lambda_r \approx \lambda_{exp,\ \mathrm{Mg II}} + 5$.

Finally, I calculated $R_{pix}$ as

$$
\boxed{R_{pix} = \frac{\mathrm{LSF_{FWHM}}(\lambda \approx \lambda_{exp,\ \mathrm{Mg II}})}{\mathrm{pixel\ scale|_{Mg\ II}}}}
$$


<br>
<br>

# Code Interpretation:

```python
def get_fwhm(wavelength, redshift, specres):

    # find the observed wavelength position of MgII
    transinfo = transitions() # grab MgII info
    lamred = transinfo['lamred0'] * (1+redshift) # observed position of the red abs-line
    ind = np.argmin(abs(wavelength - lamred)) # index of the closest matching observed wavelength
    region = wavelength[ind-5:ind+5] # define a region
    
    # find the corresponding instrument LSF in Angstrom
    wavperpix = np.median(np.diff(region)) # median resolution in wavelength / pix
    i = np.argmin(abs(lamred-specres['wave']))
    res = specres['fwhm'][i] # resolution in wavelength


    fwhm = res/wavperpix # resolution in pix
    
    return fwhm
```
