# NaImcmcIFU Mg II
This branch of [NaImcmcIFU](https://github.com/khrrubin/NaImcmcIFU/) offers use of the single-component Mg II model code as described in Rubin et al. ([2014](https://iopscience.iop.org/article/10.1088/0004-637X/794/2/156)). 

Currently, this code is meant for single-spectra analysis; there is no code set up for Mg II analysis of IFU data (but feel free to add your own!).

**Note:** The instructions below assume that the eseential astronomy Python libraries (e.g. NumPy, Scipy, Matplotlib, Astropy) are already installed. 

## 1. Getting Started

#### 1.1 Required Packages:
* [emcee](https://emcee.readthedocs.io/en/stable/)
* [linetools](https://linetools.readthedocs.io/en/latest/)
* [corner](https://corner.readthedocs.io/en/latest/) <span style="color:red">(Now Optional)</span>


To install these required packages, type the following on the command line.

```
$ pip install -U emcee==3.1.4
```

```
$ git clone https://github.com/linetools/linetools.git
$ cd linetools
$ python setup.py develop
```

```
$ python -m pip install corner==2.2.2
```

#### 1.2 Installing NaImcmcIFU
You can download the repository using `git clone` and then switch to the `mod_mgii` branch by entering the following in the command line:
```
$ git clone https://github.com/khrrubin/NaImcmcIFU
$ git checkout mod_mgii
```
To verify you are in the correct branch, simply type `git branch`. The `*` indicates the current branch.

```
$ git branch
  master
* mod_mgii
```

#### 1.3 Getting Familiar

Once you are on the correct branch, you should now be able to cd into the `mgii` directory!

```
$ cd mgii
$ ls
README.md			    example_data			    lnlikelihood.py			model_MgII.py
example_usage.ipynb		mcmc_analysis_template.py	model_fitter.py
```

This contains *almost* all of the code necessary to model your own Mg II absorption profile(s). However, due to the nature of variance in data storage and user preferences of data structure, there is no 'universal' function provided handle a given spectrum.

The setup is extremely easy: The script `mcmc_analysis_template.py` gives a very basic template outline of how to call the MCMC model fitter and grab the resulting best-fit model parameters for a given spectrum. Additionally, the Jupyter notebook `example_usage.ipynb` shows basic usage with real spectral data stored in `mgii/example_data/`.

Please briefly read &sect; 2 before beginning!

## 2. Notes

#### 2.1 MCMC Setup

All of the values corresponding to the MCMC fitting are set as they were for our MUSE analysis. This includes

* Number of steps
* Burn-in steps
* Number of walkers
* Starting position
* Parameter priors

These values can be adjusted directly in `model_fitter.modeling.__init__`. The model parameter priors should be adjusted in `lnlikelihood.mcmc_priors` function.

#### 2.2 FWHM

The Mg II model requires the input of the spectrograph's line spread function (LSF) value in pixels at the observed wavelength of the Mg II doublet (as the parameter `fwhm` in `model_MgII.model_MgII()`). Our model code generates the flux at a resolution of $0.1 \mathrm{Å\ pix^{-1}}$ before smoothing the model spectrum by a Gaussian kernel of full width half max equal to LSF pixel resolution, and rebinning to the size of the observed wavelength array.

When calling the `modeling` class, `fwhm` is set up as an optional argument where, if `None`, it will automatically compute the FWHM using from the MUSE LSF configuration in `/example_data/`. **To correctly analyze your spectra, please calculate and input your own `fwhm`**.

&sect; 2.2.1 Describes the calculation of the pixel resolution I used as an example.

##### 2.2.1 Computing FWHM in Pixels

To compute the FWHM for Gaussian smoothing of the model, I used

$$
\mathrm{Resolution\ (pixels)} = R_{\mathrm{pix}} = \frac{\mathrm{LSF_{FWHM}}}{\mathrm{pixel\ scale}}
$$

where
* $\mathrm{LSF_{FWHM}}$ is the value of the line spread function of the instrument in Angstroms. For our MUSE spectra, I used the udf-10 line-spread-function from Bacon et al. 2017 found in `NaImcmcIFU/mgii/example_data/LSF-Config_MUSE_WFM`.
* $\mathrm{pixel\ scale}$ is the number of Angstroms per pixel in the spectrograph's wavelength range.

To calculate the pixel resolution for the Mg II doublet $(R_{pix})$, I first computed the *expected* observed wavelength red line of Mg II by 

$$
\lambda_{exp,\ \mathrm{Mg II}} = 2803.53 (1 + z)
$$ 

where $z$ is the redshift of the galaxy. 

I then found the $\mathrm{LSF_{FWHM}}$ of the closest matching observed wavelength to $\lambda_{exp,\ \mathrm{Mg II}}$ from the LSF configuration mentioned above to get the LSF in Angstroms at Mg II $(\mathrm{LSF_{FWHM,\ Mg\ II}})$.

To calculate the pixel scale at Mg II $(\mathrm{pixel\ scale\vert_{Mg\ II}})$, I took the array of observed wavelength, and computed the median difference in wavelength between each pixel in a $10\ \mathrm{Å}$ region, centered on $\lambda_{exp,\ \mathrm{Mg II}}$. That is,

$$
\mathrm{pixel\ scale\vert_{Mg\ II}} = \mathrm{med}\left[\Delta \lambda(\lambda \in [\lambda_l, \lambda_r]) \right]
$$

where $\lambda_l \approx \lambda_{exp,\ \mathrm{Mg II}} - 5$ and $\lambda_r \approx \lambda_{exp,\ \mathrm{Mg II}} + 5$.

Finally, I calculated $R_{pix}$ as

$$
\boxed{R_{pix} = \frac{\mathrm{LSF_{FWHM}}(\lambda \approx \lambda_{exp,\ \mathrm{Mg II}})}{\mathrm{pixel\ scale\vert_{Mg\ II}}}}
$$

This is implemented in the function `model_MgII.get_fwhm_MUSE_UDF`.