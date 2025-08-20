import numpy as np
import os
import glob
from astropy.io import fits
import model_NaI
import model_fitter
import continuum_normalize_NaI
import continuum_analyses
from linetools.spectra.xspectrum1d import XSpectrum1D
from mangadap.config import defaults
from mangadap.util.parser import DefaultConfig
import json
import argparse

def get_data_path():
    config_filepath = 'config.json'
    with open(config_filepath) as config_file:
        config = json.load(config_file)
    
    for key in config.keys():
        data_path = config[key]

        if os.path.exists(data_path):
            return data_path
        else:
            continue

    raise ValueError(f"""Path-to-data does not exist. Please setup the data_path configuration in {config_filepath}
                            data_path should specify the absolute path to the location of the subdirectories containing
                            the muse_cubes, dap_outputs, and mcmc_outputs.
                            """)

def arguments():
    parser = argparse.ArgumentParser(description="Test Run of MCMC on a single bin")

    parser.add_argument('galname', type=str, help="Input galaxy name (NGC4030)")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method (default: SQUARE0.6)")
    parser.add_argument('binID', type=int, help="Specific bin to test fit")
    parser.add_argument('-p', '--plot', help = "Plot figures for inspection (default: False)", action='store_true', default=False)
    parser.add_argument('-q','--quiet', help = "Suppress verbose outputs (default: False)", action='store_true', default = False)
    return parser.parse_args()

def print_results(bin_number, percentiles, velocity, equivw):
    print(f"_____RESULTS FOR BIN {bin_number}_____")
    print(f"Centroid Velocity: {velocity:.3f} km / s")
    print(f"Lambda = {percentiles[0,0]:.3f} + {percentiles[0,2]:.3f} - {percentiles[0,1]:.3f} Å")
    print(f"log N = {percentiles[1,0]:.3f} + {percentiles[1,2]:.3f} - {percentiles[1,1]:.3f} / cm^2")
    print(f"b_D = {percentiles[2,0]:.3f} + {percentiles[2,2]:.3f} - {percentiles[2,1]:.3f} km / s")
    print(f"C_f = {percentiles[3,0]:.3f} + {percentiles[3,2]:.3f} - {percentiles[3,1]:.3f}")
    print(f"W_eq = {equivw:.3f} Å")

def main(galname, bin_key, binID, plot = False, quiet = False):
    blim = [5850.0, 5870.0]
    rlim = [5910.0, 5930.0]
    fitlim = [5880.0, 5910.0]
    c = 2.998e5

    data_root_dir = get_data_path()
    main_cube_dir = os.path.join(data_root_dir, 'muse_cubes')

    cube_dir = os.path.join(main_cube_dir, galname)
    if not os.path.isdir(cube_dir):
        raise ValueError(f'{cube_dir} is not a directory within /MUSE_cubes')
    if len(glob.glob(f"{cube_dir}/*.ini")) > 1:
        raise ValueError(f'Multiple .ini files within {cube_dir}. {cube_dir} directory must only have '
                         f'configuration file.')
    
    config_fil = glob.glob(f"{cube_dir}/*.ini")[0]
    if not os.path.isfile(config_fil):
        raise ValueError(f'{os.path.basename(config_fil)} does not exist within {cube_dir}')
    cfg = DefaultConfig(config_fil, interpolate=True)
    plate = cfg.getint('plate', default=None)
    ifu = cfg.getint('ifu', default=None)
    redshift = cfg.getfloat('z', default = None)
    if redshift is None:
        raise ValueError(f"No redshift found in {config_fil}")
    
    LSF_fil = os.path.join(main_cube_dir, 'LSF-Config_MUSE_WFM')
    if not os.path.isfile(LSF_fil):
        raise ValueError(f'LSF-Config_MUSE_WFM does not exist within {main_cube_dir}')
    configLSF = np.genfromtxt(LSF_fil, comments='#')
    configLSF_wv_air = configLSF[:, 0]
    configLSF_res = configLSF[:, 1]
    xspec = XSpectrum1D.from_tuple((configLSF_wv_air, 0.0 * configLSF_wv_air))
    xspec.meta['airvac'] = 'air'
    xspec.airtovac()
    configLSF_wv_vac = xspec.wavelength.value
    configLSF_restwv = configLSF_wv_vac / (1.0 + redshift)
    whLSF = np.where((configLSF_restwv > fitlim[0]) & (configLSF_restwv < fitlim[1]))
    median_LSFAng = np.median(configLSF_res[whLSF[0]])
    LSFvel = c * median_LSFAng / np.median(configLSF_wv_vac[whLSF[0]])

    output_root_dir = os.path.join(data_root_dir, 'dap_outputs')
    output_gal_dir = os.path.join(output_root_dir, f"{galname}-{bin_key}")
    if not os.path.isdir(output_gal_dir):
        raise ValueError(f'{output_gal_dir} is not a directory within {output_root_dir}.')
    
    beta_dirname = 'BETA-CORR'
    output_gal_sub_dir = os.path.join(output_gal_dir, beta_dirname)

    analysisplan_methods = 'MILESHC-MASTARSSP-NOISM'
    output_cube_dir = os.path.join(output_gal_sub_dir, f"{bin_key}-{analysisplan_methods}", str(plate), str(ifu))

    cube_file_path = os.path.join(output_cube_dir,
                                  f"manga-{plate}-{ifu}-LOGCUBE-{bin_key}-{analysisplan_methods}.fits")
    maps_file_path = os.path.join(output_cube_dir,
                                  f"manga-{plate}-{ifu}-MAPS-{bin_key}-{analysisplan_methods}.fits")


    hdu_map = fits.open(maps_file_path)
    binid_map = hdu_map['BINID'].data[0]
    ppxf_v_map = hdu_map['STELLAR_VEL'].data

    hdu_cube = fits.open(cube_file_path)
    spec = hdu_cube['FLUX'].data
    ivar = hdu_cube['IVAR'].data
    espec = np.sqrt(1 / ivar)
    mod = hdu_cube['MODEL'].data
    obswave = hdu_cube['WAVE'].data


    ind = binid_map == binID
    ny, nx = np.where(ind)
    y, x = ny[0], nx[0]

    binvel = ppxf_v_map[y, x]
    flux_bin = spec[:, y, x]
    err_bin = espec[:, y, x]
    mod_bin = mod[:, y, x]

    bin_z = redshift + ((1 + redshift) * (binvel / c))
    restwave = obswave / (1.0 + bin_z)

    nflux = flux_bin / mod_bin
    nerr = err_bin / mod_bin

    sres_NaI = LSFvel

    infinite_mask = ~np.isfinite(nflux) | ~np.isfinite(nerr)

    print("""Beginning fit for bin {0} """.format(binID))
    
    emission_mask = continuum_analyses.emline_mask(nflux, restwave, tuple(blim), tuple(rlim), datamask=infinite_mask, s=1 testrun=True)
    combined_mask = np.logical_or(infinite_mask, emission_mask)
    equiv_w = continuum_analyses.equivalent_width(nflux, restwave)

    if equiv_w <= 0:
        bin_number = binid_map[ind][0]
        samples = np.zeros((100, 1100, 4))
        percentiles = np.zeros((4,3))
        bin_velocity = -999
        print(f"ERROR: Equivalent width returned {equiv_w}")
        print_results(bin_number, samples, percentiles, bin_velocity)
        return


    # Cut out NaI
    select = np.where((restwave > fitlim[0]) & (restwave < fitlim[1]))
    nflux_nai = nflux[select]
    nerr_nai = nerr[select]
    restwave_nai = restwave[select]
    mask_nai = combined_mask[select]

    # check for bad data being masked
    if np.sum(mask_nai) == len(nflux_nai):
        bin_number = binid_map[ind][0]
        samples = np.zeros((100, 1100, 4))
        percentiles = np.zeros((4,3))
        bin_velocity = -999
        print(f"ERROR: All flux pixels are masked")
        print_results(bin_number, samples, percentiles, bin_velocity)
        return
    
    data = {'wave': np.ma.array(data = restwave_nai, mask = mask_nai), 'flux': np.ma.array(data = nflux_nai, mask = mask_nai), 
            'err': np.ma.array(data = nerr_nai, mask = mask_nai), 'velres': np.ma.array(data = sres_NaI, mask = mask_nai)}

    # Guess good model parameters
    lamred = 5897.5581
    logN = 14.5
    bD = 20.0
    Cf = 0.5
    theta_guess = lamred, logN, bD, Cf
    guess_mod = model_NaI.model_NaI(theta_guess, data['velres'], data['wave'])
    datfit = model_fitter.model_fitter(data, theta_guess)
    # Run the MCMC
    datfit.mcmc()

    # get gas velocity from model lambda and rest lambda
    lamred_mcmc, logN_mcmc, bD_mcmc, Cf_mcmc = datfit.theta_percentiles
    lamrest = 5897.5581
    velocity = ((lamred_mcmc[0] / lamrest) - 1) * c

    bin_number = binid_map[ind][0]
    samples = datfit.samples
    percentiles = datfit.theta_percentiles
    bin_velocity = velocity

    print_results(bin_number, samples, percentiles, bin_velocity)

    if plot:
        import matplotlib.pyplot as plt
        bf_mod = model_NaI.model_NaI((lamred_mcmc[0], logN_mcmc[0], bD_mcmc[0], Cf_mcmc[0]), data['velres'], restwave_nai.data)
        fig, ax = plt.subplots(1,1)
        mask = nflux_nai.mask

        ax.plot(restwave_nai.data, nflux_nai.data, 'k', drawstyle = 'steps-mid', linewidth=2)
        ax.plot(bf_mod['modwv'], bf_mod['modflx'], 'b')
        ax.scatter(restwave_nai.data[mask], nflux_nai.data[mask], s=5, marker='x', c='r')
        ax.set_xlabel(r'Wavelength $(\mathrm{\AA})$')
        ax.set_ylabel('Normalized Flux')
        ax.set_xlim(5880, 5910)
        plt.savefig('MCMCtestfit.pdf',bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    sys_args = arguments()
    print(f'Running MCMC test for {sys_args.galname} {sys_args.bin_method} BIN {sys_args.binID}')
    main(sys_args.galname, sys_args.bin_method, sys_args.binID, sys_args.plot, sys_args.quiet)