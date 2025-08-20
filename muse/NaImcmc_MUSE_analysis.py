from __future__ import print_function
import numpy as np
import sys
import os
import glob
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn, vstack
from linetools.spectra.xspectrum1d import XSpectrum1D
import model_NaI
import model_fitter
import continuum_normalize_NaI
import continuum_analyses
import time
from mangadap.config import defaults
from mangadap.util.parser import DefaultConfig
from IPython import embed
from datetime import datetime
import json
import math
import shlex

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


def setup_script(galname, bin_key, beta_corr, binsperrun, scripts_per_exec):
    # data root directory path
    data_root_dir = get_data_path()

    # main cube directory path
    main_cube_dir = os.path.join(data_root_dir, 'muse_cubes')
    # MUSE Line Spread Function file path
    LSF_fil = os.path.join(main_cube_dir, 'LSF-Config_MUSE_WFM')
    if not os.path.isfile(LSF_fil):
        raise ValueError(f'LSF-Config_MUSE_WFM does not exist within {main_cube_dir}')

    # cube directory path
    cube_dir = os.path.join(main_cube_dir, galname)
    if not os.path.isdir(cube_dir):
        raise ValueError(f'{cube_dir} is not a directory within /MUSE_cubes')
    # check if there is only one config file in the cube directory
    if len(glob.glob(f"{cube_dir}/*.ini")) > 1:
        raise ValueError(f'Multiple .ini files within {cube_dir}. {cube_dir} directory must only have '
                         f'configuration file.')
    # input configuration file path
    config_fil = glob.glob(f"{cube_dir}/*.ini")[0]
    if not os.path.isfile(config_fil):
        raise ValueError(f'{os.path.basename(config_fil)} does not exist within {cube_dir}')

    # get parameter values from config file
    cfg = DefaultConfig(config_fil, interpolate=True)
    plate = cfg.getint('plate', default = None)
    ifu = cfg.getint('ifu', default = None)
    redshift = cfg.getfloat('z', default = None)
    
    if redshift is None:
        raise ValueError(f"No redshift found in {config_fil}")

    # output directory path
    output_root_dir = os.path.join(data_root_dir, 'dap_outputs')
    output_gal_dir = os.path.join(output_root_dir, f"{galname}-{bin_key}")
    if not os.path.isdir(output_gal_dir):
        raise ValueError(f'{output_gal_dir} is not a directory within {output_root_dir}.')

    if beta_corr:
        # use beta corrected MUSE cube directory
        beta_dirname = 'BETA-CORR'
        output_gal_sub_dir = os.path.join(output_gal_dir, beta_dirname)

    else:
        # use uncorrected MUSE cube directory
        beta_dirname = 'NO-CORR'
        output_gal_sub_dir = os.path.join(output_gal_dir, beta_dirname)

    # key methdos from analysis plan
    #analysisplan_methods = 'MILESHC-MASTARHC2-NOISM'
    analysisplan_methods = 'MILESHC-MASTARSSP-NOISM'
    # cube directory
    output_cube_dir = os.path.join(output_gal_sub_dir, f"{bin_key}-{analysisplan_methods}", str(plate), str(ifu))
    # paths to the LOGCUBE and MAPS files
    cube_file_path = os.path.join(output_cube_dir,
                                  f"manga-{plate}-{ifu}-LOGCUBE-{bin_key}-{analysisplan_methods}.fits")

    ## directory where the MCMC script will placed in
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(repo_dir, 'mcmc_scripts')
    gal_script_dir = os.path.join(script_dir, f'{galname}-{bin_key}')        
    os.makedirs(gal_script_dir, exist_ok=True)
    # remove old scripts if they exist
    old_files = glob.glob(os.path.join(gal_script_dir, "*.sh"))
    if len(old_files)>0:
        print("Removing old scripts...")
        for file in old_files:
            try:
                os.remove(file)
                print(f'Deleted {file}')
            except Exception as e:
                print(f"Warning: could not delete {file}: {e}")

    
    # directory for logfiles containing terminal output
    log_dir = os.path.join(repo_dir, 'script_logs')
    gal_log_dir = os.path.join(log_dir, f"{galname}-{bin_key}")
    os.makedirs(gal_log_dir, exist_ok=True)

    # directories for the output data of run_mcmc
    NaImcmc_dir = os.path.join(data_root_dir, "mcmc_outputs/")
    mcmc_gal_dir = os.path.join(NaImcmc_dir, f'{galname}-{bin_key}', beta_dirname, analysisplan_methods)
    os.makedirs(mcmc_gal_dir, exist_ok=True)


    # For continuum-normalization around NaI
    # wavelength fitting range inside of NaI region
    fitlim = [5880.0, 5910.0]
    # speed of light in km/s
    c = 2.998e5

    # log maps file
    hdu_map = fits.open(cube_file_path)
    # bin ID has multiple layers of the same bin id map so use first one
    binid_map = hdu_map['BINID'].data[0]

    # Need LSF in km/s
    # This gives LSF in Ang
    # CAUTION: this does not convert air wavelengths
    # to vacuum, or accounts for velocity offset of each bin
    # update: converted LSF wavelength from a air to a vacuum
    configLSF = np.genfromtxt(LSF_fil, comments='#')
    configLSF_wv_air = configLSF[:, 0]
    configLSF_res = configLSF[:, 1]

    # convert to vacuum since LSF is in air
    xspec = XSpectrum1D.from_tuple((configLSF_wv_air, 0.0 * configLSF_wv_air))
    xspec.meta['airvac'] = 'air'
    xspec.airtovac()
    configLSF_wv_vac = xspec.wavelength.value
    # convert LSF wavelength to the restframe using galaxy's redshift
    configLSF_restwv = configLSF_wv_vac / (1.0 + redshift)
    whLSF = np.where((configLSF_restwv > fitlim[0]) & (configLSF_restwv < fitlim[1]))
    median_LSFAng = np.median(configLSF_res[whLSF[0]])
    median_LSFvel = c * median_LSFAng / np.median(configLSF_wv_vac[whLSF[0]])

    LSFvel_str = "{:.2f}".format(median_LSFvel)
    redshift_str = "{:.6f}".format(redshift)

    # number of total bins from bin ID map
    nbins = np.max(binid_map)

    # Number of separate "runs"
    nruns = int(nbins / binsperrun)

    script_commands = []

    for nn in range(nruns + 1):
        startbinid = nn * binsperrun
        endbinid = (nn + 1) * binsperrun

        if (endbinid > nbins):
            endbinid = nbins

        log_file = f"NaImcmc_bin_{startbinid}_{endbinid}_run_{nn}.log"
        log_path = os.path.abspath(os.path.join(gal_log_dir, log_file))
        escaped_log_path = shlex.quote(log_path)

        py_command = (
            f'python NaImcmc_MUSE_analysis.py 1 {galname} {bin_key} {beta_corr} '
            f'{redshift_str} {LSFvel_str} {nn} {startbinid} {endbinid} > {escaped_log_path} 2>&1'
        )

        shell_command = f"nohup {py_command} &"

        script_commands.append(shell_command)

    if scripts_per_exec is None:
        scripts_per_exec = len(script_commands)
        
    outfil = os.path.join(gal_script_dir, f'{galname}-{bin_key}-{beta_dirname}-script')

    chunks = [script_commands[i:i+scripts_per_exec] for i in range(0, len(script_commands), scripts_per_exec)]
    
    print('Writing new scripts...')
    for idx, chunk in enumerate(chunks):
        chunk_filename = f'{outfil}_{idx:02d}.sh'
        with open(chunk_filename, 'w') as f:
            f.write("#!/bin/sh\n")
            for line in chunk:
                f.write(line + '\n')
        print(f'Wrote {chunk_filename}')
    # Set up script that lists
    # input root, redshift, LSFvel, startbinid, endbinid


def append_row_to_fits(filepath, bin_number, samples, percentiles, velocity):
    row_data = Table()
    row_data['bin'] = [bin_number]
    row_data['samples'] = [samples]
    row_data['percentiles'] = [percentiles] 
    row_data['velocities'] = [velocity]

    if not os.path.exists(filepath):
        # Create new file
        row_data.write(filepath, format='fits', overwrite=True)
    else:
        # Read existing data, append new row, and write back
        existing_data = Table.read(filepath)
        combined_data = vstack([existing_data, row_data])
        combined_data.write(filepath, format='fits', overwrite=True)

def run_mcmc(galname, bin_key, beta_corr, redshift, LSFvel, binid_run, startbinid, endbinid):
    start_time1 = time.time()

    # data root directory
    data_root_dir = get_data_path()

    # main cube directory path
    main_cube_dir = os.path.join(data_root_dir, 'muse_cubes')

    # cube directory path
    cube_dir = os.path.join(main_cube_dir, galname)
    if not os.path.isdir(cube_dir):
        raise ValueError(f'{cube_dir} is not a directory within /MUSE_cubes')
    # check if there is only one config file in the cube directory
    if len(glob.glob(f"{cube_dir}/*.ini")) > 1:
        raise ValueError(f'Multiple .ini files within {cube_dir}. {cube_dir} directory must only have '
                         f'configuration file.')
    # input configuration file path
    config_fil = glob.glob(f"{cube_dir}/*.ini")[0]
    if not os.path.isfile(config_fil):
        raise ValueError(f'{os.path.basename(config_fil)} does not exist within {cube_dir}')

    # get parameter values from config file
    cfg = DefaultConfig(config_fil, interpolate=True)
    plate = cfg.getint('plate', default=None)
    ifu = cfg.getint('ifu', default=None)

    # output directory path
    output_root_dir = os.path.join(data_root_dir, 'dap_outputs')
    output_gal_dir = os.path.join(output_root_dir, f"{galname}-{bin_key}")
    if not os.path.isdir(output_gal_dir):
        raise ValueError(f'{output_gal_dir} is not a directory within {output_root_dir}.')

    if beta_corr:
        # use beta corrected MUSE cube directory
        beta_dirname = 'BETA-CORR'
        output_gal_sub_dir = os.path.join(output_gal_dir, beta_dirname)
    else:
        # use uncorrected MUSE cube directory
        beta_dirname = 'NO-CORR'
        output_gal_sub_dir = os.path.join(output_gal_dir, beta_dirname)

    # key methdos from analysis plan
    analysisplan_methods = 'MILESHC-MASTARSSP-NOISM'
    # cube directory
    output_cube_dir = os.path.join(output_gal_sub_dir, f"{bin_key}-{analysisplan_methods}", str(plate), str(ifu))
    # paths to the LOGCUBE and MAPS files
    cube_file_path = os.path.join(output_cube_dir,
                                  f"manga-{plate}-{ifu}-LOGCUBE-{bin_key}-{analysisplan_methods}.fits")

    maps_file_path = os.path.join(output_cube_dir,
                                  f"manga-{plate}-{ifu}-MAPS-{bin_key}-{analysisplan_methods}.fits")

    # main output directory where the MCMC runs will be placed in
    NaImcmc_dir = os.path.join(data_root_dir, "mcmc_outputs/")
    os.makedirs(NaImcmc_dir, exist_ok=True)

    # output mcmc galaxy directory
    mcmc_gal_dir = os.path.join(NaImcmc_dir, f'{galname}-{bin_key}', beta_dirname, analysisplan_methods)
    os.makedirs(mcmc_gal_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d')

    mcmc_save_dir = os.path.join(mcmc_gal_dir,f'Run_{timestamp}')
    os.makedirs(mcmc_save_dir, exist_ok=True)

    outfits_file_name = f'{galname}-{bin_key}-binid-{startbinid}-{endbinid}-samples-run-{binid_run}.fits'
    outfile_path = os.path.join(mcmc_save_dir, outfits_file_name)
    # For continuum-normalization around NaI
    # wavelength continuum fitting range outside of NaI region
    blim = [5850.0, 5870.0]
    rlim = [5910.0, 5930.0]
    # wavelength fitting range inside of NaI region
    fitlim = [5880.0, 5910.0]
    # speed of light in km/s
    c = 2.998e5

    # log maps file
    hdu_map = fits.open(maps_file_path)
    # bin ID has multiple layers of the same bin id map so use first one
    binid_map = hdu_map['BINID'].data[0]

    # stellar velocity map
    ppxf_v_map = hdu_map['STELLAR_VEL'].data

    # Read in binned spectra
    # extract cube data  spectrum
    hdu_cube = fits.open(cube_file_path)
    spec = hdu_cube['FLUX'].data
    # extract error spectrum from cube
    ivar = hdu_cube['IVAR'].data
    espec = np.sqrt(1 / ivar)
    # extract cube best-fit model spectrum
    mod = hdu_cube['MODEL'].data
    # observed wavelength
    obswave = hdu_cube['WAVE'].data

    # Set up array with all relevant binids
    fitbins = np.arange(startbinid, endbinid + 1)

    for qq in fitbins:
        start_time2 = time.time()
        # bin ID index
        ind = binid_map == qq

        # indices of bin spaxels
        ny, nx = np.where(ind)
        y, x = ny[0], nx[0]

        # single bin velocity
        binvel = ppxf_v_map[y, x]

        # single flux, error and model spectrum corresponding to that bin
        flux_bin = spec[:, y, x]
        err_bin = espec[:, y, x]
        mod_bin = mod[:, y, x]

        # Determine bin redshift: cz in km/s = tstellar_kin[*,0]
        bin_z = redshift + ((1 + redshift) * (binvel / c))
        restwave = obswave / (1.0 + bin_z)

        nflux = flux_bin / mod_bin
        nerr = err_bin / mod_bin

        sres_NaI = LSFvel

        infinite_mask = (~np.isfinite(nflux)) | (~np.isfinite(nerr))

        print("""Beginning fit for bin {0} """.format(qq))

        emission_mask = continuum_analyses.emline_mask(nflux, restwave, tuple(blim), tuple(rlim), datamask=infinite_mask, s=1, testrun=True)
        combined_mask = np.logical_or(infinite_mask, emission_mask)
        equiv_w = continuum_analyses.equivalent_width(nflux, restwave, testrun=True)

        if equiv_w <= 0:
            print(f"EQ_W returned {equiv_w}. Skipping fit")
            bin_number = binid_map[ind][0]
            samples = np.zeros((100, 1100, 4))
            percentiles = np.zeros((4,3))
            bin_velocity = -999
            append_row_to_fits(outfile_path, bin_number, samples, percentiles, bin_velocity)
            continue

        # Cut out NaI
        select = np.where((restwave > fitlim[0]) & (restwave < fitlim[1]))
        nflux_nai = nflux[select]
        nerr_nai = nerr[select]
        restwave_nai = restwave[select]
        mask_nai = combined_mask[select]
        
        # check for bad data being masked
        if np.sum(mask_nai) == len(nflux_nai):
            print("All flux pixels masked. Skipping fit")
            bin_number = binid_map[ind][0]
            samples = np.zeros((100, 1100, 4))
            percentiles = np.zeros((4,3))
            bin_velocity = -999
            append_row_to_fits(outfile_path, bin_number, samples, percentiles, bin_velocity)
            continue
        
        data = {'wave': np.ma.array(data = restwave_nai, mask = mask_nai), 'flux': np.ma.array(data = nflux_nai, mask = mask_nai), 
                'err': np.ma.array(data = nerr_nai, mask = mask_nai), 'velres':sres_NaI}
        
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

        append_row_to_fits(outfile_path, bin_number, samples, percentiles, bin_velocity)
        end_time2 = time.time()
        print('Time elapsed for this bin {:.2f} minutes'.format((end_time2 - start_time2) / 60))

    end_time1 = time.time()
    print('Total time elapsed {:.2f} hours'.format((end_time1 - start_time1) / 3600))



def main():
    flg = int(sys.argv[1])
    gal = sys.argv[2] # galaxy name
    bin_key = sys.argv[3] # binning method
    if sys.argv[4].lower() not in ('true', 'false'):
        raise ValueError(f'Correlation correction flag must be either True or False. Input: {sys.argv[4]}')
    beta_corr = sys.argv[4].lower() == 'true' # beta correction flag

    if (flg == 0):
        binsperrun = int(sys.argv[5]) # number of bins per subscript
        try:
            scripts_per_exec = int(sys.argv[6])
        except IndexError:
            scripts_per_exec = None
            
        setup_script(gal, bin_key, beta_corr, binsperrun, scripts_per_exec)

    if (flg == 1):
        redshift = float(sys.argv[5])
        LSFvel = float(sys.argv[6])
        binid_run = int(sys.argv[7])
        startbin = int(sys.argv[8])
        endbin = int(sys.argv[9])

        run_mcmc(galname=gal, bin_key=bin_key, beta_corr=beta_corr, redshift=redshift,
                 LSFvel=LSFvel, binid_run=binid_run,
                 startbinid=startbin, endbinid=endbin)

main()
