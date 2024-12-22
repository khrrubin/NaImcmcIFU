from __future__ import print_function
import numpy as np
import sys
import os
import glob
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from linetools.spectra.xspectrum1d import XSpectrum1D
import model_NaI
import model_fitter
import continuum_normalize_NaI
import time
from mangadap.config import defaults
from mangadap.util.parser import DefaultConfig
from IPython import embed
from datetime import datetime

def setup_script(galname, bin_key, beta_corr):
    # mangadap_muse root directory path
    #mangadap_muse_dir = os.path.dirname(os.path.dirname(defaults.dap_data_root()))
    mangadap_muse_dir = "/data2/muse/"
    # main cube directory path
    main_cube_dir = os.path.join(mangadap_muse_dir, 'muse_cubes')
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
    plate = cfg.getint('plate', default=None)
    ifu = cfg.getint('ifu', default=None)

    # output directory path
    output_root_dir = os.path.join(mangadap_muse_dir, 'dap_outputs')
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

    # directory where the MCMC script will placed in
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outfil = f'{script_dir}/mcmc_scripts/{galname}-{bin_key}-{beta_dirname}-TEST-script'


    test_log_dir = os.path.join(script_dir, 'test_logs')
    if not os.path.exists(test_log_dir):
        os.mkdir(test_log_dir)

    testlog_fname = os.path.join(test_log_dir, f"{galname}-{bin_key}-TESTmcmc.log")

    # For continuum-normalization around NaI
    # wavelength fitting range inside of NaI region
    fitlim = [5880.0, 5910.0]
    # speed of light in km/s
    c = 2.998e5
    # NGC 4030 redshift's
    redshift = 0.00489
    # NGC 1042 redshift's
    # redshift = 0.00460

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

    f = open(outfil, "w")
    f.write("#!/bin/sh\n")

    # number of total bins from bin ID map
    nbins = np.max(binid_map)

    # Number of separate "runs"

    startbinid = 0
    endbinid = nbins

    jobname = 'NaImcmc' + '_bin_' + str(startbinid) + '_' + str(endbinid) + '_run'

    f.write('screen -mdLS ' + jobname + ' -Logfile' + ' ' + testlog_fname + ' sh -c python NaImcmc_MUSE_analysis_testing_copy.py 1 ' +
            galname + ' ' + bin_key + ' ' + str(beta_corr) + ' ' +
            redshift_str + ' ' + LSFvel_str + ' ' + str(0) + ' ' +
            str(startbinid) + ' ' + str(endbinid))

    f.close()
    # Set up script that lists
    # input root, redshift, LSFvel, startbinid, endbinid

def run_mcmc(galname, bin_key, beta_corr,redshift, LSFvel, binid_run, startbinid, endbinid):
    start_time1 = time.time()
    # mangadap_muse root directory path
    #mangadap_muse_dir = os.path.dirname(os.path.dirname(defaults.dap_data_root()))
    mangadap_muse_dir = "/data2/muse"
    
    # main cube directory path
    main_cube_dir = os.path.join(mangadap_muse_dir, 'muse_cubes')

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
    output_root_dir = os.path.join(mangadap_muse_dir, 'dap_outputs')
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
    #NaImcmc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NaI_MCMC_output')
    NaImcmc_dir = "/data2/muse/mcmc_outputs/"
    if not os.path.isdir(NaImcmc_dir):
        os.makedirs(NaImcmc_dir)

    # output mcmc galaxy directory
    mcmc_gal_dir = os.path.join(NaImcmc_dir, f'{galname}-{bin_key}', beta_dirname, analysisplan_methods)
    if not os.path.isdir(mcmc_gal_dir):
        os.makedirs(mcmc_gal_dir)

    run_num = len(os.listdir(mcmc_gal_dir))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mcmc_save_dir = os.path.join(mcmc_gal_dir,f'TEST_Run_{run_num}-{timestamp}')


    outfits_file_name = f'{galname}-{bin_key}-binid-{startbinid}-{endbinid}-samples-run-{binid_run}.fits'

    # log maps file
    hdu_map = fits.open(maps_file_path)
    # bin ID has multiple layers of the same bin id map so use first one
    binid_map = hdu_map['BINID'].data[0]

    # Read in binned spectra
    # extract cube data  spectrum
    hdu_cube = fits.open(cube_file_path)

    sv_samples = []
    sv_binnumber = []
    sv_percentiles = []
    sv_velocities = []

    # Set up array with all relevant binids
    fitbins = np.arange(startbinid, endbinid + 1)

    for qq in fitbins:
        start_time2 = time.time()
        # bin ID index
        ind = binid_map == qq

        sv_binnumber.append(binid_map[ind][0])
        sv_samples.append(0)
        sv_percentiles.append(0)
        sv_velocities.append(0)
        end_time2 = time.time()
        print('Time elapsed for this bin {:.2f} minutes'.format((end_time2 - start_time2) / 60))

    t = Table([sv_binnumber, sv_samples, sv_percentiles, sv_velocities],
              names=('bin', 'samples', 'percentiles', 'velocities'))
    fits.writeto(os.path.join(mcmc_save_dir, outfits_file_name), np.array(t), overwrite=True)
    end_time1 = time.time()
    print('Total time elapsed {:.2f} hours'.format((end_time1 - start_time1) / 3600))


def main():
    flg = int(sys.argv[1])

    if (flg == 0):
        # galaxy name
        gal = sys.argv[2]
        # binning method
        bin_key = sys.argv[3]
        # correlation correction flag
        if sys.argv[4] == 'True':
            beta_corr = True
        elif sys.argv[4] == 'False':
            beta_corr = False
        else:
            raise ValueError('correlation correction flag must be either True or False')
        # number of bins per run
        setup_script(gal, bin_key, beta_corr)

    if (flg == 1):
        # pdb.set_trace()
        gal = sys.argv[2]
        bin_key = sys.argv[3]
        if sys.argv[4] == 'True':
            beta_corr = True
        else:
            beta_corr = False
        redshift = float(sys.argv[5])
        LSFvel = float(sys.argv[6])
        binid_run = int(sys.argv[7])
        startbin = int(sys.argv[8])
        endbin = int(sys.argv[9])

        run_mcmc(galname=gal, bin_key=bin_key, beta_corr=beta_corr, redshift=redshift,
                 LSFvel=LSFvel, binid_run=binid_run,
                 startbinid=startbin, endbinid=endbin)

main()
