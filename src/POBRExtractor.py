import os
import yaml
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

from Image import (gaussian_2d, remove_noise, plot, column_size, line_size)
from Psf_fitting import (optimize_cmodel, optimize_psf,
                         compute_reduced_chi2_from_leastsq, flux_dev, flux_exp)
from Astrometry import (estimate_position, estimate_ellipticity)

def main(config):
    """
    Main script to run source detection and photometry.
    """
    print('\n')
    print('############### Photometric Object and Background Reduction Extractor (POBRExtractor) Initialized ###############')
    print('\n')
    print("Starting analysis with the provided configuration.")
    print('\n')
    
    # Unpack config variables for easier access
    io_params = config['io']
    det_params = config['detection']
    conv_params = config['convolution']
    phot_params = config['photometry']

    # Auxiliar objects
    final_data = pd.DataFrame([])
    mags = []
    chi2 = []
    
    # Prepare output directory for plots
    if io_params['save_plots']:
        os.makedirs(io_params['plot_dir'], exist_ok=True)

    # Load detection image
    print(f"Loading detection image: {det_params['band']}")
    image = fits.getdata(io_params['image_files'][det_params['band']])

    plot(image)
    plt.title('Raw image')
    plt.savefig(f"{io_params['plot_dir']}/raw_image.png", dpi=400)
    plt.close()

    # Remove noise and mask regions of interest
    residual_final = remove_noise(image, m_size=conv_params['matrix_size'],
                                  sigma_detection=det_params['sigma_threshold'],
                                  sigma_convolution=conv_params['sigma'])
    
    plot(residual_final)
    plt.title('Image after masking')
    plt.savefig(f"{io_params['plot_dir']}/residual.png", dpi=400)
    plt.close()

    n = 1
    while np.any(residual_final > 0):
        print(f'{n} objects until now...')
        n+=1

        if n >= io_params['n_stop']:
            break
        
        ### Finding the maximum ###
        idx = np.argmax(residual_final)
        i_c, j_c = np.unravel_index(idx, residual_final.shape)

        ### Clipping the image ###
        i_max, i_min = column_size(residual_final, i_c, j_c)
        j_max, j_min = line_size(residual_final, i_c, j_c)
        for band in 'grizy':
            if band == det_params['band']:
                sub_image = image[i_c-i_min:i_c+i_max, j_c-j_min:j_c+j_max]
            else:
                image_tmp = fits.getdata(io_params['image_files'][band])
                sub_image = image_tmp[i_c-i_min:i_c+i_max, j_c-j_min:j_c+j_max]

            ### Fitting the Gaussian PSF ###
            res = optimize_psf(sub_image, i_min, i_max, j_min, j_max)
            i0_psf, j0_psf, offset_psf, amp_psf, sigma_i_psf, sigma_j_psf = res.x

            ### Fitting the Composite Model (CModel) ###
            res_cmodel = optimize_cmodel(sub_image, i_min, j_min)
            i0, j0, off_set, re_dev, re_exp, Ie_dev, Ie_exp, q = res_cmodel.x

            ### Computing the chi2 for each model ###
            redchi_psf = compute_reduced_chi2_from_leastsq(res, sub_image.size)
            redchi_c  = compute_reduced_chi2_from_leastsq(res_cmodel, sub_image.size)
            chi2.append([redchi_psf, redchi_c])

            ### Calculating position and ellipticity ###
            pos_psf, pos_cmodel = estimate_position(io_params['image_files'][band], (i0_psf, j0_psf), (i0, j0))
            e = estimate_ellipticity(q)

            final_data[f'RA'] = pos_psf[0]
            final_data[f'DEC'] = pos_psf[1]
            final_data[f'RA_cmodel'] = pos_cmodel[0]
            final_data[f'DEC_cmodel'] = pos_cmodel[1]
            final_data['ellipticity'] = e
            
            ### Estimating magnitudes ###
            flux_psf = 2.0 * np.pi * amp_psf * sigma_i_psf * sigma_j_psf
            mag_gaussian = -2.5 * np.log10(flux_psf) + phot_params['zero_point']

            flux_d = flux_dev(Ie_dev, re_dev, q)
            flux_e = flux_exp(Ie_exp, re_exp, q)
            flux_cmodel = flux_d + flux_e
            mag_cmodel = -2.5 * np.log10(flux_cmodel) + phot_params['zero_point']
            
            final_data[f'mag_{band}_gaussian'] = mag_gaussian
            final_data[f'mag_{band}_cmodel'] = mag_cmodel

            ### Removing objects from the detection image ###
            residual_final[i_c-i_min:i_c+i_max, j_c-j_min:j_c+j_max] = 0.0

    final_data.to_parquet(io_params['output_filepath'])
    print(f"{n} Objects found !!!")
    print("\n")
    print('############### POBRExtractor Finished ###############')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run photometry on FITS images based on a YAML config file.")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_file}")
        exit()
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        exit()

    main(config)