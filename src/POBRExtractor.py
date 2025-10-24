import os
import yaml
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

from Image import (gaussian_2d, remove_noise, plot, column_size, line_size)
from Psf_fitting import (optimize_cmodel, optimize_psf, flux_dev, flux_exp)
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
    object_data_list = []
    
    # Prepare output directory for plots
    if io_params['save_plots']:
        os.makedirs(io_params['plot_dir'], exist_ok=True)

    # Load detection image
    print(f"Loading detection image: {det_params['band']}")
    image = fits.getdata(io_params['image_files'][det_params['band']])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot(image)
    plt.title('Raw image', color='white')
    plt.axis('off') 
    
    # Remove noise and mask regions of interest
    residual_final = remove_noise(
        image,
        m_size=conv_params['matrix_size'],
        sigma_detection=det_params['sigma_threshold'],
        sigma_convolution=conv_params['sigma']
    )
    
    plt.subplot(1, 2, 2)
    plot(residual_final, vmin=0)
    plt.title('Image after masking', color='white')
    plt.axis('off') 
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    plt.savefig(f"{io_params['plot_dir']}/residual_vs_raw.png", dpi=500, transparent=True)
    plt.close()

    n = 1
    psf_kernels = {}

    while np.any(residual_final > 0):
        if n >= io_params['n_stop']:
            break
        print(f'{n} objects detected until now...')
        
        current_object_data = {}
        
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
                
            sub_image = sub_image.astype(np.float32)

            j = np.arange(sub_image.shape[1])
            i = np.arange(sub_image.shape[0])
            j, i = np.meshgrid(j, i)
            
            ### Fitting the Gaussian PSF ###
            params_psf = optimize_psf(sub_image, i_max, i_min, j_max, j_min, i, j)
            i0_psf, j0_psf, offset_psf, amp_psf, sigma_i_psf, sigma_j_psf = params_psf

            if n == 1:
                print(f"    Estimating and storing PSF kernel for band '{band}' from the brightest object.")
                i_range = np.arange(sub_image.shape[0])
                j_range = np.arange(sub_image.shape[1])
                jj, ii = np.meshgrid(j_range, i_range)

                unnormalized_kernel = gaussian_2d((ii, jj), i0_psf, j0_psf, amp_psf, sigma_i_psf, sigma_j_psf)
                psf_kernels[band] = unnormalized_kernel / np.sum(unnormalized_kernel)

            ### Fitting the Composite Model (CModel) ###
            params_cmodel = optimize_cmodel(sub_image, psf_kernels[band], i, j, psf_params=params_psf)
            
            i0, j0, off_set, re_dev, re_exp, Ie_dev, Ie_exp, q, theta = params_cmodel
                
            ### Calculating position and ellipticity ###
            i_global_psf = i0_psf + (i_c - i_min)
            i_global_cmodel = i0 + (i_c - i_min)
            j_global_psf = j0_psf + (j_c - j_min)
            j_global_cmodel = j0 + (j_c - j_min)
          
            ### Calculating position and ellipticity ###
            pos_psf, pos_cmodel = estimate_position(io_params['image_files'][band], 
                                                    (i_global_psf, j_global_psf), 
                                                    (i_global_cmodel, j_global_cmodel)
                                                   )
            e = estimate_ellipticity(q)

            current_object_data[f'RA_{band}_psf'] = pos_psf[0]
            current_object_data[f'DEC_{band}_psf'] = pos_psf[1]
            current_object_data[f'RA_{band}_cmodel'] = pos_cmodel[0]
            current_object_data[f'DEC_{band}_cmodel'] = pos_cmodel[1]
            current_object_data[f'ellipticity_{band}'] = e
            
            ### Estimating magnitudes ###
            flux_psf = 2.0 * np.pi * amp_psf * sigma_i_psf * sigma_j_psf
            mag_gaussian = -2.5 * np.log10(flux_psf) + phot_params['zero_point']

            flux_d = flux_dev(Ie_dev, re_dev, q)
            flux_e = flux_exp(Ie_exp, re_exp, q)
            flux_cmodel = flux_d + flux_e
            mag_cmodel = -2.5 * np.log10(flux_cmodel) + phot_params['zero_point']

            current_object_data[f'mag_{band}_gaussian'] = mag_gaussian
            current_object_data[f'mag_{band}_cmodel'] = mag_cmodel

        object_data_list.append(current_object_data)

        ### Removing objects from the detection image ###
        residual_final[i_c-i_min:i_c+i_max, j_c-j_min:j_c+j_max] = 0.0
        n += 1

    final_data = pd.DataFrame(object_data_list)
    final_data.to_parquet(io_params['output_filepath'])
    print(f"{n-1} Objects found !!!")
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