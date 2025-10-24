import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

def psf_gaussian(ij, i0, j0, offset, amp, sigma_i, sigma_j):
    i, j = ij
    value = offset + amp * np.exp(
        -(((i - i0)**2) / (2 * sigma_i**2) + ((j - j0)**2) / (2 * sigma_j**2))
    )
    return value.ravel()

def cmodel_2d(ij, i0, j0, off_set, re_dev, re_exp, Ie_dev, Ie_exp, q, theta):
    i, j = ij
    i_prime = i - i0
    j_prime = j - j0

    i_rot = i_prime * np.cos(theta) + j_prime * np.sin(theta)
    j_rot = -i_prime * np.sin(theta) + j_prime * np.cos(theta)

    r = np.sqrt((i_rot / q)**2 + j_rot**2)

    b_dev = 7.669
    b_exp = 1.678
    r_re_dev = r / re_dev
    r_re_exp = r / re_exp
    I_dev = Ie_dev * np.exp(-b_dev * (r_re_dev**0.25 - 1))
    I_exp = Ie_exp * np.exp(-b_exp * (r_re_exp - 1))

    return (off_set + I_dev + I_exp).ravel()

def optimize_psf(sub_image, i_max, i_min, j_max, j_min, i, j):
    y_data = sub_image.ravel()
    x_data = (i, j)
    
    offset = np.median(sub_image)
    amp_ini = max(np.max(sub_image) - offset, 0)
    i_center_guess = i_min - 1
    j_center_guess = j_min - 1
    sigma_i = np.mean([i_max, i_min]) / 9
    sigma_j = np.mean([j_max, j_min]) / 9
    initial_guess = (i_center_guess, j_center_guess, offset, amp_ini, sigma_i, sigma_j)

    lower_bounds = [0, 0, 0, 0, 0.001, 0.1]
    upper_bounds = [sub_image.shape[0], sub_image.shape[1], np.inf, np.inf, np.inf, np.inf]

    try:
        popt, pcov = curve_fit(psf_gaussian, x_data, y_data, p0=initial_guess,
                               bounds=(lower_bounds, upper_bounds), maxfev=20000)
        return popt

    # This fit doesn't work always !!! 
    except (ValueError, RuntimeError):
        print("PSF Fit failed (initial guess likely out of bounds or fit did not converge). Returning NaN.")
        return (np.nan,) * 6

def get_moments_guesses(image, background_level):
    """
    Calculates initial guesses for q, theta, and the effective radius (re)
    from the second-order moments of the image.
    """
    img = image.astype(np.float64) - background_level
    img[img < 0] = 0
    total_flux = np.sum(img)
    if total_flux <= 1e-6:
        return 0.9, 0.0, 2.0

    y, x = np.indices(img.shape)
    x_c = np.sum(x * img) / total_flux
    y_c = np.sum(y * img) / total_flux
    dx = x - x_c
    dy = y - y_c
    mu_xx = np.sum(dx**2 * img) / total_flux
    mu_yy = np.sum(dy**2 * img) / total_flux
    mu_xy = np.sum(dx * dy * img) / total_flux

    # Theta and q calculation (unchanged)
    theta_guess = 0.5 * np.arctan2(2 * mu_xy, mu_xx - mu_yy)
    if theta_guess < 0:
        theta_guess += np.pi
    
    term = np.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
    lambda_1 = 0.5 * ((mu_xx + mu_yy) + term) 
    lambda_2 = 0.5 * ((mu_xx + mu_yy) - term) 

    if lambda_1 > 1e-6 and lambda_2 > 1e-6:
        q_guess = np.sqrt(lambda_2 / lambda_1)
    else:
        q_guess = 0.9
        
    re_guess = 1.5 * np.sqrt(lambda_1)

    return np.clip(q_guess, 0.05, 1.0), theta_guess, max(re_guess, 0.5)

def optimize_cmodel(sub_image, psf_kernel, i, j, psf_params):
    psf_kernel = psf_kernel.astype(np.float32)
    y_data = sub_image.ravel()
    x_data = (i, j)

    i0_psf, j0_psf, offset_psf, amp_psf, sigma_i_psf, sigma_j_psf = psf_params
    
    q_guess, theta_guess, re_guess = get_moments_guesses(sub_image, offset_psf)

    peak_pixel_value = np.max(sub_image) - offset_psf
    b_dev, b_exp = 7.669, 1.678
    
    Ie_dev_guess = max((peak_pixel_value / 2.0) / np.exp(b_dev), 1e-4)
    Ie_exp_guess = max((peak_pixel_value / 2.0) / np.exp(b_exp), 1e-4)

    initial_guess_cmodel = (
        re_guess, re_guess,
        Ie_dev_guess, Ie_exp_guess,
        q_guess, theta_guess
    )

    min_re = np.sqrt(sigma_i_psf**2 + sigma_j_psf**2)
    max_re = max(sub_image.shape)
    
    max_Ie_dev = Ie_dev_guess * 50
    max_Ie_exp = Ie_exp_guess * 50

    lower_bounds = [min_re, min_re, 0, 0, 0.05, 0.0]
    upper_bounds = [max_re, max_re, max_Ie_dev, max_Ie_exp, 1.0, np.pi]

    def wrapped_cmodel_fixed_center(ij, re_dev, re_exp, Ie_dev, Ie_exp, q, theta):
        pure_model_1d = cmodel_2d(
            ij, i0_psf, j0_psf, offset_psf, re_dev, re_exp, Ie_dev, Ie_exp, q, theta
        )
        pure_model_2d = pure_model_1d.reshape(sub_image.shape)
        convolved_model = fftconvolve(pure_model_2d, psf_kernel, mode='same')
        return convolved_model.ravel()

    try:
        popt, pcov = curve_fit(
            wrapped_cmodel_fixed_center, x_data, y_data,
            p0=initial_guess_cmodel,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000)
        
        re_dev, re_exp, Ie_dev, Ie_exp, q, theta = popt
        return (i0_psf, j0_psf, offset_psf, re_dev, re_exp, Ie_dev, Ie_exp, q, theta)

    except (RuntimeError, ValueError):
        print("CModel Fit failed, returning NaN.")
        return (np.nan,) * 9


def flux_dev(Ie_dev, re_dev, q):
    return 22.67 * Ie_dev * re_dev**2 * q

def flux_exp(Ie_exp, re_exp, q):
    return 3.80 * Ie_exp * re_exp**2 * q