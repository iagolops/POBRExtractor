import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.optimize import curve_fit, least_squares

def psf_gaussian(ij, i0, j0, offset, amp, sigma_i, sigma_j):
    """
    Calculates a 2D elliptical Gaussian function.
    
    """
    i, j = ij
    value = offset + amp * np.exp(
        -(((i - i0)**2) / (2 * sigma_i**2) + ((j - j0)**2) / (2 * sigma_j**2))
    )
    return value.ravel()
    

def cmodel_2d(ij, i0, j0, off_set, re_dev, re_exp, Ie_dev, Ie_exp, q):
    """
    Calculates a 2D composite galaxy model (Sérsic + Exponential).

    This model combines a de Vaucouleurs (n=4 Sérsic profile) for the bulge
    and an exponential disk profile (n=1 Sérsic profile) for the disk.

    """
    i, j = ij
    
    i_prime = i - i0
    j_prime = j - j0

    r = np.sqrt(i_prime**2 + (j_prime/q)**2)

    b_dev = 7.669  # Constant for n=4 Sérsic profile
    b_exp = 1.678  # Constant for n=1 Sérsic profile
    
    I_dev = Ie_dev * np.exp(-b_dev * ((r / re_dev)**(1/4) - 1))
    I_exp = Ie_exp * np.exp(-b_exp * ((r / re_exp) - 1))

    return (off_set + I_dev + I_exp).ravel()


def residuals(params, i, j, data):
    """
    Calculates the residuals for the psf_gaussian model fit.

    """
    i0, j0, offset, amp, sigma_i, sigma_j = params
    model = psf_gaussian((i, j), i0, j0, offset, amp, sigma_i, sigma_j)
    return (model - data.ravel())


def residuals_cmodel_2d(params, i, j, data):
    """
    Calculates the residuals for the cmodel_2d galaxy model fit.

    """
    i0, j0, off_set, re_dev, re_exp, Ie_dev, Ie_exp, q = params
    model = cmodel_2d((i, j), i0, j0, off_set, re_dev, re_exp, Ie_dev, Ie_exp, q)
    return (model - data.ravel())


def optimize_psf(sub_image, i_min, i_max, j_min, j_max):
    """
    Performs a least-squares fit of the 2D Gaussian PSF model to an image.

    This function sets up the initial guesses and bounds for the PSF parameters
    and runs the optimization routine.

    Args:
        sub_image (numpy.ndarray): The 2D image data of a star.
        i_min, i_max, j_min, j_max (int): The coordinate boundaries of the star,
                                         used for initial guesses.

    Returns:
        scipy.optimize.OptimizeResult: The result object from least_squares.
    """
    j = np.arange(sub_image.shape[1])
    i = np.arange(sub_image.shape[0])
    j, i = np.meshgrid(j, i)

    # Initial guess
    offset = max(sub_image.mean(), 0)
    amp_ini = max(np.max(sub_image) - offset, 0)
    sigma_i = max(np.mean([i_max, i_min]) / 10, 0)
    sigma_j = max(np.mean([j_max, j_min]) / 10, 0)
    i_min = max(i_min, 0)
    j_min = max(j_min, 0)

    initial_guess = (i_min, j_min, offset, amp_ini, sigma_i, sigma_j)

    # Optimizing
    res = least_squares(residuals, initial_guess, args=(i, j, sub_image),
                        method='trf', loss='huber', f_scale=0.1,
                        max_nfev=20000)
    return res

def optimize_cmodel(sub_image, i_min, j_min):
    """
    Performs a least-squares fit of the 2D composite galaxy model to an image.

    This function sets up the initial guesses and bounds for the galaxy model
    parameters and runs the optimization routine.

    Args:
        sub_image (numpy.ndarray): The 2D image data of a galaxy.
        i_min, j_min (int): The approximate center coordinates of the galaxy for
                            the initial guess.

    Returns:
        scipy.optimize.OptimizeResult: The result object from least_squares.
    """
    j = np.arange(sub_image.shape[1])
    i = np.arange(sub_image.shape[0])
    j, i = np.meshgrid(j, i)

    # Initial guess
    peak_intensity = max(np.max(sub_image) - sub_image.mean(), 0)
    Ie_dev_guess = max(peak_intensity / 2140.0, 0)
    Ie_exp_guess = max(peak_intensity / 5.35, 0)
    i_min = max(i_min, 0)
    j_min = max(j_min, 0)
    offset = max(sub_image.mean(), 0)
    re = max(np.mean(sub_image.shape[0] + sub_image.shape[1]) / 5, 0) # for dev and exp
    q = 0.8

    initial_guess_cmodel = (
        i_min, j_min, offset, re, re, Ie_dev_guess, Ie_exp_guess, q)
    
    # Optimizing
    res_cmodel = least_squares(residuals_cmodel_2d, initial_guess_cmodel,
        args=(i, j, sub_image), method='trf', loss='huber', f_scale=0.1,
        max_nfev=20000,
        bounds=(
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0]))
    
    return res_cmodel
    

def compute_reduced_chi2_from_leastsq(res, ndata):
    """
    Computes the reduced chi-squared statistic from a least_squares result.

    """
    sumsq = 2.0 * res.cost  # cost is 0.5 * sum(residuals**2)
    dof = max(1, ndata - len(res.x)) # Degrees of freedom
    return sumsq / dof


def flux_dev(Ie_dev, re_dev, q):
    """
    Calculates the total integrated flux of the de Vaucouleurs component.

    """
    return 22.67 * Ie_dev * re_dev**2 * q


def flux_exp(Ie_exp, re_exp, q):
    """
    Calculates the total integrated flux of the exponential disk component.

    """
    return 3.80 * Ie_exp * re_exp**2 * q
