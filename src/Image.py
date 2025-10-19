import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
from scipy.ndimage import convolve
from scipy.optimize import least_squares



def gaussian_2d(ij, i0, j0, offset, amp, sigma):
    i, j = ij
    return offset + amp * np.exp(-((i-i0)**2 + (j-j0)**2)/(2*sigma**2))

def remove_noise(image, m_size, sigma_detection, sigma_convolution):
    i,j = np.meshgrid(np.arange(m_size), np.arange(m_size))
    
    # Convolving with a gaussian matrix
    gaussian_matrix = gaussian_2d((i,j), i0=m_size/2, j0=m_size/2 , 
                                  offset=0, amp=1, sigma=sigma_convolution)
    blurry = convolve(image, gaussian_matrix, mode='constant', cval=0)

    # Removing noise
    residual = blurry-image
    residual_no_zero = np.where(residual>0, residual,0)

    # Masking interesting regions
    residual_subtracted = residual_no_zero - sigma_detection*residual_no_zero.std()
    residual_final = np.where(residual_subtracted>0, residual_subtracted, 0)

    return residual_final

def plot(image_input, vmin=None, vmax=None):
    if vmin==None:
        if vmax==None:
            plt.imshow(image_input, vmin=image_input.mean()-image_input.std(), vmax=image_input.mean()+image_input.std(), cmap='gray')
        else:
            plt.imshow(image_input, vmin=image_input.mean()-image_input.std(), vmax=vmax, cmap='gray')
    elif vmax==None:
        plt.imshow(image_input, vmin=vmin, vmax=image_input.mean()+image_input.std(), cmap='gray')
    else:
        plt.imshow(image_input, vmin=vmin, vmax=vmax, cmap='gray')

def column_size(matrix, i_initial, j_initial):
    max_i = matrix.shape[0] - 1
    
    u_size = 0
    d_size = 0

    i_u = i_initial
    i_d = i_initial

    # Up movement
    while u_size == 0:
        if i_u <= max_i and matrix[i_u, j_initial] > 0:
            i_u += 1
        else:
            u_size = np.abs(i_initial - i_u)
    # Down movement
    while d_size == 0:
        if i_d >= 0 and matrix[i_d, j_initial] > 0:
            i_d -= 1
        else:
            d_size = np.abs(i_initial - i_d)

    return u_size, d_size-1

def line_size(matrix, i_initial, j_initial):
    max_i = matrix.shape[1] - 1

    l_size = 0
    r_size = 0

    j_l = j_initial
    j_r = j_initial

    # Left Movement
    while l_size == 0:
        if j_l >= 0 and matrix[i_initial, j_l] > 0:
            j_l -= 1
        else:
            l_size = np.abs(j_initial - j_l)

    # Right movement
    while r_size == 0:
        if j_r <= max_i and matrix[i_initial, j_r] > 0:
            j_r += 1
        else:
            r_size = np.abs(j_initial - j_r)

    return r_size, l_size-1