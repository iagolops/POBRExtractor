from astropy.io import fits

def estimate_position(image_file, psf_coords, obj_coords):
    i0_psf, j0_psf = psf_coords
    i0, j0 = obj_coords
    
    with fits.open(image_file) as im:
        hdr = im[1].header
    
    crpix1, crpix2 = hdr['CRPIX1'], hdr['CRPIX2']
    crval1, crval2 = hdr['CRVAL1'], hdr['CRVAL2']
    cd11, cd22 = hdr['CD1_1'], hdr['CD2_2']
    
    ra_psf = crval1 + (i0_psf - crpix1) * cd11
    dec_psf = crval2 + (j0_psf - crpix2) * cd22
    ra = crval1 + (i0 - crpix1) * cd11
    dec = crval2 + (j0 - crpix2) * cd22
    
    return (ra_psf, dec_psf), (ra, dec)
    
    
def estimate_ellipticity(q):
    return (1 - q) / (1 + q)
