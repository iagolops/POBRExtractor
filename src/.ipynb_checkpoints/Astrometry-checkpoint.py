from astropy.io import fits
from astropy.wcs import WCS

def estimate_position(image_file, psf_coords, obj_coords):
    i_psf, j_psf = psf_coords
    i_obj, j_obj = obj_coords

    with fits.open(image_file) as im:
        hdr = im[1].header
        wcs = WCS(hdr)

    ra_psf, dec_psf = wcs.wcs_pix2world(j_psf, i_psf, 0)
    ra_obj, dec_obj = wcs.wcs_pix2world(j_obj, i_obj, 0)

    ra_psf = float(ra_psf)
    dec_psf = float(dec_psf)
    ra_obj = float(ra_obj)
    dec_obj = float(dec_obj)

    return (ra_psf, dec_psf), (ra_obj, dec_obj)
    return (ra_psf, dec_psf), (ra_obj, dec_obj)


def estimate_ellipticity(q):
    return (1 - q) / (1 + q)