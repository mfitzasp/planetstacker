import os
import numpy as np
from astropy.io import fits
import sep
from scipy.interpolate import NearestNDInterpolator

# parameters
INPUT_DIR  = 'planet_images'
OUTPUT_DIR = 'processed'
STAMP_SIZE = 600  # pixels
first_mjd_written = False  

def interp_nans(arr):
    """Fill NaNs in 2D array by nearest‐neighbor interpolation."""
    ny, nx = arr.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    # known points
    x_good = X[~mask].ravel()
    y_good = Y[~mask].ravel()
    vals   = arr[~mask].ravel()
    interp = NearestNDInterpolator(list(zip(x_good, y_good)), vals)
    arr[mask] = interp(X[mask], Y[mask])
    return arr

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith('.fits.fz'):
        continue

    infile = os.path.join(INPUT_DIR, fname)
    with fits.open(infile) as hdul:
        data   = hdul[1].data.astype(float)
        hdr    = hdul[1].header
        bpm    = np.asarray(hdul[3].data)
        
        
        # on the first image, extract and write MJD
        if not first_mjd_written:
            # try both common header keys
            mjd = hdr.get('MJD-OBS', hdr.get('MJD'))
            if mjd is not None:
                txt_path = os.path.join(OUTPUT_DIR, 'first_image_mjd.txt')
                with open(txt_path, 'w') as tf:
                    tf.write(str(mjd))
                print(f" wrote first-image MJD ({mjd}) to {txt_path}")
            else:
                print(" no MJD found in header of first image")
            first_mjd_written = True
        
    # apply bad-pixel mask
    mask = bpm > 0
    data[mask] = np.nan

    # background-subtract & source extraction
    bkg = sep.Background(data, mask=mask)
    data_sub = data - bkg.back()
    sources = sep.extract(data_sub, thresh=5.0, err=bkg.rms())
    if len(sources) == 0:
        print(f"[!] no sources found in {fname}, skipping.")
        continue

    # pick the brightest object
    idx = np.argmax(sources['flux'])
    xcen, ycen = sources['x'][idx], sources['y'][idx]
    x0, y0 = int(round(xcen)), int(round(ycen))

    # crop stamp
    half = STAMP_SIZE // 2
    y1 = max(y0 - half, 0)
    x1 = max(x0 - half, 0)
    y2 = y1 + STAMP_SIZE
    x2 = x1 + STAMP_SIZE
    stamp = data[y1:y2, x1:x2]

    # if stamp is smaller than desired, pad with nan
    h, w = stamp.shape
    if h != STAMP_SIZE or w != STAMP_SIZE:
        new = np.full((STAMP_SIZE, STAMP_SIZE), np.nan, dtype=float)
        new[:h, :w] = stamp
        stamp = new

    # interpolate NaNs
    stamp_filled = interp_nans(stamp)

    # prepare output dir based on FILTER header
    filt = hdr.get('FILTER', 'unknown').strip()
    out_dir = os.path.join(OUTPUT_DIR, filt)
    os.makedirs(out_dir, exist_ok=True)

    # write out
    outfile = os.path.join(out_dir, fname)
    fits.writeto(outfile, stamp_filled, header=hdr, overwrite=True)
    print(fname)

print("All done.")
