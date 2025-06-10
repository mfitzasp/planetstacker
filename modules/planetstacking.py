import os
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

PROCESSED_DIR = 'processed'
STACKS_DIR    = 'stacks'
FACTOR        = 2       # upsampling factor
PIXFRACT      = 1.0     # drop size fraction
SIGMA         = 3.0     # clipping threshold (σ)
MAX_ITER      = 5       # max sigma‐clip iterations

os.makedirs(STACKS_DIR, exist_ok=True)

def drizzle_single(img, factor=2, pixfrac=1.0):
    """
    Drizzle one image onto a finer grid.
    Returns an (H*factor, W*factor) array.
    """
    H, W    = img.shape
    OH, OW  = H * factor, W * factor
    out_arr = np.zeros((OH, OW), dtype=float)
    weight  = np.zeros((OH, OW), dtype=float)
    drop    = pixfrac * factor

    for i in range(H):
        for j in range(W):
            val = img[i, j]
            if np.isnan(val):
                continue
            yc = (i + 0.5) * factor
            xc = (j + 0.5) * factor
            y0, y1 = yc - drop/2, yc + drop/2
            x0, x1 = xc - drop/2, xc + drop/2

            yi0, yi1 = int(np.floor(y0)), int(np.ceil(y1))
            xi0, xi1 = int(np.floor(x0)), int(np.ceil(x1))

            for yy in range(yi0, yi1):
                if yy < 0 or yy >= OH: continue
                for xx in range(xi0, xi1):
                    if xx < 0 or xx >= OW: continue
                    oy = min(y1, yy+1) - max(y0, yy)
                    ox = min(x1, xx+1) - max(x0, xx)
                    if oy > 0 and ox > 0:
                        area = oy*ox
                        out_arr[yy, xx] += val * area
                        weight[yy, xx]  += area

    # normalize; leave empty pixels as NaN
    mask = weight > 0
    result = np.full_like(out_arr, np.nan)
    result[mask] = out_arr[mask] / weight[mask]
    return result

for filt in sorted(os.listdir(PROCESSED_DIR)):
    dir_in = os.path.join(PROCESSED_DIR, filt)
    if not os.path.isdir(dir_in):
        continue

    drizzled = []
    hdr0 = None

    for fn in sorted(os.listdir(dir_in)):
        if not fn.lower().endswith('.fits.fz'):
            continue
        path = os.path.join(dir_in, fn)
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(float)
            hdr  = hdul[0].header
        if hdr0 is None:
            hdr0 = hdr.copy()

        drizzled.append(drizzle_single(data, factor=FACTOR, pixfrac=PIXFRACT))

    if not drizzled:
        print(f"[!] no stamps for filter '{filt}', skipping.")
        continue

    # stack into 3D array: shape = (N_images, OH, OW)
    stack_cube = np.stack(drizzled, axis=0)

    # sigma‐clip across axis=0 (the image axis)
    clipped = sigma_clip(
        stack_cube,
        sigma=SIGMA,
        sigma_lower=SIGMA,
        sigma_upper=SIGMA,
        maxiters=MAX_ITER,
        axis=0
    )

    # take the mean of the clipped values
    final_stack = np.ma.mean(clipped, axis=0).filled(np.nan)

    # update header axes
    hdr0['NAXIS1'] *= FACTOR
    hdr0['NAXIS2'] *= FACTOR

    out_path = os.path.join(STACKS_DIR, f"{filt}_drizzled_stack.fits")
    fits.writeto(out_path, final_stack, header=hdr0, overwrite=True)

print("All done.")
