# -*- coding: utf-8 -*-
"""
Created on Thu May  1 09:59:30 2025

@author: psyfi
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance  # ← add this


# get current directory
dir_name = os.path.basename(os.getcwd())

#  READ THE FIRST-MJD TXT & SANITISE FOR A FILENAME
mjd_file = os.path.join('processed', 'first_image_mjd.txt')
try:
    with open(mjd_file) as mf:
        mjd_obs = mf.read().strip()
        mjd_label = mjd_obs.replace('.', 'p')
except FileNotFoundError:
    print(f"[!] could not find {mjd_file}, falling back to no-MJD")
    mjd_label = 'noMJD'

# build all your output filenames to include both dir and MJD
OUT_BASE    = f"planet_color_{dir_name}_{mjd_label}"
OUT_FITS       = OUT_BASE + '.fits'
OUT_PNG        = OUT_BASE + '.png'
OUT_PNG_SAT    = OUT_BASE + '_saturated.png'

def mid_stretch_jpeg(data):
    """
    Apply a midtones stretch to the image data using the PixInsight method.

    Args:
        data (np.array): The original image data array.

    Returns:
        np.array: The stretched image data.
    """

    target_bkg = 0.02
    shadows_clip = -1.25

    # Normalize data
    try:
        data = data / np.max(data)
    except ZeroDivisionError:
        pass  # Avoids division by zero if the image is flat

    # Compute average deviation from the median
    median = np.median(data.ravel())
    avg_dev = np.mean(np.abs(data - median))
    c0 = np.clip(median + (shadows_clip * avg_dev), 0, 1)

    # Midtones Transfer Function (MTF)
    def apply_mtf(x, m):
        """Applies the Midtones Transfer Function to an array."""
        shape = x.shape
        x = x.ravel()

        zeros = x == 0
        halfs = x == m
        ones = x == 1
        others = ~(zeros | halfs | ones)

        x[zeros] = 0
        x[halfs] = 0.5
        x[ones] = 1
        x[others] = (m - 1) * x[others] / ((((2 * m) - 1) * x[others]) - m)

        return x.reshape(shape)

    # Initial stretch
    x = median - c0
    m = apply_mtf(x, target_bkg)

    # Clip everything below the shadows clipping point
    data[data < c0] = 0
    above = data >= c0

    # Apply the stretch to the remaining pixels
    x = (data[above] - c0) / (1 - c0)
    data[above] = apply_mtf(x, m)

    return data

# — CONFIGURATION —
STACKS_DIR = 'stacks'

# nominal central wavelengths (nm)
filter_waves = {
    'B': 440,
    'H-Alpha': 656.3,
    'OIII': 500.7,
    'SII': 672.4,
    'up': 365,
    'gp': 475,
    'w': 550,   
    'V': 550,
    'rp': 622,
    'ip': 763,
    'zs': 905
}

# target channel centers (nm)
chan_centers = {'R':650, 'G':550, 'B':450}

# bandwidth σ for weighting (nm)
SIGMA = 100.0

# — LOAD STACKS —
stacks = {}
for fn in os.listdir(STACKS_DIR):
    if fn.lower().endswith('_stack.fits'):
        filt = fn.split('_stack.fits')[0].replace('_drizzled','')
        path = os.path.join(STACKS_DIR, fn)
        data = fits.getdata(path).astype(float)
        
        data= mid_stretch_jpeg(data)
        
        stacks[filt] = data

if not stacks:
    raise RuntimeError("No '*_stack.fits' files found in 'stacks/'")

# assume all stacks same shape
ny, nx = next(iter(stacks.values())).shape

# prepare empty R/G/B & weight maps
R = np.zeros((ny, nx), float)
G = np.zeros((ny, nx), float)
B = np.zeros((ny, nx), float)
wR = np.zeros((ny, nx), float)
wG = np.zeros((ny, nx), float)
wB = np.zeros((ny, nx), float)

# — ACCUMULATE WEIGHTS & FLUXES —
for filt, img in stacks.items():
    lam = filter_waves.get(filt)
    if lam is None:
        print(f"[!] skipping unknown filter '{filt}'")
        continue

    # compute Gaussian weight for each channel
    wr = np.exp(-0.5*((lam - chan_centers['R'])/SIGMA)**2)
    wg = np.exp(-0.5*((lam - chan_centers['G'])/SIGMA)**2)
    wb = np.exp(-0.5*((lam - chan_centers['B'])/SIGMA)**2)

    # broadcast scalar weights across image
    R  += img * wr
    G  += img * wg
    B  += img * wb
    wR += wr
    wG += wg
    wB += wb

# — NORMALIZE & COMBINE —
# avoid division by zero
maskR = wR > 0
maskG = wG > 0
maskB = wB > 0

R[maskR] /= wR[maskR]
G[maskG] /= wG[maskG]
B[maskB] /= wB[maskB]

# clip negatives and normalize channels to [0,1]
def norm_channel(chan):
    chan = np.clip(chan, 0, None)
    m = chan.max()
    return chan/m if m>0 else chan

Rn = norm_channel(R)
Gn = norm_channel(G)
Bn = norm_channel(B)

# build RGB cube
rgb = np.dstack([Rn, Gn, Bn])

SAT_FACTOR=4

# — SAVE OUTPUTS — 
# as PNG preview
plt.imsave(OUT_PNG, rgb, origin='lower')

# boost saturation with Pillow
img = Image.open(OUT_PNG)
enhancer = ImageEnhance.Color(img)
img_sat = enhancer.enhance(SAT_FACTOR)
img_sat.save(OUT_PNG_SAT)

channels = {
    'R': Rn,
    'V': Gn,
    'B': Bn
}

for label, data in channels.items():
    out_fname = f"{OUT_BASE}_{label}.fits"
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.header['FILTER'] = label
    hdu.header['COMMENT'] = f"{label}-channel image"
    hdu.writeto(out_fname, overwrite=True)
    print(f"saved {label}-channel FITS: {out_fname}")
