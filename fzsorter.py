import shutil
from astropy.io import fits
import os
import glob

for f in glob.glob('*.fits.fz'):
    with fits.open(f, mode='readonly') as hdul:
        reqnum = hdul[1].header.get('REQNUM')
    if not reqnum:
        print(f"⚠️  No REQNUM in {f.name}; skipping")
        continue

    target = str(reqnum)
    os.makedirs(target, exist_ok=True)
    os.makedirs(target +'/planet_images', exist_ok=True)
    dest = target +'/planet_images/' + f

    shutil.move(str(f), str(dest))
