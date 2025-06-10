import os
import shutil
import subprocess
from PIL import Image
import glob

MODULES = ["preplaneting.py", "planetstacking.py", "planetcolour.py"]
MODULES_DIR = "modules"
PLANET_DIR = "planet"
OUTPUT_FILE  = "planet_color.fits"

for entry in os.scandir(PLANET_DIR):
    if not entry.is_dir():
        continue

    print(f"\n=== Processing folder: {entry.name} ===")
    target_dir = entry.path
    
    # skip this folder if the final output already exists
    # if os.path.isfile(os.path.join(target_dir, OUTPUT_FILE)):
    #     print(f"\n=== Skipping {entry.name}: '{OUTPUT_FILE}' already present ===")
    #     continue

    # Copy the three .py files in
    for m in MODULES:
        src = os.path.join(MODULES_DIR, m)
        dst = os.path.join(target_dir, m)
        shutil.copy(src, dst)

    # Run each script in sequence
    for m in MODULES:
        print(f"→ Running {m} ...", end=" ")
        try:
            result = subprocess.run(
                ["python", m],
                cwd=target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,      
                check=True      
            )
            print("OK")
        except subprocess.CalledProcessError as e:
            print(f"FAILED (exit {e.returncode})")
            print("  stdout:")
            print("\n".join("    " + line for line in e.stdout.splitlines()))
            print("  stderr:")
            print("\n".join("    " + line for line in e.stderr.splitlines()))
            break
        
# Collect and sort all the “saturated” PNGs
saturated_paths = []
for entry in os.scandir(PLANET_DIR):
    if not entry.is_dir():
        continue
    pattern = os.path.join(entry.path, '*saturated*.png')
    saturated_paths.extend(glob.glob(pattern))

# Sort alphanumerically
saturated_paths = sorted(saturated_paths, key=lambda p: os.path.basename(p))

if not saturated_paths:
    print("No saturated PNGs found – nothing to animate.")
else:
    #  Load images
    frames = [Image.open(p) for p in saturated_paths]

    # Save as a looping GIF
    out_gif = os.path.join(PLANET_DIR, 'planet_saturated.gif')
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )
    print(f"Animated GIF written to {out_gif}")
    
print("\nAll done.")
