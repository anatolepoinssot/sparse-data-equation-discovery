"""
cleanup_and_resume.py
=====================
Run this BEFORE launching main_resume.edp.

Removes all incomplete lines from sensor_T.txt and sensor_phi2.txt,
keeping only the 82 blocks that are complete in BOTH files.
Then launches the FreeFem++ resume script automatically.
"""

import os, subprocess, sys

RAW_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
T_FILE    = os.path.join(RAW_DIR, "sensor_T.txt")
PHI2_FILE = os.path.join(RAW_DIR, "sensor_phi2.txt")

# Safe restart: keep only a values strictly below this threshold
# (ii=82 -> a=0.896482 is the first block NOT complete in phi2)
A_CUTOFF  = 0.896482 - 1e-6   # keep a < 0.896482
NCOLS_OK  = 22

def clean_file(path, label):
    with open(path, "r") as f:
        lines = [l for l in f if l.strip()]

    kept, removed = [], 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) != NCOLS_OK:
            removed += 1
            continue
        if float(parts[0]) >= A_CUTOFF:
            removed += 1
        else:
            kept.append(line if line.endswith("\n") else line + "\n")

    with open(path, "w") as f:
        f.writelines(kept)

    print(f"  {label}: kept {len(kept)} lines, removed {removed}")
    return len(kept)

print("Cleaning raw data files (keeping 82 complete blocks = 8200 lines each)...")
n_T    = clean_file(T_FILE,    "sensor_T.txt   ")
n_phi2 = clean_file(PHI2_FILE, "sensor_phi2.txt")

if n_T != n_phi2:
    print(f"[ERROR] Still mismatched: T={n_T}, phi2={n_phi2}")
    sys.exit(1)

if n_T != 8200:
    print(f"[WARNING] Expected 8200 lines (82 x 100), got {n_T}")
else:
    print(f"OK — {n_T} lines in each file (82 complete blocks x 100 steps)")

print("\nFiles are clean. You can now run:")
print("  FreeFem++ freefem/main_resume.edp")
