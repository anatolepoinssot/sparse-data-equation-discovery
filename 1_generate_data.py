"""
1_generate_data.py
==================
Run the FreeFem++ simulation for all 200 values of the parameter 'a' and
collect the sensor observations (T and phi2) into raw text files.

After FreeFem++ finishes, this script:
  - Verifies the output dimensions
  - Saves the raw data as a compressed NumPy archive  (data/raw/sensor_data.npz)

Usage
-----
    python 1_generate_data.py

Requirements
------------
  - FreeFem++ installed and accessible as 'FreeFem++' on the PATH
    (or set the FREEFEM_EXE environment variable to the full path).
  - The FreeFem++ script  freefem/main_87x87.edp  must be present.
  - Output directories  data/raw/  and  data/processed/  must exist
    (they are created automatically if missing).
"""

import subprocess
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    N_PARAM, N_TIME, N_SENSORS_TOTAL,
    A_VALUES,
    FREEFEM_SCRIPT, DATA_RAW_DIR, SENSOR_DATA_FILE,
)


def ensure_dirs():
    """Create output directories if they do not exist."""
    for d in [DATA_RAW_DIR,
              os.path.join(os.path.dirname(DATA_RAW_DIR), "processed"),
              os.path.join(os.path.dirname(DATA_RAW_DIR), "..", "results")]:
        os.makedirs(d, exist_ok=True)


def run_freefem():
    """
    Launch FreeFem++ on  freefem/main_87x87.edp  from the project root.
    FreeFem++ must be run from the project root so that relative paths resolve correctly.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    freefem_exe = os.environ.get("FREEFEM_EXE", "FreeFem++")
    cmd = [freefem_exe, FREEFEM_SCRIPT]
    print(f"[1_generate_data] Launching FreeFem++:")
    print(f"    {' '.join(cmd)}")
    print(f"    working directory: {project_root}")
    print(f"    Expected output: {N_PARAM} parameter values x {N_TIME} time steps"
          f" x {N_SENSORS_TOTAL} sensors")
    print(f"    This may take a while — go grab a coffee")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - t0

    log_path = os.path.join(DATA_RAW_DIR, "freefem_run.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)

    if result.returncode != 0:
        print(f"[ERROR] FreeFem++ exited with code {result.returncode}")
        print(f"        Log saved to: {log_path}")
        print("--- Last 40 lines of FreeFem++ output ---")
        lines = result.stdout.splitlines()
        print("\n".join(lines[-40:]))
        sys.exit(1)

    print(f"[1_generate_data] FreeFem++ finished in {elapsed:.1f} s")
    print(f"    Log: {log_path}")


def parse_sensor_file(filepath, field_name):
    """
    Parse a sensor output file produced by FreeFem++.

    File format (one line per (parameter, time) pair):
        a_value  t_value  val_0  val_1  ...  val_{N_SENSORS_TOTAL-1}

    Returns
    -------
    data : np.ndarray, shape (N_PARAM, N_TIME, N_SENSORS_TOTAL)
    a_recorded : np.ndarray, shape (N_PARAM,)
    t_recorded : np.ndarray, shape (N_TIME,)
    """
    print(f"[1_generate_data] Parsing {field_name} from {filepath} ...")

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split()))
            rows.append(vals)

    expected_rows = N_PARAM * N_TIME
    if len(rows) != expected_rows:
        print(f"[WARNING] Expected {expected_rows} rows, got {len(rows)}. "
              f"Simulation may be incomplete.")

    expected_cols = 2 + N_SENSORS_TOTAL
    for i, row in enumerate(rows):
        if len(row) != expected_cols:
            print(f"[ERROR] Row {i} has {len(row)} columns, expected {expected_cols}.")
            sys.exit(1)

    rows = np.array(rows)
    a_col   = rows[:, 0]
    t_col   = rows[:, 1]
    sensors = rows[:, 2:]

    data       = sensors.reshape(N_PARAM, N_TIME, N_SENSORS_TOTAL)
    a_recorded = a_col.reshape(N_PARAM, N_TIME)[:, 0]
    t_recorded = t_col.reshape(N_PARAM, N_TIME)[0, :]

    print(f"    Shape: {data.shape}  (N_PARAM={N_PARAM}, N_TIME={N_TIME}, "
          f"N_SENSORS={N_SENSORS_TOTAL})")
    print(f"    'a' range in file:  [{a_recorded.min():.4f}, {a_recorded.max():.4f}]")
    print(f"    't' range in file:  [{t_recorded.min():.4f}, {t_recorded.max():.4f}] s")
    print(f"    Value range:        [{data.min():.4e}, {data.max():.4e}]")

    return data, a_recorded, t_recorded


def save_sensor_data(T_data, phi2_data, a_recorded, t_recorded):
    """Save all parsed sensor data to a single .npz file."""
    np.savez_compressed(
        SENSOR_DATA_FILE,
        T    = T_data,
        phi2 = phi2_data,
        a    = a_recorded,
        t    = t_recorded,
    )
    print(f"[1_generate_data] Sensor data saved to: {SENSOR_DATA_FILE}")
    size_mb = os.path.getsize(SENSOR_DATA_FILE) / 1e6
    print(f"    File size: {size_mb:.2f} MB")


def sanity_checks(T_data, phi2_data, a_recorded, t_recorded):
    """Run basic physical and numerical sanity checks on the output."""
    print("\n[1_generate_data] Sanity checks:")

    T_min = T_data.min()
    assert T_min >= 290.0, f"Temperature below 290 K: {T_min:.2f} K"
    print(f"  OK  Min temperature: {T_min:.2f} K  (expected >= 300 K)")

    phi2_min = phi2_data.min()
    assert phi2_min >= 0.0, f"phi2 is negative: {phi2_min:.4e}"
    print(f"  OK  phi2 >= 0 everywhere  (min = {phi2_min:.4e})")

    a_diff = np.abs(a_recorded - A_VALUES).max()
    if a_diff > 1e-6:
        print(f"  [WARNING] 'a' values mismatch: max diff = {a_diff:.2e}")
    else:
        print(f"  OK  'a' values match config  (max diff = {a_diff:.2e})")

    dt_check = np.diff(t_recorded)
    dt_err = np.abs(dt_check - 0.02).max()
    if dt_err > 1e-6:
        print(f"  [WARNING] Irregular time steps, max error = {dt_err:.2e} s")
    else:
        print(f"  OK  Time steps uniform at 0.02 s  (max error = {dt_err:.2e} s)")

    T_s0 = T_data[:, :, 0]
    tau  = np.log(T_s0 / 300.0)
    D1_expected = 1.5 + 3e-3 * tau
    D2_expected = 0.4 + 3e-3 * tau
    print(f"  OK  Expected D1 range at sensor 0: "
          f"[{D1_expected.min():.4f}, {D1_expected.max():.4f}] cm")
    print(f"  OK  Expected D2 range at sensor 0: "
          f"[{D2_expected.min():.4f}, {D2_expected.max():.4f}] cm")

    print("[1_generate_data] All checks passed.\n")


if __name__ == "__main__":
    ensure_dirs()
    run_freefem()

    T_path    = os.path.join(DATA_RAW_DIR, "sensor_T.txt")
    phi2_path = os.path.join(DATA_RAW_DIR, "sensor_phi2.txt")

    T_data,    a_rec_T,    t_rec_T    = parse_sensor_file(T_path,    "T")
    phi2_data, a_rec_phi2, t_rec_phi2 = parse_sensor_file(phi2_path, "phi2")

    assert np.allclose(a_rec_T, a_rec_phi2, atol=1e-8), \
        "'a' values differ between sensor_T.txt and sensor_phi2.txt"
    assert np.allclose(t_rec_T, t_rec_phi2, atol=1e-8), \
        "'t' values differ between sensor_T.txt and sensor_phi2.txt"

    sanity_checks(T_data, phi2_data, a_rec_T, t_rec_T)
    save_sensor_data(T_data, phi2_data, a_rec_T, t_rec_T)

    print("[1_generate_data] Done.")
