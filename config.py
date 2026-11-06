"""
config.py
=========
Central configuration file for the sparse-data equation discovery project.
All physical parameters, grid settings, sensor positions, and SINDy options
are defined here so that every other script can import from a single source.

Physical context
----------------
IAEA 2D PWR benchmark — neutronics / thermal-hydraulics coupling.
True relation to recover (eq. 9 of the GEIM paper):
    D_g(x, t) = D_g^0(x) + gamma_D_g * ln(T(x, t) / T_ref)
with gamma_D1 = gamma_D2 = 3e-3 cm,  T_ref = 300 K.
"""

import numpy as np

# =============================================================================
# PHYSICAL CONSTANTS  (from parameter.edp)
# =============================================================================

T_REF = 300.0          # Reference temperature [K]
GAMMA_D1 = 3e-3        # True coupling coefficient for D1 [cm]
GAMMA_D2 = 3e-3        # True coupling coefficient for D2 [cm]

# D_g^0 per region (initial steady-state values from parameter.edp, before
# thermal feedback).  These are treated as KNOWN in the identification step.
#   Region labels: fuel1, fuel2, cr (control rods), ref (reflector)
D1_REF_BY_REGION = {
    "fuel1": 1.5,
    "fuel2": 1.5,
    "cr":    1.5,
    "ref":   2.0,
}
D2_REF_BY_REGION = {
    "fuel1": 0.4,
    "fuel2": 0.4,
    "cr":    0.4,
    "ref":   0.3,
}

# =============================================================================
# GEOMETRY  (from geometry.edp)
# =============================================================================

DOMAIN_SIZE   = 170.0  # [cm]  — square domain [0, 170] x [0, 170]

# Active core boundary (fuel regions, excluding reflector which starts at 150 cm)
CORE_MIN = 10.0        # [cm]
CORE_MAX = 130.0       # [cm]

# =============================================================================
# OUTPUT GRID  (87 x 87 uniform grid sampled from FreeFem++ solution)
# =============================================================================

NX = 87                # Number of grid points along x
NY = 87                # Number of grid points along y
HX = DOMAIN_SIZE / (NX - 1)   # ~1.977 cm
HY = DOMAIN_SIZE / (NY - 1)   # ~1.977 cm

# 1D arrays of grid coordinates
X_GRID = np.linspace(0, DOMAIN_SIZE, NX)
Y_GRID = np.linspace(0, DOMAIN_SIZE, NY)

# =============================================================================
# TEMPORAL DISCRETISATION
# =============================================================================

DT_FREEFEM  = 0.01     # [s]  FreeFem++ internal time step
T_END       = 2.0      # [s]  End of transient (reduced from 5 s to save compute)
DT_OUTPUT   = 0.02     # [s]  Output every other FreeFem++ step  -> 100 snapshots
N_TIME      = int(round(T_END / DT_OUTPUT))   # 100 time points

# =============================================================================
# PARAMETRIC STUDY  —  absorption cross-section reduction factor 'a'
# =============================================================================
# 'a' multiplies Sigma_a in the control-rod region (region 3).
# a = 1  -> no perturbation;  a < 1 -> reactivity insertion.
# We span a physically meaningful range around the nominal value a = 0.9.

N_PARAM      = 200                          # Number of parameter samples
A_MIN        = 0.880                        # Lower bound of 'a'
A_MAX        = 0.920                        # Upper bound of 'a'
A_VALUES     = np.linspace(A_MIN, A_MAX, N_PARAM)   # 200 uniform samples

# =============================================================================
# SENSOR PLACEMENT  —  20 sensors on the 87x87 grid near the core centre
# =============================================================================
# We place sensors on a regular 4x5 sub-grid inside the active core
# (between SENSOR_X_MIN and SENSOR_X_MAX in both directions).
# This avoids the reflector region and boundary cells.

N_SENSORS_TOTAL = 20   # Total number of virtual sensors
N_SENSORS_OBS   = 7    # Number of sensors actually observed  (approx 1/3 of 20)

# Grid indices for sensor placement: 4 columns x 5 rows inside the core
# Core centre region: x,y in [30, 110] cm  -> indices on the 87x87 grid
SENSOR_X_CM = np.linspace(30.0, 110.0, 4)   # 4 x-positions [cm]
SENSOR_Y_CM = np.linspace(30.0, 110.0, 5)   # 5 y-positions [cm]

def _cm_to_idx(coord_cm, grid_1d):
    """Return the nearest grid index for a given coordinate in cm."""
    return int(np.argmin(np.abs(grid_1d - coord_cm)))

# Build the list of (ix, iy) index pairs for all 20 sensors
SENSOR_INDICES = []
for sx in SENSOR_X_CM:
    for sy in SENSOR_Y_CM:
        ix = _cm_to_idx(sx, X_GRID)
        iy = _cm_to_idx(sy, Y_GRID)
        SENSOR_INDICES.append((ix, iy))

# Coordinates [cm] of all 20 sensors (for reference / plotting)
SENSOR_COORDS_CM = [
    (X_GRID[ix], Y_GRID[iy]) for (ix, iy) in SENSOR_INDICES
]

# Reproducible random selection of the 7 observed sensors
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
_perm = rng.permutation(N_SENSORS_TOTAL)
OBS_SENSOR_INDICES = sorted(_perm[:N_SENSORS_OBS].tolist())   # indices into SENSOR_INDICES

# =============================================================================
# SINDy  —  candidate function library
# =============================================================================
# We search for D_g = D_g^0 + f(tau)  where  tau = ln(T / T_ref).
# The library contains monomials up to MAX_POLY_DEGREE plus trigonometric terms.

SINDY_MAX_POLY_DEGREE = 5    # Test polynomials: tau^1, tau^2, ..., tau^5
SINDY_TRIG_FUNCTIONS  = True # Include sin(tau), cos(tau), tan(tau)
SINDY_THRESHOLD       = 1e-3 # Sparsity threshold for sequential thresholded LS

# =============================================================================
# FILE PATHS
# =============================================================================

import os
_BASE = os.path.dirname(os.path.abspath(__file__))

FREEFEM_SCRIPT   = os.path.join(_BASE, "freefem", "main_87x87.edp")
DATA_RAW_DIR     = os.path.join(_BASE, "data", "raw")
DATA_PROC_DIR    = os.path.join(_BASE, "data", "processed")
RESULTS_DIR      = os.path.join(_BASE, "results")

# Raw sensor data file (written by script 1, read by script 2)
SENSOR_DATA_FILE = os.path.join(DATA_RAW_DIR, "sensor_data.npz")

# Processed training set
TRAINING_SET_FILE = os.path.join(DATA_PROC_DIR, "training_set_Y.npz")

# SINDy results
SINDY_RESULTS_FILE = os.path.join(RESULTS_DIR, "sindy_results.npz")


# =============================================================================
# QUICK SANITY CHECKS  (run when the module is imported)
# =============================================================================

assert len(SENSOR_INDICES) == N_SENSORS_TOTAL, \
    f"Expected {N_SENSORS_TOTAL} sensors, got {len(SENSOR_INDICES)}"
assert len(OBS_SENSOR_INDICES) == N_SENSORS_OBS, \
    f"Expected {N_SENSORS_OBS} observed sensors, got {len(OBS_SENSOR_INDICES)}"
assert N_TIME == 100, \
    f"Expected 100 time snapshots, got {N_TIME}"
