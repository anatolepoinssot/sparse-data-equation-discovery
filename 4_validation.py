"""
4_validation.py
===============
Validate the equation recovered by SINDy against held-out sensor data.

What this script does
---------------------
1. Load the SINDy results (recovered coefficients xi_D1, xi_D2).
2. Load the full sensor dataset (all 200 parameter values, all 20 sensors).
3. Reconstruct D_g using the recovered relation and compare to ground truth.
4. Produce summary statistics and validation plots.

Usage
-----
    python 4_validation.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    T_REF, GAMMA_D1, GAMMA_D2,
    D1_REF_BY_REGION, D2_REF_BY_REGION,
    SINDY_MAX_POLY_DEGREE, SINDY_TRIG_FUNCTIONS,
    N_SENSORS_TOTAL, N_TIME, N_PARAM,
    OBS_SENSOR_INDICES,
    TRAINING_SET_FILE, SINDY_RESULTS_FILE, RESULTS_DIR,
)


def build_library(tau_flat, max_degree=5, include_trig=True):
    """Candidate function library (identical to script 3 — avoids circular import)."""
    cols, names = [], []
    for d in range(1, max_degree + 1):
        cols.append(tau_flat ** d)
        names.append(f"tau^{d}" if d > 1 else "tau")
    if include_trig:
        cols.append(np.sin(tau_flat));  names.append("sin(tau)")
        cols.append(np.cos(tau_flat));  names.append("cos(tau)")
        tan_tau = np.tan(np.clip(tau_flat, -1.4, 1.4))
        cols.append(tan_tau);           names.append("tan(tau)")
    return np.column_stack(cols), names


def r_squared(y_true, y_pred):
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load SINDy results
    print(f"[4_validation] Loading SINDy results from: {SINDY_RESULTS_FILE}")
    sindy = np.load(SINDY_RESULTS_FILE, allow_pickle=True)

    xi_D1         = sindy["xi_D1"]
    xi_D2         = sindy["xi_D2"]
    active_D1     = sindy["active_D1"]
    active_D2     = sindy["active_D2"]
    feature_names = sindy["feature_names"].tolist()
    gamma_D1_rec  = float(sindy["gamma_D1_rec"])
    gamma_D2_rec  = float(sindy["gamma_D2_rec"])

    print(f"    Feature names: {feature_names}")
    print(f"    xi_D1 (active): " +
          "  ".join(f"{n}={v:.4e}" for n, v, a in
                    zip(feature_names, xi_D1, active_D1) if a))
    print(f"    xi_D2 (active): " +
          "  ".join(f"{n}={v:.4e}" for n, v, a in
                    zip(feature_names, xi_D2, active_D2) if a))

    # 2. Load full sensor dataset
    print(f"\n[4_validation] Loading full sensor dataset from: {TRAINING_SET_FILE}")
    data = np.load(TRAINING_SET_FILE)

    T_all   = data["T_all"]    # (N_param, N_time, N_sensors_total)
    D1_all  = data["D1_all"]
    D2_all  = data["D2_all"]
    tau_all = data["tau_all"]
    D1_refs = data["D1_refs"]
    D2_refs = data["D2_refs"]
    a_vals  = data["a_vals"]
    t_vals  = data["t_vals"]
    obs_idx = data["obs_idx"].tolist()

    # 3. Reconstruct D_g using recovered SINDy relation
    print("[4_validation] Reconstructing D_g from SINDy relation ...")
    tau_flat = tau_all.ravel()
    Theta, _ = build_library(tau_flat, SINDY_MAX_POLY_DEGREE, SINDY_TRIG_FUNCTIONS)

    D1_ref_broadcast = D1_refs[np.newaxis, np.newaxis, :] * np.ones_like(tau_all)
    D2_ref_broadcast = D2_refs[np.newaxis, np.newaxis, :] * np.ones_like(tau_all)

    D1_sindy = (D1_ref_broadcast.ravel() + Theta @ xi_D1).reshape(tau_all.shape)
    D2_sindy = (D2_ref_broadcast.ravel() + Theta @ xi_D2).reshape(tau_all.shape)

    # 4. Error metrics
    def rel_error(true, pred):
        return np.abs(true - pred) / (np.abs(true) + 1e-30)

    err_D1 = rel_error(D1_all, D1_sindy)
    err_D2 = rel_error(D2_all, D2_sindy)
    r2_D1  = r_squared(D1_all.ravel(), D1_sindy.ravel())
    r2_D2  = r_squared(D2_all.ravel(), D2_sindy.ravel())

    print("\n" + "="*60)
    print("  VALIDATION SUMMARY — ALL 20 SENSORS, ALL PARAMS & TIMES")
    print("="*60)
    print(f"  D1:  max rel error = {err_D1.max():.4e}   "
          f"mean = {err_D1.mean():.4e}   R² = {r2_D1:.8f}")
    print(f"  D2:  max rel error = {err_D2.max():.4e}   "
          f"mean = {err_D2.mean():.4e}   R² = {r2_D2:.8f}")
    print()
    print(f"  True   gamma_D1 = {GAMMA_D1:.6e}  cm")
    print(f"  Recov. gamma_D1 = {gamma_D1_rec:.6e}  cm  "
          f"(rel. error = {abs(gamma_D1_rec-GAMMA_D1)/GAMMA_D1*100:.4f} %)")
    print(f"  True   gamma_D2 = {GAMMA_D2:.6e}  cm")
    print(f"  Recov. gamma_D2 = {gamma_D2_rec:.6e}  cm  "
          f"(rel. error = {abs(gamma_D2_rec-GAMMA_D2)/GAMMA_D2*100:.4f} %)")
    print("="*60)

    # 5. Validation plots
    tau_flat2 = tau_all.ravel()

    # Plot 1: True vs predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, D_true, D_pred, name, color in [
        (axes[0], D1_all.ravel(), D1_sindy.ravel(), "D1", "steelblue"),
        (axes[1], D2_all.ravel(), D2_sindy.ravel(), "D2", "darkorange"),
    ]:
        ax.scatter(D_true, D_pred, s=1, alpha=0.15, color=color)
        lims = [min(D_true.min(), D_pred.min()), max(D_true.max(), D_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
        ax.set_xlabel(f"{name} ground truth [cm]")
        ax.set_ylabel(f"{name} SINDy prediction [cm]")
        ax.set_title(f"{name}: True vs Predicted  (R²={r_squared(D_true, D_pred):.8f})")
        ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "val_true_vs_pred.png")
    plt.savefig(out, dpi=120);  plt.close()
    print(f"    [plot] {out}")

    # Plot 2: Residuals vs tau
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, D_true, D_sindy_v, name in [
        (axes[0], D1_all.ravel(), D1_sindy.ravel(), "D1"),
        (axes[1], D2_all.ravel(), D2_sindy.ravel(), "D2"),
    ]:
        residual = D_true - D_sindy_v
        ax.scatter(tau_flat2, residual, s=1, alpha=0.1)
        ax.axhline(0, color="red", lw=1.5, linestyle="--")
        ax.set_xlabel("tau = ln(T / T_ref)")
        ax.set_ylabel(f"Residual {name} [cm]")
        ax.set_title(f"{name} residuals  (max |res| = {np.abs(residual).max():.3e})")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "val_residuals.png")
    plt.savefig(out, dpi=120);  plt.close()
    print(f"    [plot] {out}")

    # Plot 3: Error vs parameter a
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, err, name in [
        (axes[0], err_D1, "D1"),
        (axes[1], err_D2, "D2"),
    ]:
        mean_err_per_a = err.mean(axis=(1, 2))
        ax.plot(a_vals, mean_err_per_a)
        ax.set_xlabel("a  (reactivity parameter)")
        ax.set_ylabel(f"Mean rel. error on {name}")
        ax.set_title(f"{name}: Error vs parameter a")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "val_error_vs_a.png")
    plt.savefig(out, dpi=120);  plt.close()
    print(f"    [plot] {out}")

    # Plot 4: D_g time evolution at one sensor
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, D_true, D_pred, name in [
        (axes[0], D1_all, D1_sindy, "D1"),
        (axes[1], D2_all, D2_sindy, "D2"),
    ]:
        s_idx, a_idx = 0, 99
        ax.plot(t_vals, D_true[a_idx, :, s_idx], "b-",  lw=2, label="Ground truth")
        ax.plot(t_vals, D_pred[a_idx, :, s_idx], "r--", lw=2, label="SINDy")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"{name} [cm]")
        ax.set_title(f"{name} time evolution — sensor 0, a={a_vals[a_idx]:.3f}")
        ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "val_time_evolution.png")
    plt.savefig(out, dpi=120);  plt.close()
    print(f"    [plot] {out}")

    print("\n[4_validation] Done.")


if __name__ == "__main__":
    main()
