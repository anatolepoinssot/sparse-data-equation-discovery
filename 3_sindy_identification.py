"""
3_sindy_identification.py
=========================
Use SINDy (Sparse Identification of Nonlinear Dynamics) to recover the
functional relation  D_g = f(tau)  where  tau = ln(T / T_ref).

Background
----------
SINDy builds a library of candidate functions:
    Theta(tau) = [tau, tau^2, ..., tau^p, sin(tau), cos(tau), tan(tau)]
and solves for a sparse coefficient vector xi such that:
    D_g ≈ D_g^0 + Theta(tau) . xi

The sparsity is enforced by Sequential Thresholded Least Squares (STLS):
    1. Solve min ||Theta.xi - (D_g - D_g^0)||
    2. Zero out coefficients below threshold lambda
    3. Re-solve on remaining terms
    4. Repeat until convergence

Expected result: xi[tau] = 3e-3 cm  (= gamma_D1 = gamma_D2)

Usage
-----
    python 3_sindy_identification.py
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
    SINDY_MAX_POLY_DEGREE, SINDY_TRIG_FUNCTIONS, SINDY_THRESHOLD,
    N_SENSORS_OBS, N_TIME, N_PARAM,
    OBS_SENSOR_INDICES,
    TRAINING_SET_FILE, SINDY_RESULTS_FILE, RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Library of candidate functions
# ---------------------------------------------------------------------------

def build_library(tau_flat, max_degree=5, include_trig=True):
    """
    Build the SINDy candidate function library evaluated on tau_flat.

    Parameters
    ----------
    tau_flat : np.ndarray, shape (N,)
    max_degree : int
    include_trig : bool

    Returns
    -------
    Theta : np.ndarray, shape (N, n_features)
    feature_names : list of str
    """
    cols  = []
    names = []

    # Polynomial terms: tau^1, tau^2, ..., tau^max_degree
    # No constant term (D_g^0 is already subtracted from the target)
    for d in range(1, max_degree + 1):
        cols.append(tau_flat ** d)
        names.append(f"tau^{d}" if d > 1 else "tau")

    # Trigonometric terms
    if include_trig:
        cols.append(np.sin(tau_flat));  names.append("sin(tau)")
        cols.append(np.cos(tau_flat));  names.append("cos(tau)")
        tan_tau = np.tan(np.clip(tau_flat, -1.4, 1.4))  # clip to avoid blow-up
        cols.append(tan_tau);           names.append("tan(tau)")

    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# Sequential Thresholded Least Squares (STLS)
# ---------------------------------------------------------------------------

def stls(Theta, target, threshold, max_iter=100):
    """
    Sequential Thresholded Least Squares.

    Solves  min ||Theta.xi - target||_2  with sparsity constraint.

    Parameters
    ----------
    Theta     : np.ndarray, shape (N, n_features)
    target    : np.ndarray, shape (N,)
    threshold : float  — coefficients below this are pruned to zero
    max_iter  : int

    Returns
    -------
    xi          : np.ndarray, shape (n_features,)
    active_mask : np.ndarray of bool, shape (n_features,)
    """
    n_features = Theta.shape[1]
    active = np.ones(n_features, dtype=bool)

    # Initial least-squares fit
    xi, _, _, _ = np.linalg.lstsq(Theta, target, rcond=None)

    for iteration in range(max_iter):
        small = np.abs(xi) < threshold
        xi[small]     = 0.0
        active[small] = False

        if not np.any(active):
            print(f"    [STLS] All coefficients thresholded out after {iteration+1} iter.")
            break

        xi_active, _, _, _ = np.linalg.lstsq(Theta[:, active], target, rcond=None)
        xi[active] = xi_active

        new_small = np.abs(xi) < threshold
        new_small[~active] = True
        if np.all(new_small == ~active):
            break

    return xi, active


# ---------------------------------------------------------------------------
# Identification for one field (D1 or D2)
# ---------------------------------------------------------------------------

def identify_field(tau_data, D_data, D_ref_data, field_name,
                   max_degree, include_trig, threshold):
    """
    Run SINDy to identify  D_g = D_g^0 + f(tau).

    Parameters
    ----------
    tau_data   : np.ndarray, shape (N_sensors_obs, N_time, N_param)
    D_data     : np.ndarray, same shape  — target D_g values
    D_ref_data : np.ndarray, shape (N_sensors_obs,)  — D_g^0 per sensor
    field_name : str  — 'D1' or 'D2'

    Returns
    -------
    xi, feature_names, active_mask, residual_rel
    """
    print(f"\n[3_sindy] === Identifying {field_name} ===")

    tau_flat = tau_data.ravel()
    D_flat   = D_data.ravel()

    D_ref_broadcast = D_ref_data[:, np.newaxis, np.newaxis] * np.ones_like(D_data)
    target = D_flat - D_ref_broadcast.ravel()   # D_g - D_g^0

    print(f"    Data points: {len(tau_flat)}")
    print(f"    tau range:   [{tau_flat.min():.4f}, {tau_flat.max():.4f}]")
    print(f"    target range (D_g - D_g0): [{target.min():.6f}, {target.max():.6f}]")

    Theta, names = build_library(tau_flat, max_degree=max_degree,
                                 include_trig=include_trig)
    print(f"    Library size: {len(names)} features: {names}")

    xi, active = stls(Theta, target, threshold=threshold)

    print(f"\n    SINDy result for {field_name}:")
    print(f"    {'Feature':<15}  {'Coefficient':>14}  {'Active?':>8}")
    print(f"    {'-'*42}")
    for name, coef, act in zip(names, xi, active):
        marker = "  <- ACTIVE" if act else ""
        print(f"    {name:<15}  {coef:>14.6e}  {str(act):>8}{marker}")

    D_approx = D_ref_broadcast.ravel() + Theta @ xi
    residual_rel = np.linalg.norm(D_flat - D_approx) / np.linalg.norm(D_flat)
    print(f"\n    Relative L2 residual: {residual_rel:.4e}")

    return xi, names, active, residual_rel


# ---------------------------------------------------------------------------
# Threshold / degree sweep
# ---------------------------------------------------------------------------

def threshold_sweep(tau_data, D_data, D_ref_data, field_name,
                    thresholds, max_degrees, include_trig):
    """Run SINDy for a grid of (threshold, max_degree) and report results."""
    print(f"\n[3_sindy] Threshold / degree sweep for {field_name} ...")
    true_gamma = GAMMA_D1 if field_name == "D1" else GAMMA_D2

    results = []
    tau_flat = tau_data.ravel()
    D_flat   = D_data.ravel()
    D_ref_broadcast = D_ref_data[:, np.newaxis, np.newaxis] * np.ones_like(D_data)
    target   = D_flat - D_ref_broadcast.ravel()

    for deg in max_degrees:
        Theta, names = build_library(tau_flat, max_degree=deg,
                                     include_trig=include_trig)
        for lam in thresholds:
            xi, active = stls(Theta, target, threshold=lam)
            D_approx = D_ref_broadcast.ravel() + Theta @ xi
            rel_err  = np.linalg.norm(D_flat - D_approx) / np.linalg.norm(D_flat)
            n_active = int(active.sum())
            tau_idx  = 0
            gamma_recovered = xi[tau_idx]
            gamma_err       = abs(gamma_recovered - true_gamma)
            results.append({
                "field": field_name, "max_degree": deg, "threshold": lam,
                "n_active": n_active, "rel_err": rel_err,
                "gamma_rec": gamma_recovered, "gamma_err": gamma_err,
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load training set
    print(f"[3_sindy] Loading training set from: {TRAINING_SET_FILE}")
    data = np.load(TRAINING_SET_FILE)

    tau_obs  = data["tau_obs"]   # (7, 100, 200)
    D1_obs   = data["D1_obs"]    # (7, 100, 200)
    D2_obs   = data["D2_obs"]    # (7, 100, 200)
    obs_idx  = data["obs_idx"]
    D1_refs  = data["D1_refs"][obs_idx]
    D2_refs  = data["D2_refs"][obs_idx]
    a_vals   = data["a_vals"]
    t_vals   = data["t_vals"]

    print(f"    tau_obs shape: {tau_obs.shape}")
    print(f"    Observed sensor indices: {obs_idx.tolist()}")

    # 2. Full SINDy run
    xi_D1, names, active_D1, res_D1 = identify_field(
        tau_obs, D1_obs, D1_refs, "D1",
        SINDY_MAX_POLY_DEGREE, SINDY_TRIG_FUNCTIONS, SINDY_THRESHOLD,
    )
    xi_D2, names, active_D2, res_D2 = identify_field(
        tau_obs, D2_obs, D2_refs, "D2",
        SINDY_MAX_POLY_DEGREE, SINDY_TRIG_FUNCTIONS, SINDY_THRESHOLD,
    )

    # 3. Summary
    tau_idx = names.index("tau")
    print("\n" + "="*55)
    print("  RECOVERY SUMMARY")
    print("="*55)
    print(f"  True gamma_D1 = {GAMMA_D1:.4e}  cm")
    print(f"  True gamma_D2 = {GAMMA_D2:.4e}  cm")
    print(f"  Recovered gamma_D1 = {xi_D1[tau_idx]:.6e}  cm  "
          f"(error = {abs(xi_D1[tau_idx]-GAMMA_D1)/GAMMA_D1*100:.2f} %)")
    print(f"  Recovered gamma_D2 = {xi_D2[tau_idx]:.6e}  cm  "
          f"(error = {abs(xi_D2[tau_idx]-GAMMA_D2)/GAMMA_D2*100:.2f} %)")
    print(f"  Relative L2 residual D1: {res_D1:.4e}")
    print(f"  Relative L2 residual D2: {res_D2:.4e}")
    print("="*55)

    # 4. Threshold / degree sweep
    thresholds  = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
    max_degrees = [1, 2, 3, 4, 5]
    sweep_D1 = threshold_sweep(tau_obs, D1_obs, D1_refs, "D1",
                                thresholds, max_degrees, SINDY_TRIG_FUNCTIONS)
    sweep_D2 = threshold_sweep(tau_obs, D2_obs, D2_refs, "D2",
                                thresholds, max_degrees, SINDY_TRIG_FUNCTIONS)

    print("\n[3_sindy] Sweep results (D1):")
    print(f"  {'deg':>4}  {'lambda':>8}  {'n_active':>8}  "
          f"{'rel_err':>10}  {'gamma_rec':>12}  {'gamma_err':>12}")
    for r in sweep_D1:
        print(f"  {r['max_degree']:>4}  {r['threshold']:>8.1e}  {r['n_active']:>8}  "
              f"  {r['rel_err']:>10.4e}  {r['gamma_rec']:>12.6e}  {r['gamma_err']:>12.4e}")

    # 5. Save results
    np.savez_compressed(
        SINDY_RESULTS_FILE,
        xi_D1         = xi_D1,
        xi_D2         = xi_D2,
        active_D1     = active_D1,
        active_D2     = active_D2,
        feature_names = np.array(names),
        gamma_D1_rec  = np.array([xi_D1[tau_idx]]),
        gamma_D2_rec  = np.array([xi_D2[tau_idx]]),
        res_D1        = np.array([res_D1]),
        res_D2        = np.array([res_D2]),
    )
    print(f"\n[3_sindy] Results saved to: {SINDY_RESULTS_FILE}")

    # 6. Plots
    _plot_library_fit(tau_obs, D1_obs, D1_refs, xi_D1, names, "D1")
    _plot_library_fit(tau_obs, D2_obs, D2_refs, xi_D2, names, "D2")
    _plot_sweep(sweep_D1, "D1", GAMMA_D1)
    _plot_sweep(sweep_D2, "D2", GAMMA_D2)

    print("[3_sindy] Done.")


def _plot_library_fit(tau_obs, D_obs, D_refs, xi, names, field_name):
    """Plot D_g vs tau with the SINDy fit overlaid."""
    tau_flat = tau_obs.ravel()
    D_flat   = D_obs.ravel()
    D_ref_broadcast = D_refs[:, np.newaxis, np.newaxis] * np.ones_like(D_obs)
    Theta, _ = build_library(tau_flat, SINDY_MAX_POLY_DEGREE, SINDY_TRIG_FUNCTIONS)
    D_sindy  = D_ref_broadcast.ravel() + Theta @ xi

    tau_range  = np.linspace(tau_flat.min(), tau_flat.max(), 300)
    ref_val    = D_refs[0]
    true_gamma = GAMMA_D1 if field_name == "D1" else GAMMA_D2
    D_true     = ref_val + true_gamma * tau_range

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau_flat, D_flat,  s=1, alpha=0.2, label="Ground truth", color="steelblue")
    ax.scatter(tau_flat, D_sindy, s=1, alpha=0.2, label="SINDy fit",    color="orange")
    ax.plot(tau_range, D_true, "r-", lw=2,
            label=f"True: {ref_val:.2f} + {true_gamma:.3e}*tau")
    ax.set_xlabel("tau = ln(T / T_ref)")
    ax.set_ylabel(f"{field_name} [cm]")
    ax.set_title(f"SINDy identification of {field_name}(tau)")
    ax.legend(markerscale=8)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"sindy_fit_{field_name}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"    [plot] {out}")


def _plot_sweep(sweep_results, field_name, true_gamma):
    """Plot recovered gamma vs threshold for each polynomial degree."""
    degrees = sorted(set(r["max_degree"] for r in sweep_results))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(degrees)))
    for deg, col in zip(degrees, colors):
        subset   = [r for r in sweep_results if r["max_degree"] == deg]
        lams     = [r["threshold"] for r in subset]
        gammas   = [r["gamma_rec"] for r in subset]
        rel_errs = [r["rel_err"]   for r in subset]
        ax1.semilogx(lams, gammas, "o-", color=col, label=f"degree {deg}")
        ax2.loglog(lams, rel_errs, "o-",  color=col, label=f"degree {deg}")
    ax1.axhline(true_gamma, color="red", linestyle="--", lw=2, label="True gamma")
    ax1.set_xlabel("Threshold lambda")
    ax1.set_ylabel("Recovered gamma [cm]")
    ax1.set_title(f"{field_name}: Recovered gamma vs threshold")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Threshold lambda")
    ax2.set_ylabel("Relative L2 residual")
    ax2.set_title(f"{field_name}: Fit quality vs threshold")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"sindy_sweep_{field_name}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"    [plot] {out}")


if __name__ == "__main__":
    main()
