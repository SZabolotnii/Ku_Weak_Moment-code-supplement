"""Real-data illustration: Belgium telephone-calls benchmark (Rousseeuw & Leroy 1987).

Regresses annual call totals on year. The 1964-1969 block was recorded in a
different unit, giving six gross vertical outliers (a pure error-contamination
case with no x-leverage). We fit the full estimator panel, sweep the weak-PMM
window/scale, and report point estimates, sandwich/bootstrap inference, the
recovered slope, and which observations each fit flags as outliers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ku_weak_moment.estimators import (
    cauchy_mle_regression, huber_regression, lad_regression,
    mm_estimator_regression, ols_regression, s_estimator_regression,
    tukey_biweight_regression, weak_cf_regression, weak_pmm3_regression,
    weak_pmm_regression,
)
from ku_weak_moment.inference import bootstrap_ci, weak_pmm1_sandwich
from ku_weak_moment.simulation import start_manifest, write_manifest
from ku_weak_moment.windows import robust_scale_mad

# The degree-1 weak-PMM member used for sandwich/bootstrap inference and as the
# residual reference: Cauchy-like window at one robust scale (matched MLE form).
REF_WINDOW = "cauchy_like"
REF_SCALE_MULT = 1.0


def fitted(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta[0] + beta[1] * x


def flag_outliers(r: np.ndarray, thresh: float = 2.5) -> np.ndarray:
    """Outlier mask from standardized residuals using a robust residual scale."""
    s = robust_scale_mad(r)
    if s <= 0.0:
        s = float(np.std(r)) or 1.0
    return np.abs(r) > thresh * s


def main(out_dir: Path) -> None:
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(Path(__file__).parent / cfg["data_file"])
    year = data[cfg["predictor"]].to_numpy(float)
    y = data[cfg["response"]].to_numpy(float)
    x = year - year.mean() if cfg.get("center_year", True) else year.copy()

    manifest = start_manifest("real_data_phones", cfg, str(REPO))

    # Robust scale from a LAD start drives the Cauchy gamma and the window scale.
    b_lad = lad_regression(x, y).beta_hat
    sc = robust_scale_mad(y - fitted(b_lad, x))
    if sc <= 0.0:
        sc = 1.0
    gamma = sc

    rows = []

    def record(name, fam, sm, est, t):
        b = np.asarray(est.beta_hat, float)
        rows.append({
            "estimator": name, "window_family": fam, "scale_mult": sm,
            "b0_hat": float(b[0]), "b1_hat": float(b[1]),
            "runtime_sec": float(t), "success": bool(est.success),
            "n_iter": int(est.n_iter), "objective_value": float(est.objective_value),
        })

    panel = [
        ("ols", "-", float("nan"), lambda: ols_regression(x, y)),
        ("lad", "-", float("nan"), lambda: lad_regression(x, y)),
        ("huber", "huber_1.345_MAD", float("nan"), lambda: huber_regression(x, y)),
        ("tukey_biweight", "tukey_compact_4.685_MAD", float("nan"),
         lambda: tukey_biweight_regression(x, y)),
        ("s_estimator", "s_tukey", float("nan"),
         lambda: s_estimator_regression(x, y, seed=cfg["bootstrap"]["seed"])),
        ("mm_estimator", "mm_tukey", float("nan"),
         lambda: mm_estimator_regression(x, y, seed=cfg["bootstrap"]["seed"])),
        ("cauchy_mle_plugin", "cauchy_mle", float(gamma),
         lambda: cauchy_mle_regression(x, y, gamma, beta0=b_lad)),
        ("weak_cf", "cf_symmetric", float("nan"),
         lambda: weak_cf_regression(x, y)),
    ]
    for name, fam, sm, fn in panel:
        t0 = time.perf_counter()
        est = fn()
        record(name, fam, sm, est, time.perf_counter() - t0)

    # weak-PMM degree 1 and 3 window/scale sweep
    for w in cfg["windows"]:
        for mult in w["scale_mults"]:
            scale = mult * sc
            t0 = time.perf_counter()
            est1 = weak_pmm_regression(x, y, w["family"], scale, beta0=b_lad)
            record(f"wpmm1_{w['family']}", w["family"], float(mult), est1,
                   time.perf_counter() - t0)
            t0 = time.perf_counter()
            est3 = weak_pmm3_regression(x, y, w["family"], scale, beta0=b_lad)
            record(f"wpmm3_{w['family']}", w["family"], float(mult), est3,
                   time.perf_counter() - t0)

    estimates = pd.DataFrame(rows)
    estimates["slope_per_year"] = estimates["b1_hat"]  # x is centered, slope unchanged
    estimates.to_csv(out_dir / "estimates.csv", index=False)

    # ---- inference for the reference degree-1 weak-PMM member and OLS/MM ----
    ref_scale = REF_SCALE_MULT * sc
    beta_ref, _, se_ref = weak_pmm1_sandwich(x, y, REF_WINDOW, ref_scale)

    def fit_wpmm1(xx, yy):
        b0 = lad_regression(xx, yy).beta_hat
        return weak_pmm_regression(xx, yy, REF_WINDOW, ref_scale, beta0=b0).beta_hat

    bcfg = cfg["bootstrap"]
    lo_w, hi_w, se_w = bootstrap_ci(x, y, fit_wpmm1, level=bcfg["level"],
                                    n_boot=bcfg["n_boot"], seed=bcfg["seed"])
    lo_o, hi_o, se_o = bootstrap_ci(
        x, y, lambda xx, yy: ols_regression(xx, yy).beta_hat,
        level=bcfg["level"], n_boot=bcfg["n_boot"], seed=bcfg["seed"])

    inference = pd.DataFrame([
        {"estimator": f"wpmm1_{REF_WINDOW}", "coef": "intercept",
         "estimate": float(beta_ref[0]), "sandwich_se": float(se_ref[0]),
         "boot_se": float(se_w[0]), "boot_lo": float(lo_w[0]), "boot_hi": float(hi_w[0])},
        {"estimator": f"wpmm1_{REF_WINDOW}", "coef": "slope",
         "estimate": float(beta_ref[1]), "sandwich_se": float(se_ref[1]),
         "boot_se": float(se_w[1]), "boot_lo": float(lo_w[1]), "boot_hi": float(hi_w[1])},
        {"estimator": "ols", "coef": "intercept",
         "estimate": float(rows[0]["b0_hat"]), "sandwich_se": float("nan"),
         "boot_se": float(se_o[0]), "boot_lo": float(lo_o[0]), "boot_hi": float(hi_o[0])},
        {"estimator": "ols", "coef": "slope",
         "estimate": float(rows[0]["b1_hat"]), "sandwich_se": float("nan"),
         "boot_se": float(se_o[1]), "boot_lo": float(lo_o[1]), "boot_hi": float(hi_o[1])},
    ])
    inference.to_csv(out_dir / "inference.csv", index=False)

    # ---- residual diagnostics: robust (weak-PMM-1) reference vs OLS ----
    b_ols = np.array([rows[0]["b0_hat"], rows[0]["b1_hat"]])
    r_ref = y - fitted(beta_ref, x)
    r_ols = y - fitted(b_ols, x)
    resid = pd.DataFrame({
        "year": year.astype(int), "calls": y,
        "fitted_wpmm1": fitted(beta_ref, x), "resid_wpmm1": r_ref,
        "fitted_ols": fitted(b_ols, x), "resid_ols": r_ols,
        "flagged_wpmm1": flag_outliers(r_ref),
        "flagged_ols": flag_outliers(r_ols),
        "known_outlier": np.isin(year.astype(int), cfg["known_outlier_years"]),
    })
    resid.to_csv(out_dir / "residuals.csv", index=False)

    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)

    print(f"OLS slope        = {b_ols[1]:.3f}")
    print(f"weak-PMM-1 slope = {beta_ref[1]:.3f}  (sandwich SE {se_ref[1]:.3f})")
    print(f"flagged by weak-PMM-1: "
          f"{sorted(resid.loc[resid.flagged_wpmm1, 'year'].tolist())}")
    print(f"flagged by OLS:        "
          f"{sorted(resid.loc[resid.flagged_ols, 'year'].tolist())}")
    print(f"wrote estimates/inference/residuals to {out_dir}")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
