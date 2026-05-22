"""Phase 2a: design-stress Monte Carlo (heavy-tailed X, leverage, heteroskedasticity).

Tests whether weak-moment robustness / interchangeability survive when the
*design* is adversarial, with high-breakdown S/MM as the reference baselines.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ku_weak_moment.estimators import (  # noqa: E402
    cauchy_mle_regression, huber_regression, lad_regression, mm_estimator_regression,
    ols_regression, s_estimator_regression, tukey_biweight_regression,
    weak_cf_regression, weak_pmm_regression, weak_pmm3_regression,
)
from ku_weak_moment.simulation import (  # noqa: E402
    sample_contaminated_gaussian, start_manifest, write_manifest,
)
from ku_weak_moment.windows import robust_scale_mad  # noqa: E402

BETA = np.array([0.5, 1.0])


def gen_x(rng, design, n):
    k = design["kind"]
    if k == "uniform":
        return rng.uniform(-1.0, 1.0, n)
    if k == "heavy_x":
        return rng.standard_t(design["df"], size=n)
    if k == "leverage":
        x = rng.uniform(-1.0, 1.0, n)
        m = rng.random(n) < design["frac"]
        x[m] = design["lev"] * rng.choice([-1.0, 1.0], size=int(m.sum()))
        return x
    if k == "hetero":
        return rng.uniform(-1.0, 1.0, n)
    raise ValueError(k)


def gen_noise(rng, noise, n):
    k = noise["kind"]
    if k == "cauchy":
        return noise["gamma"] * rng.standard_cauchy(n)
    if k == "contam":
        return sample_contaminated_gaussian(rng, n, noise["eps"], noise["scale_contam"])
    raise ValueError(k)


def run_panel(x, y, fam, sm):
    b_lad = lad_regression(x, y).beta_hat
    sc = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
    if sc <= 0.0:
        sc = 1.0
    scale = sm * sc
    out = {
        "ols": ols_regression(x, y),
        "lad": lad_regression(x, y),
        "huber": huber_regression(x, y),
        "tukey": tukey_biweight_regression(x, y),
        "s_est": s_estimator_regression(x, y, n_resamples=40, seed=0),
        "mm_est": mm_estimator_regression(x, y, n_resamples=40, seed=0),
        "cauchy_mle": cauchy_mle_regression(x, y, 1.0, beta0=b_lad),
        "wpmm1_gauss": weak_pmm_regression(x, y, fam, scale, beta0=b_lad),
        "wpmm3_gauss": weak_pmm3_regression(x, y, fam, scale, beta0=b_lad),
        "weak_cf": weak_cf_regression(x, y, beta0=b_lad),
    }
    return out


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = start_manifest("design_stress", cfg, str(REPO))
    fam = cfg["window_family"]
    sm = float(cfg["scale_mult"])
    R = int(cfg["replications"])
    rows = []
    for design in cfg["designs"]:
        for noise in cfg["noises"]:
            for n in cfg["sample_sizes"]:
                t0 = time.perf_counter()
                hetero = design["kind"] == "hetero"
                for r_idx in range(R):
                    rng = np.random.default_rng(cfg["base_seed"] + 19 * n
                                                + 7919 * r_idx)
                    x = gen_x(rng, design, n)
                    eps = gen_noise(rng, noise, n)
                    if hetero:
                        eps = eps * (1.0 + design["slope"] * np.abs(x))
                    y = BETA[0] + BETA[1] * x + eps
                    for name, est in run_panel(x, y, fam, sm).items():
                        e0 = float(est.beta_hat[0]) - BETA[0]
                        e1 = float(est.beta_hat[1]) - BETA[1]
                        rows.append({
                            "design": design["name"], "noise": noise["name"],
                            "n": n, "rep": r_idx, "estimator": name,
                            "err0": e0, "err1": e1,
                            "err_linf": max(abs(e0), abs(e1)),
                        })
                print(f"  {design['name']}/{noise['name']} n={n}: "
                      f"{time.perf_counter()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "results.parquet", index=False)
    summary = (
        df.groupby(["design", "noise", "n", "estimator"], as_index=False)
        .agg(mae_b0=("err0", lambda e: float(np.mean(np.abs(e)))),
             mae_b1=("err1", lambda e: float(np.mean(np.abs(e)))),
             med_abs_b1=("err1", lambda e: float(np.median(np.abs(e)))),
             cat_fail=("err_linf", lambda e: float(np.mean(e > 5.0))))
    )
    summary["mae_combined"] = summary[["mae_b0", "mae_b1"]].mean(axis=1)
    summary.to_csv(out_dir / "summary.csv", index=False)
    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {len(df)} rows, {len(summary)} groups, "
          f"{manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
