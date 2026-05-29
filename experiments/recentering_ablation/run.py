"""JSCS referee Q5: window re-centering vs degree-2 under asymmetric noise.

Estimators compared on the intercept bias (beta0 = 0.5):
  * wpmm1            -- degree-1 weak PMM (window centered at 0)
  * wpmm2            -- degree-2 weak PMM (windowed-skewness term)
  * wpmm1_recentered -- degree-1 with the window RE-CENTERED at a robust (mode)
                        location of the residuals (mean-shift), folded into the
                        intercept. This is the "correct fix" the paper conjectures.
  * lad, ols         -- references.

Plus an init/scale sensitivity sweep for wpmm1/wpmm2 (init in {lad, ols,
median-slope}, scale in {mad, iqr}) to show the degree-2 negative result is not
an artifact of the preliminary fit.
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

from ku_weak_moment.estimators import (  # noqa: E402
    lad_regression, ols_regression, weak_pmm_regression, weak_pmm2_regression,
)
from ku_weak_moment.simulation import (  # noqa: E402
    make_seed_grid, sample_alpha_stable, start_manifest, write_manifest,
)
from ku_weak_moment.windows import robust_scale_mad  # noqa: E402

BETA = np.array([0.5, 1.0])


def sample_skewed_contam(rng, n, eps_frac, shift, scale_contam):
    """Identical to asymmetric_regression.sample_skewed_contam."""
    is_c = rng.random(n) < eps_frac
    base = rng.standard_normal(n)
    contam = shift + scale_contam * rng.standard_cauchy(n)
    return np.where(is_c, contam, base)


def sample_skew_t(rng, n, df, alpha_skew):
    """Azzalini-Capitanio skew-t mixture (== asymmetric_regression.sample_skew_t)."""
    z0 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    delta = alpha_skew / np.sqrt(1.0 + alpha_skew ** 2)
    v = 1.0 / rng.gamma(shape=df / 2.0, scale=2.0 / df, size=n)
    return np.sqrt(v) * (delta * np.abs(z0) + np.sqrt(1.0 - delta ** 2) * z1)


def gen_residual(rng, regime, n):
    k = regime["kind"]
    if k == "skewed_contam":
        return sample_skewed_contam(rng, n, regime["eps"], regime["shift"], regime["scale_contam"])
    if k == "skew_t":
        return sample_skew_t(rng, n, regime["df"], regime["alpha_skew"])
    if k == "alpha_stable":
        return sample_alpha_stable(rng, n, regime["alpha"], regime["beta"])
    raise ValueError(k)


def median_slope_start(x, y):
    """Theil-Sen-like cheap start: median pairwise slope + median intercept."""
    xm, ym = np.median(x), np.median(y)
    b1 = np.median((y - ym) / np.where(np.abs(x - xm) < 1e-9, np.nan, x - xm))
    if not np.isfinite(b1):
        b1 = 0.0
    b0 = np.median(y - b1 * x)
    return np.array([b0, b1])


def iqr_scale(r):
    q1, q3 = np.percentile(r, [25, 75])
    return float((q3 - q1) / 1.349)


def mode_shift(r, bw, iters=100, tol=1e-10):
    """Gaussian mean-shift mode of residuals r (bulk peak)."""
    c = float(np.median(r))
    for _ in range(iters):
        w = np.exp(-0.5 * ((r - c) / bw) ** 2)
        sw = w.sum()
        if sw <= 0:
            break
        c_new = float((w * r).sum() / sw)
        if abs(c_new - c) < tol:
            c = c_new
            break
        c = c_new
    return c


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = start_manifest("recentering_ablation", cfg, str(REPO))
    seeds = make_seed_grid(cfg["base_seed"], cfg["replications"])
    fam = cfg["window_family"]
    rows = []
    for regime in cfg["regimes"]:
        for n in cfg["sample_sizes"]:
            for sm in cfg["scale_mults"]:
                t0 = time.perf_counter()
                for r_idx, seed in enumerate(seeds):
                    rng = np.random.default_rng(seed + 17 * n)
                    x = rng.uniform(-1.0, 1.0, n)
                    eps = gen_residual(rng, regime, n)
                    y = BETA[0] + BETA[1] * x + eps
                    X = np.column_stack([np.ones_like(x), x])
                    b_lad = lad_regression(x, y).beta_hat
                    b_ols = ols_regression(x, y).beta_hat
                    sc_mad = robust_scale_mad(y - X @ b_lad)
                    if sc_mad <= 0.0:
                        sc_mad = 1.0
                    scale = sm * sc_mad

                    def push(name, b0hat, init="lad", scale_kind="mad"):
                        rows.append({"regime": regime["name"], "n": n, "scale_mult": sm,
                                     "estimator": name, "init": init, "scale_kind": scale_kind,
                                     "b0_hat": float(b0hat)})

                    # main estimators (LAD init, MAD scale)
                    e1 = weak_pmm_regression(x, y, fam, scale, beta0=b_lad)
                    e2 = weak_pmm2_regression(x, y, fam, scale, beta0=b_lad)
                    push("wpmm1", e1.beta_hat[0])
                    push("wpmm2", e2.beta_hat[0])
                    # re-centered degree-1: shift intercept to the residual mode
                    r1 = y - X @ e1.beta_hat
                    c_hat = mode_shift(r1, bw=0.5 * sc_mad)
                    push("wpmm1_recentered", e1.beta_hat[0] + c_hat)
                    push("lad", b_lad[0]); push("ols", b_ols[0])

                    # sensitivity: init x scale (only at sm consistent; record once per rep)
                    for init_name, b_init in [("ols", b_ols), ("medslope", median_slope_start(x, y))]:
                        ee1 = weak_pmm_regression(x, y, fam, scale, beta0=b_init)
                        ee2 = weak_pmm2_regression(x, y, fam, scale, beta0=b_init)
                        push("wpmm1", ee1.beta_hat[0], init=init_name)
                        push("wpmm2", ee2.beta_hat[0], init=init_name)
                    sc_iqr = iqr_scale(y - X @ b_lad) or sc_mad
                    ee1 = weak_pmm_regression(x, y, fam, sm * sc_iqr, beta0=b_lad)
                    ee2 = weak_pmm2_regression(x, y, fam, sm * sc_iqr, beta0=b_lad)
                    push("wpmm1", ee1.beta_hat[0], scale_kind="iqr")
                    push("wpmm2", ee2.beta_hat[0], scale_kind="iqr")
                print(f"  {regime['name']} n={n} sm={sm}: {time.perf_counter()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df["bias0"] = df.b0_hat - BETA[0]
    try:
        df.to_parquet(out_dir / "results.parquet", index=False)
    except Exception as exc:
        print(f"  [warn] parquet skipped ({exc.__class__.__name__})", flush=True)
    summary = (
        df.groupby(["regime", "n", "scale_mult", "estimator", "init", "scale_kind"], as_index=False)
        .agg(bias_b0=("bias0", "mean"),
             abs_bias_b0=("bias0", lambda e: float(np.abs(np.mean(e)))),
             mae_b0=("bias0", lambda e: float(np.mean(np.abs(e)))),
             med_b0=("bias0", "median"))
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {len(df)} rows, {len(summary)} groups, "
          f"{manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
