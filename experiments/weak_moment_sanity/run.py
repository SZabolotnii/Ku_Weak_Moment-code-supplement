"""W0: weak moment sanity experiment.

Runs Monte Carlo over (distribution, n, window family, window scale) and
records raw m_2 and weak m_2^w for each replication. No estimators here:
the question is whether the weak moment functionals themselves behave.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ku_weak_moment.moments import raw_moment, weak_moment
from ku_weak_moment.simulation import (
    make_seed_grid, sample_alpha_stable, sample_cauchy,
    sample_contaminated_gaussian, sample_student_t,
    start_manifest, write_manifest,
)


def sample(rng, name, params, n):
    if name == "gaussian":
        return params["loc"] + params["scale"] * rng.standard_normal(n)
    if name == "student_t":
        return rng.standard_t(params["df"], size=n)
    if name == "cauchy":
        return sample_cauchy(rng, n, params["scale"], params["loc"])
    if name == "contaminated_gaussian":
        return sample_contaminated_gaussian(rng, n, params["eps"], params["scale_contam"])
    if name == "alpha_stable":
        return sample_alpha_stable(rng, n, params["alpha"], params.get("beta", 0.0))
    raise ValueError(f"unknown distribution {name}")


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = start_manifest("weak_moment_sanity", cfg, str(REPO))
    seeds = make_seed_grid(cfg["base_seed"], cfg["replications"])

    rows = []
    for dist in cfg["distributions"]:
        for n in cfg["sample_sizes"]:
            for r_idx, seed in enumerate(seeds):
                rng = np.random.default_rng(seed + 1009 * n)
                x = sample(rng, dist["name"], dist["params"], n)
                raw_m2 = raw_moment(x, 2)
                base_row = dict(distribution=dist["name"], n=n,
                                rep=r_idx, seed=int(seed), raw_m2=raw_m2)
                # weak m2 for each (window, scale)
                for w in cfg["windows"]:
                    for sm in w["scale_mults"]:
                        wm2 = weak_moment(x, 2, w["family"], sm)
                        rows.append({**base_row, "window": w["family"],
                                     "scale_mult": float(sm), "weak_m2": float(wm2)})
        print(f"  {dist['name']}: done")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "results.parquet", index=False)
    df.to_csv(out_dir / "results.csv", index=False)

    summary = (
        df.groupby(["distribution", "n", "window", "scale_mult"])
        .agg(
            wm2_mean=("weak_m2", "mean"),
            wm2_median=("weak_m2", "median"),
            wm2_var=("weak_m2", "var"),
            wm2_q025=("weak_m2", lambda s: float(np.quantile(s, 0.025))),
            wm2_q975=("weak_m2", lambda s: float(np.quantile(s, 0.975))),
            raw_m2_mean=("raw_m2", "mean"),
            raw_m2_median=("raw_m2", "median"),
            raw_m2_var=("raw_m2", "var"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    import time
    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {out_dir / 'results.parquet'} ({len(df)} rows)")
    print(f"wrote {out_dir / 'summary.csv'} ({len(summary)} groups)")
    print(f"duration {manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
