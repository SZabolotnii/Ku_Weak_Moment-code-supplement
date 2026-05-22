"""Acceptance for the weak-CF tuning sensitivity study.

Pass criteria (per noise regime):
  (A) the model is identified everywhere: median smallest Jacobian singular
      value > 0 for every (n_freq, u_max_mult, weighting) cell;
  (B) MAE is tuning-robust: the spread of combined MAE across all grid/weighting
      cells is <= 25% of the best cell (relative range);
  (C) the efficient two-step weighting does not hurt: its best-cell MAE is no
      worse than 1.05x the identity best-cell MAE.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def main() -> int:
    s = pd.read_csv(HERE / "results" / "summary.csv")
    rep = {"checks": {}, "metrics": {}}
    chk_a = bool((s.med_jac_min_sv > 0).all())
    rep["checks"]["identified_everywhere"] = chk_a

    chk_b = True
    for noise, g in s.groupby("noise"):
        lo, hi = float(g.mae_combined.min()), float(g.mae_combined.max())
        spread = (hi - lo) / lo if lo > 0 else float("inf")
        rep["metrics"][f"mae_rel_spread__{noise}"] = round(spread, 3)
        chk_b = chk_b and (spread <= 0.25)
    rep["checks"]["mae_tuning_robust_within_25pct"] = chk_b

    chk_c = True
    for noise, g in s.groupby("noise"):
        best_id = float(g[g.weighting == "identity"].mae_combined.min())
        best_2s = float(g[g.weighting == "2step"].mae_combined.min())
        rep["metrics"][f"twostep_over_identity_best__{noise}"] = round(best_2s / best_id, 3)
        chk_c = chk_c and (best_2s <= 1.05 * best_id)
    rep["checks"]["twostep_not_worse"] = chk_c

    rep["overall_pass"] = chk_a and chk_b and chk_c
    (HERE / "results" / "verify_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0 if rep["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
