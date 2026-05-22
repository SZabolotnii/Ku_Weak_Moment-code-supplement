"""Core invariants for the weak-moment library.

These are the numerical-equivalence claims that hold the whole
methodology together. If any of these break, the rest of the program
is invalidated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ku_weak_moment.estimators import (
    cauchy_mle_location, cauchy_mle_regression, lad_regression,
    tukey_biweight_location, tukey_biweight_regression,
    weak_pmm_location, weak_pmm_regression,
)
from ku_weak_moment.moments import raw_moment, weak_moment
from ku_weak_moment.windows import (
    cauchy_like_window, gaussian_window, hann_value_window,
    tukey_compact_window, robust_scale_mad,
)


def test_windows_are_nonnegative():
    r = np.linspace(-10, 10, 201)
    for fn in [gaussian_window, cauchy_like_window,
               tukey_compact_window, hann_value_window]:
        w = fn(r, 1.5)
        assert (w >= 0).all()
        assert np.isfinite(w).all()


def test_compact_windows_zero_outside_support():
    r = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    c = 1.0
    assert tukey_compact_window(r, c)[0] == 0.0
    assert tukey_compact_window(r, c)[-1] == 0.0
    assert hann_value_window(r, c)[0] == 0.0


def test_wide_gaussian_window_recovers_raw_m2_on_gaussian():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(20000)
    raw = raw_moment(x, 2)
    wide = weak_moment(x, 2, "gaussian", 50.0)
    assert abs(wide - raw) < 0.01


def test_wpmm_cauchy_like_equals_cauchy_mle_location():
    rng = np.random.default_rng(2)
    y = 0.7 + rng.standard_cauchy(2000)
    mle = cauchy_mle_location(y, gamma=1.0).beta_hat[0]
    wpmm = weak_pmm_location(y, "cauchy_like", 1.0).beta_hat[0]
    assert abs(mle - wpmm) < 1e-3


def test_wpmm_cauchy_like_equals_cauchy_mle_regression():
    rng = np.random.default_rng(3)
    n = 1000
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.standard_cauchy(n)
    mle = cauchy_mle_regression(x, y, gamma=1.0).beta_hat
    wpmm = weak_pmm_regression(x, y, "cauchy_like", 1.0).beta_hat
    assert np.max(np.abs(mle - wpmm)) < 1e-6


def test_wpmm_tukey_compact_equals_tukey_biweight_location():
    rng = np.random.default_rng(4)
    y = 0.3 + rng.standard_normal(500)
    tb = tukey_biweight_location(y).beta_hat[0]
    sc = robust_scale_mad(y)
    wpmm = weak_pmm_location(y, "tukey_compact", 4.685 * sc,
                             mu0=float(np.median(y))).beta_hat[0]
    assert abs(tb - wpmm) < 1e-9


def test_wpmm_tukey_compact_equals_tukey_biweight_regression():
    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(-1, 1, n)
    y = 0.4 + 0.9 * x + 0.5 * rng.standard_normal(n)
    tb = tukey_biweight_regression(x, y).beta_hat
    b_lad = lad_regression(x, y).beta_hat
    sc = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
    wpmm = weak_pmm_regression(x, y, "tukey_compact", 4.685 * sc, beta0=b_lad).beta_hat
    assert np.max(np.abs(tb - wpmm)) < 1e-9


def test_weak_moment_finite_under_cauchy():
    """Raw m2 of Cauchy diverges; weak m2 must stay finite."""
    rng = np.random.default_rng(6)
    x = rng.standard_cauchy(5000)
    for fam in ["gaussian", "cauchy_like", "tukey_compact", "hann_value"]:
        wm = weak_moment(x, 2, fam, 2.0)
        assert np.isfinite(wm)
        assert wm < 100.0   # raw m2 of Cauchy(5000) is typically >> 1


def test_wpmm_iterates_converge():
    rng = np.random.default_rng(7)
    n = 500
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.standard_cauchy(n)
    est = weak_pmm_regression(x, y, "cauchy_like", 1.0)
    assert est.success
    assert est.n_iter < 200


def test_wpmm2_reduces_to_wpmm1_under_symmetric_noise():
    """Degree-2 must reduce to degree-1 under symmetric residuals."""
    from ku_weak_moment.estimators import weak_pmm2_location, weak_pmm2_regression
    rng = np.random.default_rng(99)
    # Pure Cauchy (symmetric)
    y = 0.3 + rng.standard_cauchy(3000)
    mu1 = weak_pmm_location(y, "cauchy_like", 1.0).beta_hat[0]
    mu2 = weak_pmm2_location(y, "cauchy_like", 1.0).beta_hat[0]
    assert abs(mu1 - mu2) < 1e-3, f"location: degree-1={mu1}, degree-2={mu2}"

    # Symmetric Student-t (regression)
    n = 1000
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.standard_t(3, n)
    b1 = weak_pmm_regression(x, y, "cauchy_like", 1.0).beta_hat
    b2 = weak_pmm2_regression(x, y, "cauchy_like", 1.0).beta_hat
    assert np.max(np.abs(b1 - b2)) < 5e-2, f"regression: deg1={b1}, deg2={b2}"


def test_wpmm2_solves_exact_score_equation():
    """Returned degree-2 estimate must drive the EXACT PMM-2 score to ~0.

    Guards against the incomplete-Jacobian bug: success=True must imply the
    real estimating-equation residual (with beta-dependent window weights)
    is below tolerance, across all window families and an asymmetric regime.
    """
    from ku_weak_moment.estimators import (
        _pmm2_a2_coefficient, _weak_cumulants_residual, weak_pmm2_regression,
        weak_pmm2_location, lad_regression,
    )
    from ku_weak_moment.windows import get_window, robust_scale_mad
    rng = np.random.default_rng(2024)
    n = 800
    x = rng.uniform(-1, 1, n)
    # one-sided asymmetric contamination so a_2 != 0
    eps = np.where(rng.random(n) < 0.15, 4.0 + 3.0 * np.abs(rng.standard_cauchy(n)),
                   rng.standard_normal(n))
    y = 0.5 + 1.0 * x + eps

    for fam, sm in [("gaussian", 1.5), ("cauchy_like", 1.0),
                    ("tukey_compact", 4.685), ("hann_value", 4.685)]:
        b_lad = lad_regression(x, y).beta_hat
        X = np.column_stack([np.ones_like(x), x])
        sc = robust_scale_mad(y - X @ b_lad)
        scale = sm * sc
        est = weak_pmm2_regression(x, y, fam, scale, beta0=b_lad)
        if not est.success:
            continue  # non-convergence is allowed; false success is not
        # Recompute the exact score at the returned beta and assert it is ~0
        w_fn = get_window(fam)
        r0 = y - X @ b_lad
        w0 = w_fn(r0, scale)
        _, s2, k3, k4 = _weak_cumulants_residual(r0, w0)
        a2 = _pmm2_a2_coefficient(s2, k3, k4)
        r = y - X @ est.beta_hat
        w = w_fn(r, scale)
        score = X.T @ (w * (r + a2 * (r ** 2 - s2))) / w.sum()
        assert np.linalg.norm(score) < 1e-5, f"{fam}: score norm {np.linalg.norm(score)}"
        assert "score_norm" in est.diagnostics


def test_wpmm2_analytic_jacobian_matches_numerical():
    """Analytic regression Jacobian must match finite differences."""
    from ku_weak_moment.estimators import (
        _pmm2_a2_coefficient, _weak_cumulants_residual, lad_regression,
    )
    from ku_weak_moment.windows import get_window, get_window_deriv, robust_scale_mad
    rng = np.random.default_rng(7)
    n = 300
    x = rng.uniform(-1, 1, n)
    eps = np.where(rng.random(n) < 0.15, 4.0 + 3.0 * np.abs(rng.standard_cauchy(n)),
                   rng.standard_normal(n))
    y = 0.5 + 1.0 * x + eps
    X = np.column_stack([np.ones_like(x), x])
    b_lad = lad_regression(x, y).beta_hat
    sc = robust_scale_mad(y - X @ b_lad)

    for fam in ["gaussian", "cauchy_like", "tukey_compact", "hann_value"]:
        scale = (1.0 if fam in ("gaussian", "cauchy_like") else 4.685) * sc
        w_fn = get_window(fam)
        wp_fn = get_window_deriv(fam)
        r0 = y - X @ b_lad
        w0 = w_fn(r0, scale)
        _, s2, k3, k4 = _weak_cumulants_residual(r0, w0)
        a2 = _pmm2_a2_coefficient(s2, k3, k4)

        def score(beta):
            r = y - X @ beta
            w = w_fn(r, scale)
            return X.T @ (w * (r + a2 * (r ** 2 - s2))) / w.sum()

        def jac(beta):
            r = y - X @ beta
            w = w_fn(r, scale); wp = wp_fn(r, scale); sw = w.sum()
            s = r + a2 * (r ** 2 - s2)
            N = X.T @ (w * s); Xtwp = X.T @ wp
            p = wp * s + w * (1.0 + 2.0 * a2 * r)
            XtpX = X.T @ (p[:, None] * X)
            return -XtpX / sw + np.outer(N, Xtwp) / sw ** 2

        beta = b_lad + np.array([0.1, -0.05])
        J_analytic = jac(beta)
        h = 1e-6
        J_num = np.zeros((2, 2))
        for k in range(2):
            bp = beta.copy(); bp[k] += h
            bm = beta.copy(); bm[k] -= h
            J_num[:, k] = (score(bp) - score(bm)) / (2 * h)
        assert np.max(np.abs(J_analytic - J_num)) < 1e-4, \
            f"{fam}: analytic-numerical Jacobian mismatch {np.max(np.abs(J_analytic - J_num))}"


def test_wpmm2_a2_nonzero_under_asymmetric_noise():
    """Under asymmetric residuals, a_2 coefficient should be non-trivial."""
    from ku_weak_moment.estimators import weak_pmm2_location
    rng = np.random.default_rng(123)
    n = 2000
    # 80% N(0,1) + 20% one-sided shifted Cauchy
    eps = np.where(rng.random(n) < 0.2,
                   3.0 + 5.0 * np.abs(rng.standard_cauchy(n)),
                   rng.standard_normal(n))
    y = 0.0 + eps
    from ku_weak_moment.windows import robust_scale_mad
    sc = robust_scale_mad(y - np.median(y))
    est = weak_pmm2_location(y, "tukey_compact", 4.685 * sc)
    assert "a_2" in est.diagnostics
    assert abs(est.diagnostics["a_2"]) > 1e-3


def test_wpmm3_reduces_to_degree1_under_gaussian():
    """Under Gaussian residuals (mesokurtic) wPMM3 must match degree-1."""
    from ku_weak_moment.estimators import weak_pmm3_location, weak_pmm3_regression
    from ku_weak_moment.windows import robust_scale_mad
    rng = np.random.default_rng(11)
    y = 2.0 + rng.standard_normal(8000)
    sc = robust_scale_mad(y - np.median(y))
    mu1 = weak_pmm_location(y, "gaussian", 1.5 * sc).beta_hat[0]
    mu3 = weak_pmm3_location(y, "gaussian", 1.5 * sc).beta_hat[0]
    assert abs(mu1 - mu3) < 5e-3, f"deg1={mu1}, deg3={mu3}"

    n = 1500
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.standard_normal(n)
    from ku_weak_moment.estimators import lad_regression
    b_lad = lad_regression(x, y).beta_hat
    X = np.column_stack([np.ones_like(x), x])
    sc = robust_scale_mad(y - X @ b_lad)
    b1 = weak_pmm_regression(x, y, "gaussian", 1.5 * sc, beta0=b_lad).beta_hat
    b3 = weak_pmm3_regression(x, y, "gaussian", 1.5 * sc, beta0=b_lad).beta_hat
    assert np.max(np.abs(b1 - b3)) < 2e-2, f"deg1={b1}, deg3={b3}"


def test_wpmm3_solves_exact_score_and_jacobian():
    """wPMM3 success implies exact odd-cubic score ~0; analytic jac matches FD."""
    from ku_weak_moment.estimators import (
        weak_pmm3_regression, _weak_pmm3_kappa, lad_regression,
    )
    from ku_weak_moment.windows import get_window, get_window_deriv, robust_scale_mad
    rng = np.random.default_rng(13)
    n = 600
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.standard_t(3, n)  # symmetric heavy-tail
    X = np.column_stack([np.ones_like(x), x])
    b_lad = lad_regression(x, y).beta_hat
    sc = robust_scale_mad(y - X @ b_lad)

    for fam in ["gaussian", "tukey_compact", "hann_value"]:
        scale = (1.5 if fam == "gaussian" else 3.0) * sc
        est = weak_pmm3_regression(x, y, fam, scale, beta0=b_lad)
        if est.diagnostics.get("reduced_to_degree1"):
            continue
        if not est.success:
            continue
        w_fn = get_window(fam); wp_fn = get_window_deriv(fam)
        kappa, _, _ = _weak_pmm3_kappa(y - X @ b_lad, w_fn(y - X @ b_lad, scale))

        def score(beta):
            r = y - X @ beta
            w = w_fn(r, scale)
            return X.T @ (w * (kappa * r - r ** 3)) / w.sum()

        def jac(beta):
            r = y - X @ beta
            w = w_fn(r, scale); wp = wp_fn(r, scale); sw = w.sum()
            s = kappa * r - r ** 3
            N = X.T @ (w * s); Xtwp = X.T @ wp
            p = wp * s + w * (kappa - 3.0 * r ** 2)
            return -X.T @ (p[:, None] * X) / sw + np.outer(N, Xtwp) / sw ** 2

        assert np.linalg.norm(score(est.beta_hat)) < 1e-5
        beta = b_lad + np.array([0.05, -0.03])
        Ja = jac(beta)
        h = 1e-6
        Jn = np.zeros((2, 2))
        for k in range(2):
            bp = beta.copy(); bp[k] += h
            bm = beta.copy(); bm[k] -= h
            Jn[:, k] = (score(bp) - score(bm)) / (2 * h)
        assert np.max(np.abs(Ja - Jn)) < 1e-4, f"{fam}: {np.max(np.abs(Ja-Jn))}"


def test_weak_cf_gaussian_sanity():
    """weak-CF location on Gaussian ~ sample mean; regression ~ OLS."""
    from ku_weak_moment.estimators import weak_cf_location, weak_cf_regression, ols_regression
    rng = np.random.default_rng(31)
    y = 3.0 + rng.standard_normal(6000)
    assert abs(weak_cf_location(y).beta_hat[0] - np.mean(y)) < 0.05
    n = 2000
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.standard_normal(n)
    b_cf = weak_cf_regression(x, y).beta_hat
    b_ols = ols_regression(x, y).beta_hat
    assert np.max(np.abs(b_cf - b_ols)) < 0.1


def test_weak_cf_robust_under_cauchy():
    """weak-CF stays near truth under Cauchy where OLS explodes."""
    from ku_weak_moment.estimators import weak_cf_regression, ols_regression
    rng = np.random.default_rng(32)
    n = 800
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.standard_cauchy(n)
    b_cf = weak_cf_regression(x, y).beta_hat
    assert abs(b_cf[0] - 0.5) < 0.2 and abs(b_cf[1] - 1.0) < 0.2
    # OLS should be far off (catastrophic) — confirms the contrast
    b_ols = ols_regression(x, y).beta_hat
    assert np.max(np.abs(b_ols - np.array([0.5, 1.0]))) > 0.3


def test_weak_cf_bounded_score_handles_alpha_stable():
    """weak-CF must produce a finite, sane estimate on very heavy alpha-stable
    (alpha<1.5) where windowed high moments are unstable."""
    from ku_weak_moment.estimators import weak_cf_regression
    from ku_weak_moment.simulation import sample_alpha_stable
    rng = np.random.default_rng(33)
    n = 800
    x = rng.uniform(-1, 1, n)
    eps = sample_alpha_stable(rng, n, alpha=1.2, beta=0.0)
    y = 0.5 + 1.0 * x + eps
    b = weak_cf_regression(x, y).beta_hat
    assert np.all(np.isfinite(b))
    assert abs(b[0] - 0.5) < 0.5 and abs(b[1] - 1.0) < 0.5


def test_wpmm_window_scale_independence_under_symmetric_cauchy():
    """At plateau, different windows give similar location estimates."""
    rng = np.random.default_rng(8)
    y = 1.0 + rng.standard_cauchy(3000)
    sc = robust_scale_mad(y - np.median(y))
    estimates = []
    for fam, sm in [("gaussian", 1.0), ("cauchy_like", 1.0),
                    ("tukey_compact", 3.0), ("hann_value", 3.0)]:
        est = weak_pmm_location(y, fam, sm * sc)
        estimates.append(est.beta_hat[0])
    spread = max(estimates) - min(estimates)
    assert spread < 0.05  # all windows within 0.05 of each other on plateau


def test_kappaW_solves_first_order_condition():
    """Mirror of Lean `Foundations.foc_kappaW`: the implemented kappa^w is the
    root of the degree-3 asymptotic-variance first-order condition
    kappa*(m4 - 3 m2^2) - (m6 - 3 m2 m4) = 0."""
    from ku_weak_moment.estimators import _weak_pmm3_kappa
    from ku_weak_moment.windows import get_window
    rng = np.random.default_rng(7)
    r = rng.standard_t(4, 4000)
    w = get_window("gaussian")(r, 1.5)
    kappa, _, denom = _weak_pmm3_kappa(r, w)
    assert np.isfinite(kappa) and abs(denom) > 1e-6
    sw = w.sum()
    m1 = (w * r).sum() / sw
    c = r - m1
    m2 = (w * c ** 2).sum() / sw
    m4 = (w * c ** 4).sum() / sw
    m6 = (w * c ** 6).sum() / sw
    foc = kappa * (m4 - 3.0 * m2 ** 2) - (m6 - 3.0 * m2 * m4)
    assert abs(foc) < 1e-6 * max(abs(m6), 1.0), f"FOC residual {foc}"


def test_cf_score_bounded_and_odd():
    """Mirror of Lean `Score.abs_cfScore_le_one` / `Score.cfScore_odd`:
    sin(u r) is bounded by 1 and odd in r."""
    rng = np.random.default_rng(11)
    u = rng.uniform(0.1, 5.0, 50)
    r = rng.standard_cauchy(50)
    s = np.sin(np.outer(u, r))
    assert np.all(np.abs(s) <= 1.0 + 1e-12)
    assert np.allclose(np.sin(np.outer(u, -r)), -s, atol=1e-12)


def test_s_and_mm_estimators_robust_under_cauchy():
    """S- and MM-estimators recover beta under Cauchy regression where OLS fails."""
    from ku_weak_moment.estimators import (
        s_estimator_regression, mm_estimator_regression, ols_regression,
    )
    rng = np.random.default_rng(5)
    n = 500
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.standard_cauchy(n)
    s_est = s_estimator_regression(x, y)
    mm = mm_estimator_regression(x, y)
    assert np.max(np.abs(s_est.beta_hat - np.array([0.5, 1.0]))) < 0.3
    assert np.max(np.abs(mm.beta_hat - np.array([0.5, 1.0]))) < 0.3
    # OLS is destroyed by the infinite-variance noise
    assert np.max(np.abs(ols_regression(x, y).beta_hat - np.array([0.5, 1.0]))) > 0.5


def test_mm_estimator_high_efficiency_on_gaussian():
    """MM (c=4.685) tracks OLS closely on clean Gaussian noise (high efficiency)."""
    from ku_weak_moment.estimators import mm_estimator_regression, ols_regression
    rng = np.random.default_rng(9)
    n = 4000
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + 0.7 * rng.standard_normal(n)
    mm = mm_estimator_regression(x, y)
    ols = ols_regression(x, y)
    assert np.max(np.abs(mm.beta_hat - ols.beta_hat)) < 0.05


def test_sandwich_ci_coverage_on_gaussian():
    """Sandwich 95% CI for degree-1 weak PMM has ~nominal coverage on Gaussian."""
    from ku_weak_moment.inference import weak_pmm1_sandwich
    from ku_weak_moment.windows import robust_scale_mad
    from ku_weak_moment.estimators import lad_regression
    beta_true = np.array([0.5, 1.0])
    n, R = 400, 250
    cover1 = 0
    for r_idx in range(R):
        rng = np.random.default_rng(1000 + r_idx)
        x = rng.uniform(-1, 1, n)
        y = beta_true[0] + beta_true[1] * x + 0.8 * rng.standard_normal(n)
        b_lad = lad_regression(x, y).beta_hat
        sc = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
        beta, cov, se = weak_pmm1_sandwich(x, y, "gaussian", 1.5 * sc)
        if abs(beta[1] - beta_true[1]) <= 1.96 * se[1]:
            cover1 += 1
    cov_rate = cover1 / R
    assert 0.88 <= cov_rate <= 0.99, f"slope coverage {cov_rate}"


def test_weak_cf_two_step_weighting_identified():
    """Two-step weak-CF GMM recovers beta and is identified (Jacobian full rank)."""
    from ku_weak_moment.estimators import weak_cf_regression
    rng = np.random.default_rng(17)
    n = 500
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.standard_cauchy(n)
    e = weak_cf_regression(x, y, weighting="2step")
    assert e.success
    assert np.max(np.abs(e.beta_hat - np.array([0.5, 1.0]))) < 0.3
    assert e.diagnostics["jac_smallest_sv"] > 0.0  # identification (full rank)
