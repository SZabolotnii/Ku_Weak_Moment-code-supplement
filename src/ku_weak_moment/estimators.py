"""Estimators for location and linear regression.

Weak PMM (degree-1 score) is implemented via iteratively reweighted
least squares with the value-domain window as the weight. For specific
window choices it recapitulates well-known redescending M-estimators:

  - cauchy_like  window, scale = gamma  -->  Cauchy MLE (location/regression)
  - tukey_compact window, scale = c     -->  Tukey biweight M-estimator
  - gaussian     window, scale = sigma  -->  Welsch M-estimator
  - hann_value   window, scale = c      -->  Hann-tapered M-estimator (new)

This is the polynomial-structure-preserving family from spec section 13.2:
the window selects a member of a redescending M-estimator family, while
the surrounding PMM scaffolding (polynomial degree, basis, contrast)
remains intact.

For Cauchy linear regression (W1) we also include classical PMM-2 as
a naive implementation that exposes its failure mode under heavy tails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import least_squares, minimize, root

from ku_weak_moment.windows import get_window, get_window_deriv, robust_scale_mad


@dataclass
class Estimate:
    beta_hat: np.ndarray
    success: bool
    objective_value: float = float("nan")
    n_iter: int = 0
    runtime_sec: float = float("nan")
    window_family: str = ""
    window_scale: float = float("nan")
    diagnostics: dict = field(default_factory=dict)


def _irwls_location(y: np.ndarray, weight_fn: Callable[[np.ndarray], np.ndarray],
                    mu0: Optional[float] = None, max_iter: int = 200,
                    tol: float = 1e-9) -> tuple[float, bool, int]:
    if mu0 is None:
        mu0 = float(np.median(y))
    mu = mu0
    for it in range(max_iter):
        r = y - mu
        w = weight_fn(r)
        s = float(w.sum())
        if s <= 0.0:
            return mu, False, it
        mu_new = float((w * y).sum() / s)
        if abs(mu_new - mu) < tol:
            return mu_new, True, it + 1
        mu = mu_new
    return mu, False, max_iter


def _irwls_regression(x: np.ndarray, y: np.ndarray,
                      weight_fn: Callable[[np.ndarray], np.ndarray],
                      beta0: np.ndarray, max_iter: int = 200,
                      tol: float = 1e-9) -> tuple[np.ndarray, bool, int]:
    X = np.column_stack([np.ones_like(x), x])
    beta = beta0.astype(float).copy()
    for it in range(max_iter):
        r = y - X @ beta
        w = weight_fn(r)
        if w.sum() <= 0.0:
            return beta, False, it
        XtW = X.T * w
        A = XtW @ X
        b = XtW @ y
        try:
            beta_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return beta, False, it
        if np.max(np.abs(beta_new - beta)) < tol:
            return beta_new, True, it + 1
        beta = beta_new
    return beta, False, max_iter


def weak_pmm_location(y: np.ndarray, window: str, scale: float,
                      mu0: Optional[float] = None) -> Estimate:
    w_fn = get_window(window)

    def weight_fn(r):
        return w_fn(r, scale)

    mu, success, n_iter = _irwls_location(y, weight_fn, mu0=mu0)
    r = y - mu
    w = weight_fn(r)
    obj = float(np.sum(w * r ** 2) / w.sum()) if w.sum() > 0 else float("nan")
    return Estimate(beta_hat=np.array([mu]), success=success,
                    objective_value=obj, n_iter=n_iter,
                    window_family=window, window_scale=float(scale))


def weak_pmm_regression(x: np.ndarray, y: np.ndarray, window: str, scale: float,
                        beta0: Optional[np.ndarray] = None) -> Estimate:
    if beta0 is None:
        b_lad = lad_regression(x, y).beta_hat
        beta0 = b_lad
    w_fn = get_window(window)

    def weight_fn(r):
        return w_fn(r, scale)

    beta, success, n_iter = _irwls_regression(x, y, weight_fn, beta0)
    X = np.column_stack([np.ones_like(x), x])
    r = y - X @ beta
    w = weight_fn(r)
    obj = float(np.sum(w * r ** 2) / w.sum()) if w.sum() > 0 else float("nan")
    return Estimate(beta_hat=beta, success=success,
                    objective_value=obj, n_iter=n_iter,
                    window_family=window, window_scale=float(scale))


def cauchy_mle_location(y: np.ndarray, gamma: float) -> Estimate:
    """Cauchy MLE for location with known scale gamma.

    Equivalent to weak PMM with cauchy_like window at scale gamma; we
    implement directly for clarity and as a sanity baseline.
    """
    def neg_loglik(mu):
        return float(np.sum(np.log(1.0 + ((y - mu[0]) / gamma) ** 2)))

    res = minimize(neg_loglik, x0=np.array([float(np.median(y))]),
                   method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12})
    return Estimate(beta_hat=res.x, success=bool(res.success),
                    objective_value=float(res.fun), n_iter=int(res.nit),
                    window_family="cauchy_mle", window_scale=float(gamma))


def cauchy_mle_regression(x: np.ndarray, y: np.ndarray, gamma: float,
                          beta0: Optional[np.ndarray] = None) -> Estimate:
    """Cauchy regression MLE with known scale via IRLS.

    The score equation for known-scale Cauchy regression is

        sum_i x_i r_i / (gamma^2 + r_i^2) = 0.

    Multiplying all weights by gamma^2 gives exactly the `cauchy_like`
    weak-PMM weight, so the IRLS fixed point is both the known-scale
    Cauchy score solution and the scale-known weak PMM special case.
    """
    if beta0 is None:
        beta0 = lad_regression(x, y).beta_hat
    w_fn = get_window("cauchy_like")
    beta, success, n_iter = _irwls_regression(
        x, y, lambda r: w_fn(r, gamma), beta0=beta0, max_iter=300, tol=1e-10
    )
    X = np.column_stack([np.ones_like(x), x])
    r = y - X @ beta
    obj = float(np.sum(np.log1p((r / gamma) ** 2)))
    return Estimate(beta_hat=beta, success=success, objective_value=obj,
                    n_iter=n_iter, window_family="cauchy_mle",
                    window_scale=float(gamma))


def lad_regression(x: np.ndarray, y: np.ndarray,
                   beta0: Optional[np.ndarray] = None) -> Estimate:
    X = np.column_stack([np.ones_like(x), x])
    if beta0 is None:
        beta0 = np.linalg.lstsq(X, y, rcond=None)[0]

    def loss(beta):
        return float(np.sum(np.abs(y - X @ beta)))

    res = minimize(loss, x0=beta0, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000})
    return Estimate(beta_hat=res.x, success=bool(res.success),
                    objective_value=float(res.fun), n_iter=int(res.nit),
                    window_family="lad")


def tukey_biweight_location(y: np.ndarray, c_mult: float = 4.685) -> Estimate:
    """Tukey biweight location with c = c_mult * MAD/0.6745.

    Default c_mult = 4.685 gives 95% Gaussian efficiency.
    """
    sc = robust_scale_mad(y)
    if sc <= 0.0:
        return Estimate(beta_hat=np.array([float(np.median(y))]), success=True,
                        window_family="tukey_biweight", window_scale=0.0)
    return weak_pmm_location(y, "tukey_compact", c_mult * sc,
                             mu0=float(np.median(y)))._replace_meta("tukey_biweight")


def tukey_biweight_regression(x: np.ndarray, y: np.ndarray,
                              c_mult: float = 4.685) -> Estimate:
    b_lad = lad_regression(x, y).beta_hat
    X = np.column_stack([np.ones_like(x), x])
    r0 = y - X @ b_lad
    sc = robust_scale_mad(r0)
    if sc <= 0.0:
        return Estimate(beta_hat=b_lad, success=True,
                        window_family="tukey_biweight", window_scale=0.0)
    est = weak_pmm_regression(x, y, "tukey_compact", c_mult * sc, beta0=b_lad)
    est.window_family = "tukey_biweight"
    return est


def huber_regression(x: np.ndarray, y: np.ndarray, c_mult: float = 1.345) -> Estimate:
    """Huber M-estimator through IRLS with LAD/MAD initialization."""
    b_lad = lad_regression(x, y).beta_hat
    X = np.column_stack([np.ones_like(x), x])
    r0 = y - X @ b_lad
    sc = robust_scale_mad(r0)
    if sc <= 0.0:
        return Estimate(beta_hat=b_lad, success=True,
                        window_family="huber", window_scale=0.0)
    c = c_mult * sc

    def weight_fn(r: np.ndarray) -> np.ndarray:
        a = np.abs(r)
        w = np.ones_like(r, dtype=float)
        mask = a > c
        w[mask] = c / np.maximum(a[mask], 1e-15)
        return w

    beta, success, n_iter = _irwls_regression(x, y, weight_fn, b_lad)
    rr = y - X @ beta
    abs_rr = np.abs(rr)
    quad = abs_rr <= c
    obj = float(np.sum(np.where(quad, 0.5 * rr ** 2, c * (abs_rr - 0.5 * c))))
    return Estimate(beta_hat=beta, success=success, objective_value=obj,
                    n_iter=n_iter, window_family="huber",
                    window_scale=float(c))


def classical_pmm2_regression(x: np.ndarray, y: np.ndarray) -> Estimate:
    """Classical PMM-2 for linear regression.

    Under symmetric residuals (gamma_3 = 0) the PMM-2 estimator reduces
    to the same normal equations as OLS, but with the OLS variance
    replaced by an empirical second moment of residuals. For heavy-tailed
    errors with non-existent second moment the empirical second moment
    diverges with n, so PMM-2 is reported here as OLS plus a diagnostic
    on the sample variance of residuals (which we expect to be unstable).
    """
    X = np.column_stack([np.ones_like(x), x])
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta_ols
    m2 = float(np.mean(r ** 2))
    m4 = float(np.mean(r ** 4))
    gamma4 = m4 / (m2 ** 2) - 3.0 if m2 > 0 else float("nan")
    return Estimate(
        beta_hat=beta_ols, success=True,
        objective_value=m2, n_iter=0,
        window_family="classical_pmm2",
        diagnostics={"m2_residual": m2, "m4_residual": m4, "gamma4_residual": gamma4},
    )


# Patch Estimate to allow renaming family after the fact (used by Tukey wrapper).
def _replace_meta(self: Estimate, family: str) -> Estimate:
    self.window_family = family
    return self


Estimate._replace_meta = _replace_meta  # type: ignore[attr-defined]


def ols_regression(x: np.ndarray, y: np.ndarray) -> Estimate:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return Estimate(beta_hat=beta, success=True, window_family="ols")


# ---------- degree-2 weak PMM (Kunchenko PMM-2 with weak cumulants) ----------

def _weak_cumulants_residual(r: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float]:
    """Windowed central cumulants up to order 4 of residuals r with weights w.

    Returns (m_1^w, kappa_2^w, kappa_3^w, kappa_4^w).
    """
    sw = w.sum()
    if sw <= 0:
        return 0.0, 0.0, 0.0, 0.0
    m1 = float((w * r).sum() / sw)
    rc = r - m1
    m2c = float((w * rc ** 2).sum() / sw)
    m3c = float((w * rc ** 3).sum() / sw)
    m4c = float((w * rc ** 4).sum() / sw)
    return m1, m2c, m3c, m4c - 3.0 * m2c ** 2


def _pmm2_a2_coefficient(kappa_2: float, kappa_3: float, kappa_4: float) -> float:
    """Optimal a_2 in PMM-2 location score under known windowed cumulants.

    Returns a_2 = -gamma_3 / (2 sqrt(kappa_2 (2 + gamma_4))) with safeguards
    for invalid kappa_2 or negative denominator.
    """
    if kappa_2 <= 0:
        return 0.0
    gamma_3 = kappa_3 / (kappa_2 ** 1.5)
    gamma_4 = kappa_4 / (kappa_2 ** 2)
    denom = 2.0 + gamma_4
    if denom <= 1e-6:
        return 0.0
    return -gamma_3 / (2.0 * np.sqrt(kappa_2 * denom))


_SCORE_TOL = 1e-7  # acceptance tolerance for the (normalized) PMM-2 score norm


def weak_pmm2_location(y: np.ndarray, window: str, scale: float,
                       mu0: Optional[float] = None) -> Estimate:
    """Degree-2 weak PMM for location estimation (Kunchenko PMM-2 form).

    Solves the EXACT estimating equation
        g(mu) = sum_i w(r_i) [r_i + a_2 (r_i^2 - sigma2_target)] / sum_i w(r_i) = 0,
    where r_i = y_i - mu and the weights w(r_i) themselves depend on mu.
    sigma2_target and a_2 are frozen from the initial (median) residuals;
    this breaks the algebraic degeneracy that otherwise collapses degree-2
    onto degree-1.

    The root is found with scipy.optimize.root (numerical Jacobian, so the
    derivative of w through r is handled), and `success` requires the final
    score norm to be below _SCORE_TOL. Under symmetric residuals a_2 -> 0
    so this reduces to the weighted mean (= degree-1 weak PMM).
    """
    w_fn = get_window(window)
    mu_init = float(np.median(y)) if mu0 is None else float(mu0)

    r0 = y - mu_init
    w0 = w_fn(r0, scale)
    if float(w0.sum()) <= 0:
        return Estimate(beta_hat=np.array([mu_init]), success=False,
                        n_iter=0, window_family=window, window_scale=float(scale))
    _, sigma2_target, k3_0, k4_0 = _weak_cumulants_residual(r0, w0)
    a2 = _pmm2_a2_coefficient(sigma2_target, k3_0, k4_0)

    wp_fn = get_window_deriv(window)

    def score(mu_arr: np.ndarray) -> np.ndarray:
        mu = float(mu_arr[0])
        r = y - mu
        w = w_fn(r, scale)
        sw = float(w.sum())
        if sw <= 0:
            return np.array([0.0])
        return np.array([float(np.sum(w * (r + a2 * (r ** 2 - sigma2_target))) / sw)])

    def jac(mu_arr: np.ndarray) -> np.ndarray:
        mu = float(mu_arr[0])
        r = y - mu
        w = w_fn(r, scale)
        wp = wp_fn(r, scale)
        sw = float(w.sum())
        if sw <= 0:
            return np.array([[1.0]])
        s = r + a2 * (r ** 2 - sigma2_target)
        N = float(np.sum(w * s))
        p = wp * s + w * (1.0 + 2.0 * a2 * r)
        dN = -float(np.sum(p))
        dsw = -float(np.sum(wp))
        return np.array([[(dN * sw - N * dsw) / sw ** 2]])

    # Degenerate (symmetric) case: a_2 ~ 0 -> closed-form weighted-mean fixed point.
    if abs(a2) < 1e-12:
        mu = mu_init
        for _ in range(200):
            r = y - mu
            w = w_fn(r, scale)
            sw = float(w.sum())
            if sw <= 0:
                break
            mu_new = float((w * y).sum() / sw)
            if abs(mu_new - mu) < 1e-12:
                mu = mu_new
                break
            mu = mu_new
        success = abs(float(score(np.array([mu]))[0])) < _SCORE_TOL
        return Estimate(beta_hat=np.array([mu]), success=bool(success),
                        objective_value=float(sigma2_target),
                        window_family=window, window_scale=float(scale),
                        diagnostics={"a_2": float(a2), "sigma2_target": float(sigma2_target),
                                     "score_norm": float(abs(score(np.array([mu]))[0])),
                                     "kappa_3_w_init": float(k3_0),
                                     "kappa_4_w_init": float(k4_0)})

    sol = root(score, np.array([mu_init]), jac=jac, method="hybr",
               options={"xtol": 1e-10, "maxfev": 2000})
    mu = float(sol.x[0])
    score_norm = float(abs(score(np.array([mu]))[0]))
    success = bool(sol.success) and score_norm < _SCORE_TOL
    return Estimate(beta_hat=np.array([mu]), success=success,
                    objective_value=float(sigma2_target),
                    n_iter=int(sol.nfev) if hasattr(sol, "nfev") else 0,
                    window_family=window, window_scale=float(scale),
                    diagnostics={"a_2": float(a2), "sigma2_target": float(sigma2_target),
                                 "score_norm": score_norm,
                                 "kappa_3_w_init": float(k3_0),
                                 "kappa_4_w_init": float(k4_0)})


def weak_pmm2_regression(x: np.ndarray, y: np.ndarray, window: str, scale: float,
                         beta0: Optional[np.ndarray] = None) -> Estimate:
    """Degree-2 weak PMM for linear regression.

    Solves the EXACT estimating equation
        g(beta) = X^T [ w(r) * (r + a_2 (r^2 - sigma2_target)) ] / sum w(r) = 0,
    where r = y - X beta and the window weights w(r) depend on beta. The
    root is found with scipy.optimize.root (numerical Jacobian, so the
    beta-dependence of w through r is fully accounted for), and `success`
    requires the final score norm to be below _SCORE_TOL.

    sigma2_target and a_2 are frozen from the initial (LAD) residuals. Under
    symmetric residuals a_2 -> 0 and the score reduces to the degree-1 weak
    PMM normal equations (= corresponding M-estimator).
    """
    if beta0 is None:
        beta0 = lad_regression(x, y).beta_hat
    w_fn = get_window(window)
    X = np.column_stack([np.ones_like(x), x])
    beta0 = beta0.astype(float).copy()

    r0 = y - X @ beta0
    w0 = w_fn(r0, scale)
    if float(w0.sum()) <= 0:
        return Estimate(beta_hat=beta0, success=False, n_iter=0,
                        window_family=window, window_scale=float(scale))
    _, sigma2_target, k3_0, k4_0 = _weak_cumulants_residual(r0, w0)
    a2 = _pmm2_a2_coefficient(sigma2_target, k3_0, k4_0)

    wp_fn = get_window_deriv(window)

    def score(beta: np.ndarray) -> np.ndarray:
        r = y - X @ beta
        w = w_fn(r, scale)
        sw = float(w.sum())
        if sw <= 0:
            return np.zeros(2)
        return X.T @ (w * (r + a2 * (r ** 2 - sigma2_target))) / sw

    def jac(beta: np.ndarray) -> np.ndarray:
        r = y - X @ beta
        w = w_fn(r, scale)
        wp = wp_fn(r, scale)
        sw = float(w.sum())
        if sw <= 0:
            return np.eye(2)
        s = r + a2 * (r ** 2 - sigma2_target)
        N = X.T @ (w * s)              # 2-vector
        Xtwp = X.T @ wp                # 2-vector
        p = wp * s + w * (1.0 + 2.0 * a2 * r)
        XtpX = X.T @ (p[:, None] * X)  # 2x2
        return -XtpX / sw + np.outer(N, Xtwp) / sw ** 2

    # Symmetric case: a_2 ~ 0 -> exact degree-1 IRWLS fixed point.
    if abs(a2) < 1e-12:
        beta, success_irls, n_iter = _irwls_regression(
            x, y, lambda r: w_fn(r, scale), beta0
        )
        sn = float(np.linalg.norm(score(beta)))
        return Estimate(beta_hat=beta, success=bool(success_irls and sn < _SCORE_TOL),
                        objective_value=float(sigma2_target), n_iter=n_iter,
                        window_family=window, window_scale=float(scale),
                        diagnostics={"a_2": float(a2), "sigma2_target": float(sigma2_target),
                                     "score_norm": sn, "kappa_3_w_init": float(k3_0),
                                     "kappa_4_w_init": float(k4_0)})

    sol = root(score, beta0, jac=jac, method="hybr",
               options={"xtol": 1e-10, "maxfev": 4000})
    beta = sol.x
    score_norm = float(np.linalg.norm(score(beta)))
    success = bool(sol.success) and score_norm < _SCORE_TOL
    return Estimate(beta_hat=beta, success=success,
                    objective_value=float(sigma2_target),
                    n_iter=int(sol.nfev) if hasattr(sol, "nfev") else 0,
                    window_family=window, window_scale=float(scale),
                    diagnostics={"a_2": float(a2), "sigma2_target": float(sigma2_target),
                                 "score_norm": score_norm,
                                 "kappa_3_w_init": float(k3_0),
                                 "kappa_4_w_init": float(k4_0)})


# ---------- degree-3 weak PMM (Kunchenko PMM-3, odd cubic, kurtosis-coupled) ----------

# Threshold on |m4 - 3 m2^2|: below this the windowed residual distribution is
# effectively mesokurtic (Gaussian-like), kappa -> inf, and PMM-3 reduces to the
# degree-1 weak PMM (linear score dominates). We then fall back to degree-1.
_PMM3_MESO_TOL = 1e-3


def _weak_pmm3_kappa(r: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Windowed PMM-3 kappa coefficient from residual moments.

    kappa^w = (m_6^w - 3 m_4^w m_2^w) / (m_4^w - 3 (m_2^w)^2),
    computed from windowed CENTRAL moments. Returns (kappa, m2, denom).
    When |denom| is tiny the distribution is mesokurtic; caller falls back
    to degree-1.
    """
    sw = float(w.sum())
    if sw <= 0:
        return float("nan"), 0.0, 0.0
    m1 = float((w * r).sum() / sw)
    c = r - m1
    m2 = float((w * c ** 2).sum() / sw)
    m4 = float((w * c ** 4).sum() / sw)
    m6 = float((w * c ** 6).sum() / sw)
    denom = m4 - 3.0 * m2 ** 2
    if abs(denom) < _PMM3_MESO_TOL * max(m2 ** 2, 1e-12):
        return float("inf"), m2, denom
    kappa = (m6 - 3.0 * m4 * m2) / denom
    return kappa, m2, denom


def weak_pmm3_location(y: np.ndarray, window: str, scale: float,
                       mu0: Optional[float] = None) -> Estimate:
    """Degree-3 weak PMM for location (Kunchenko PMM-3, odd cubic score).

    Solves the EXACT estimating equation
        g(mu) = sum_i w(r_i) [kappa^w r_i - r_i^3] / sum_i w(r_i) = 0,
    r_i = y_i - mu, weights windowed at current mu, kappa^w frozen from the
    initial (median) residuals. This is the symmetric-error analog: the score
    is ODD, so it stays valid for symmetric noise (where degree-2 is useless),
    and the cubic term couples to windowed kurtosis gamma_4^w.

    Requires a fast-decaying window (gaussian / tukey_compact / hann_value) so
    that the windowed sixth moment m_6^w is finite and stable; cauchy_like is
    too heavy-tailed for a stable kappa^w under Cauchy residuals.

    When the windowed residual distribution is mesokurtic (kappa^w -> inf),
    falls back to the degree-1 weak PMM (weighted mean).
    """
    w_fn = get_window(window)
    wp_fn = get_window_deriv(window)
    mu_init = float(np.median(y)) if mu0 is None else float(mu0)

    r0 = y - mu_init
    w0 = w_fn(r0, scale)
    if float(w0.sum()) <= 0:
        return Estimate(beta_hat=np.array([mu_init]), success=False, n_iter=0,
                        window_family=window, window_scale=float(scale))
    kappa, m2_0, denom0 = _weak_pmm3_kappa(r0, w0)

    if not np.isfinite(kappa):
        est = weak_pmm_location(y, window, scale, mu0=mu_init)
        est.diagnostics = {"reduced_to_degree1": True, "kappa_w": float("inf"),
                           "m2_w_init": float(m2_0)}
        return est

    def score(mu_arr: np.ndarray) -> np.ndarray:
        mu = float(mu_arr[0])
        r = y - mu
        w = w_fn(r, scale)
        sw = float(w.sum())
        if sw <= 0:
            return np.array([0.0])
        return np.array([float(np.sum(w * (kappa * r - r ** 3)) / sw)])

    def jac(mu_arr: np.ndarray) -> np.ndarray:
        mu = float(mu_arr[0])
        r = y - mu
        w = w_fn(r, scale)
        wp = wp_fn(r, scale)
        sw = float(w.sum())
        if sw <= 0:
            return np.array([[1.0]])
        s = kappa * r - r ** 3
        sp = kappa - 3.0 * r ** 2
        N = float(np.sum(w * s))
        p = wp * s + w * sp
        dN = -float(np.sum(p))
        dsw = -float(np.sum(wp))
        return np.array([[(dN * sw - N * dsw) / sw ** 2]])

    sol = root(score, np.array([mu_init]), jac=jac, method="hybr",
               options={"xtol": 1e-10, "maxfev": 2000})
    mu = float(sol.x[0])
    score_norm = float(abs(score(np.array([mu]))[0]))
    success = bool(sol.success) and score_norm < _SCORE_TOL
    return Estimate(beta_hat=np.array([mu]), success=success,
                    objective_value=float(m2_0),
                    n_iter=int(sol.nfev) if hasattr(sol, "nfev") else 0,
                    window_family=window, window_scale=float(scale),
                    diagnostics={"kappa_w": float(kappa), "m2_w_init": float(m2_0),
                                 "denom_w_init": float(denom0),
                                 "score_norm": score_norm,
                                 "reduced_to_degree1": False})


def weak_pmm3_regression(x: np.ndarray, y: np.ndarray, window: str, scale: float,
                         beta0: Optional[np.ndarray] = None) -> Estimate:
    """Degree-3 weak PMM for linear regression (odd cubic, kurtosis-coupled).

    Solves the EXACT estimating equation
        g(beta) = X^T [ w(r) (kappa^w r - r^3) ] / sum w(r) = 0,
    with kappa^w frozen from initial (LAD) residuals and an analytic Jacobian
    (including w'(r)). Falls back to degree-1 weak PMM when mesokurtic.
    """
    if beta0 is None:
        beta0 = lad_regression(x, y).beta_hat
    w_fn = get_window(window)
    wp_fn = get_window_deriv(window)
    X = np.column_stack([np.ones_like(x), x])
    beta0 = beta0.astype(float).copy()

    r0 = y - X @ beta0
    w0 = w_fn(r0, scale)
    if float(w0.sum()) <= 0:
        return Estimate(beta_hat=beta0, success=False, n_iter=0,
                        window_family=window, window_scale=float(scale))
    kappa, m2_0, denom0 = _weak_pmm3_kappa(r0, w0)

    if not np.isfinite(kappa):
        est = weak_pmm_regression(x, y, window, scale, beta0=beta0)
        est.diagnostics = {"reduced_to_degree1": True, "kappa_w": float("inf"),
                           "m2_w_init": float(m2_0)}
        return est

    def score(beta: np.ndarray) -> np.ndarray:
        r = y - X @ beta
        w = w_fn(r, scale)
        sw = float(w.sum())
        if sw <= 0:
            return np.zeros(2)
        return X.T @ (w * (kappa * r - r ** 3)) / sw

    def jac(beta: np.ndarray) -> np.ndarray:
        r = y - X @ beta
        w = w_fn(r, scale)
        wp = wp_fn(r, scale)
        sw = float(w.sum())
        if sw <= 0:
            return np.eye(2)
        s = kappa * r - r ** 3
        sp = kappa - 3.0 * r ** 2
        N = X.T @ (w * s)
        Xtwp = X.T @ wp
        p = wp * s + w * sp
        XtpX = X.T @ (p[:, None] * X)
        return -XtpX / sw + np.outer(N, Xtwp) / sw ** 2

    sol = root(score, beta0, jac=jac, method="hybr",
               options={"xtol": 1e-10, "maxfev": 4000})
    beta = sol.x
    score_norm = float(np.linalg.norm(score(beta)))
    success = bool(sol.success) and score_norm < _SCORE_TOL
    return Estimate(beta_hat=beta, success=success, objective_value=float(m2_0),
                    n_iter=int(sol.nfev) if hasattr(sol, "nfev") else 0,
                    window_family=window, window_scale=float(scale),
                    diagnostics={"kappa_w": float(kappa), "m2_w_init": float(m2_0),
                                 "denom_w_init": float(denom0),
                                 "score_norm": score_norm,
                                 "reduced_to_degree1": False})


# ---------- weak-CF: model-free symmetric characteristic-function estimator ----------

# CF-PMM (kunchenko-pmm-cf skill), model-free symmetric branch. For symmetric
# noise centered at the true parameter, E[sin(u*eps)] = 0 for every frequency u.
# So the residual ECF imaginary part vanishes at the truth. We solve the
# design-weighted CF estimating equations
#     sum_i sin(u*r_i) = 0   and   sum_i sin(u*r_i) x_i = 0   (for u in U)
# combined as a Simple-CF GMM (weights = 1/M, W = I). The score sin(u*r) is
# BOUNDED (|sin| <= 1): unlike the polynomial PMM scores (r, r^3, ...), there is
# no tail amplification, so this needs no fast-decay window and stays stable even
# where windowed high moments diverge (alpha-stable alpha < 1.5).
#
# Frequency grid (Feuerverger-McDunnough 1981; skill defaults): MAD-adaptive
# log-grid, u_max ~ pi / robust_scale so |f(u_max)| stays informative.


def _cf_freq_grid(scale: float, n_freq: int, n_obs: int,
                  u_max_mult: float = 1.0) -> np.ndarray:
    u_max = u_max_mult * np.pi / max(scale, 1e-6)
    u_min = u_max / max(n_obs, 10)
    return np.geomspace(u_min, u_max, n_freq)


def weak_cf_location(y: np.ndarray, n_freq: int = 24, u_max_mult: float = 1.0,
                     mu0: Optional[float] = None) -> Estimate:
    """Model-free symmetric weak-CF location estimator.

    Minimizes sum_u [ (1/N) sum_i sin(u (y_i - mu)) ]^2 over a MAD-adaptive
    log frequency grid. Moment-free and bounded; robust to arbitrarily heavy
    tails (no moment of any order is required to exist).
    """
    mu_init = float(np.median(y)) if mu0 is None else float(mu0)
    scale = robust_scale_mad(y - np.median(y))
    U = _cf_freq_grid(scale if scale > 0 else 1.0, n_freq, len(y), u_max_mult)

    def moments(mu_arr: np.ndarray) -> np.ndarray:
        r = y - mu_arr[0]
        return np.array([np.mean(np.sin(u * r)) for u in U])

    res = least_squares(moments, np.array([mu_init]), method="lm",
                        xtol=1e-10, ftol=1e-12, max_nfev=2000)
    mu = float(res.x[0])
    return Estimate(beta_hat=np.array([mu]), success=bool(res.success),
                    objective_value=float(np.sum(res.fun ** 2)),
                    n_iter=int(res.nfev), window_family="weak_cf",
                    window_scale=float(U[-1]),
                    diagnostics={"n_freq": int(n_freq), "u_max": float(U[-1]),
                                 "u_min": float(U[0])})


def weak_cf_regression(x: np.ndarray, y: np.ndarray, n_freq: int = 24,
                       u_max_mult: float = 1.0,
                       beta0: Optional[np.ndarray] = None,
                       weighting: str = "identity") -> Estimate:
    """Model-free symmetric weak-CF linear regression estimator.

    Solves the design-weighted CF estimating equations
        sum_i sin(u r_i) = 0,  sum_i sin(u r_i) x_i = 0   (u in grid U)
    via least_squares from a LAD start. With ``weighting='identity'`` this is
    the Simple-CF GMM (W = I); with ``weighting='2step'`` the second step uses
    the efficient weight W = Omega^{-1}, Omega the empirical moment covariance
    at the first-step estimate (minimize g' W g = ||L' g||^2, W = L L').
    Bounded, moment-free score; no window or high-moment fragility. Diagnostics
    report the moment-Jacobian's smallest singular value (identification: > 0
    means full rank) and condition number.
    """
    X = np.column_stack([np.ones_like(x), x])
    if beta0 is None:
        beta0 = lad_regression(x, y).beta_hat
    beta0 = beta0.astype(float)
    scale = robust_scale_mad(y - X @ beta0)
    U = _cf_freq_grid(scale if scale > 0 else 1.0, n_freq, len(y), u_max_mult)

    def per_obs(beta: np.ndarray) -> np.ndarray:
        r = y - X @ beta
        Gi = np.empty((len(y), 2 * len(U)))
        for j, u in enumerate(U):
            s = np.sin(u * r)
            Gi[:, 2 * j] = s
            Gi[:, 2 * j + 1] = s * x
        return Gi

    def moments(beta: np.ndarray) -> np.ndarray:
        return per_obs(beta).mean(axis=0)

    res = least_squares(moments, beta0, method="lm",
                        xtol=1e-10, ftol=1e-12, max_nfev=4000)
    beta = res.x
    diag = {"n_freq": int(n_freq), "u_max": float(U[-1]), "u_min": float(U[0]),
            "weighting": weighting}

    if weighting == "2step":
        Gi = per_obs(beta)
        Omega = (Gi.T @ Gi) / len(y) + 1e-8 * np.eye(2 * len(U))
        try:
            L = np.linalg.cholesky(np.linalg.inv(Omega))

            def wmoments(b: np.ndarray) -> np.ndarray:
                return L.T @ moments(b)

            res = least_squares(wmoments, beta, method="lm",
                                xtol=1e-10, ftol=1e-12, max_nfev=4000)
            beta = res.x
        except np.linalg.LinAlgError:
            diag["twostep_failed"] = True

    sv = np.linalg.svd(res.jac, compute_uv=False)
    diag["jac_smallest_sv"] = float(sv.min())
    diag["jac_cond"] = float(sv.max() / max(sv.min(), 1e-300))

    return Estimate(beta_hat=beta, success=bool(res.success),
                    objective_value=float(np.sum(res.fun ** 2)),
                    n_iter=int(res.nfev), window_family="weak_cf",
                    window_scale=float(U[-1]), diagnostics=diag)


def sample_mean_location(y: np.ndarray) -> Estimate:
    return Estimate(beta_hat=np.array([float(np.mean(y))]), success=True,
                    window_family="sample_mean")


def median_location(y: np.ndarray) -> Estimate:
    return Estimate(beta_hat=np.array([float(np.median(y))]), success=True,
                    window_family="median")


# ---------------------------------------------------------------------------
# High-breakdown baselines: S- and MM-estimators (Tukey biweight)
#
# Referee request: standard high-breakdown competitors (S/MM) alongside the
# degree-1 = M-estimator members. Fast-S follows Salibian-Barrera & Yohai
# (elemental subsets + concentration steps, keep the smallest M-scale); MM
# refines the S-fit with a high-efficiency (c=4.685) Tukey M-step at the
# fixed S-scale (Yohai 1987).
# ---------------------------------------------------------------------------

def _tukey_rho_norm(u: np.ndarray, c: float) -> np.ndarray:
    """Normalized Tukey biweight rho in [0,1]: 0 at 0, 1 for |u| >= c."""
    a = np.clip(np.abs(u) / c, 0.0, 1.0)
    return 1.0 - (1.0 - a ** 2) ** 3


def _tukey_weight_uc(u: np.ndarray, c: float) -> np.ndarray:
    """Tukey biweight IRLS weight (1-(u/c)^2)^2 for |u|<c, else 0."""
    a = np.abs(u) / c
    w = (1.0 - a ** 2) ** 2
    w[a >= 1.0] = 0.0
    return w


def _m_scale(r: np.ndarray, c: float = 1.547, delta: float = 0.5,
             tol: float = 1e-8, max_iter: int = 100) -> float:
    """Robust M-scale: solve mean(rho_norm(r/s)) = delta by bisection
    (mean is monotone decreasing in s)."""
    r = np.asarray(r, float)
    s0 = float(np.median(np.abs(r))) / 0.6745
    if s0 <= 0.0:
        s0 = float(np.std(r))
    if s0 <= 0.0:
        return 0.0
    lo, hi = s0 * 1e-3, s0 * 1e3
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if float(np.mean(_tukey_rho_norm(r / mid, c))) > delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * mid:
            break
    return 0.5 * (lo + hi)


def s_estimator_regression(x: np.ndarray, y: np.ndarray, c: float = 1.547,
                           delta: float = 0.5, n_resamples: int = 200,
                           n_refine: int = 2, seed: int = 0) -> Estimate:
    """Fast-S regression estimator (Tukey biweight, ~50% breakdown)."""
    import time
    t0 = time.perf_counter()
    X = np.column_stack([np.ones_like(x), x])
    n = len(y)
    rng = np.random.default_rng(seed)

    def refine(beta: np.ndarray) -> tuple[np.ndarray, float]:
        b = np.asarray(beta, float).copy()
        for _ in range(n_refine):
            r = y - X @ b
            s = _m_scale(r, c, delta)
            if s <= 0.0:
                break
            w = _tukey_weight_uc(r / s, c)
            if w.sum() <= 0.0:
                break
            XtW = X.T * w
            try:
                b = np.linalg.solve(XtW @ X, XtW @ y)
            except np.linalg.LinAlgError:
                break
        return b, _m_scale(y - X @ b, c, delta)

    cand = [lad_regression(x, y).beta_hat]
    for _ in range(n_resamples):
        idx = rng.choice(n, size=2, replace=False)
        try:
            cand.append(np.linalg.solve(X[idx], y[idx]))
        except np.linalg.LinAlgError:
            continue

    best_b, best_s = None, np.inf
    for b0 in cand:
        b, s = refine(np.asarray(b0, float))
        if np.isfinite(s) and s < best_s:
            best_s, best_b = s, b
    if best_b is None:
        best_b = lad_regression(x, y).beta_hat
        best_s = _m_scale(y - X @ best_b, c, delta)
    return Estimate(beta_hat=best_b, success=True, objective_value=best_s,
                    runtime_sec=time.perf_counter() - t0,
                    window_family="s_estimator", window_scale=best_s,
                    diagnostics={"m_scale": best_s})


def mm_estimator_regression(x: np.ndarray, y: np.ndarray, c_S: float = 1.547,
                            c_M: float = 4.685, n_resamples: int = 200,
                            seed: int = 0, max_iter: int = 100,
                            tol: float = 1e-9) -> Estimate:
    """MM-estimator: high-efficiency (c=4.685) Tukey M-step at the fixed
    S-scale, initialized at the S-fit (Yohai 1987). ~50% breakdown, 95%
    Gaussian efficiency."""
    import time
    t0 = time.perf_counter()
    X = np.column_stack([np.ones_like(x), x])
    s_est = s_estimator_regression(x, y, c=c_S, n_resamples=n_resamples, seed=seed)
    scale = s_est.window_scale
    beta = np.asarray(s_est.beta_hat, float).copy()
    if scale <= 0.0:
        return Estimate(beta_hat=beta, success=True, window_family="mm_estimator",
                        window_scale=0.0, diagnostics={"m_scale": 0.0})
    success, it_used = False, 0
    for it in range(max_iter):
        r = y - X @ beta
        w = _tukey_weight_uc(r / scale, c_M)
        if w.sum() <= 0.0:
            break
        XtW = X.T * w
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ y)
        except np.linalg.LinAlgError:
            break
        it_used = it + 1
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            success = True
            break
        beta = beta_new
    return Estimate(beta_hat=beta, success=success, n_iter=it_used,
                    runtime_sec=time.perf_counter() - t0,
                    window_family="mm_estimator", window_scale=scale,
                    diagnostics={"m_scale": scale, "c_M": c_M, "s_beta": s_est.beta_hat})


# ---------------------------------------------------------------------------
# Modern heavy-tailed (deviation-optimal) baselines: adaptive Huber, Catoni.
#
# Referee request: position the weak-moment family against the sub-Gaussian
# deviation literature (paper Sec. "Modern heavy-tailed regression"). Both use
# a sample-size-adaptive robustification parameter tau ~ sigma * sqrt(n / t),
# t = log n (Sun-Zhou-Fan 2020; Catoni 2012). Crucially both scores are
# *non-redescending* (Huber's psi is monotone+bounded; Catoni's grows ~log|r|),
# so unlike the redescending weak-moment members they down-weight but never hard-
# reject gross outliers -- the trade-off the experiments quantify.
# ---------------------------------------------------------------------------

def _adaptive_tau(sc: float, n: int, tau_mult: float) -> float:
    """Sun-Zhou-Fan robustification parameter tau = tau_mult * sigma * sqrt(n/log n)."""
    t = max(np.log(max(n, 3)), 1.0)
    return float(tau_mult) * sc * float(np.sqrt(max(n, 1) / t))


def adaptive_huber_regression(x: np.ndarray, y: np.ndarray, tau_mult: float = 1.0,
                              beta0: Optional[np.ndarray] = None) -> Estimate:
    """Adaptive Huber regression (Sun, Zhou & Fan 2020).

    Huber M-estimator with the *sample-size-adaptive* threshold
    ``tau = tau_mult * (MAD/0.6745) * sqrt(n / log n)`` (vs the fixed
    c = 1.345*sigma of the classical Huber in ``huber_regression``). The Huber
    score is bounded and MONOTONE (non-redescending), giving sub-Gaussian
    deviations under a finite (1+delta) moment but no hard outlier rejection.
    """
    b_lad = lad_regression(x, y).beta_hat if beta0 is None else np.asarray(beta0, float)
    X = np.column_stack([np.ones_like(x), x])
    sc = robust_scale_mad(y - X @ b_lad)
    if sc <= 0.0:
        return Estimate(beta_hat=b_lad, success=True,
                        window_family="adaptive_huber", window_scale=0.0)
    tau = _adaptive_tau(sc, len(y), tau_mult)

    def weight_fn(r: np.ndarray) -> np.ndarray:
        a = np.abs(r)
        w = np.ones_like(r, dtype=float)
        m = a > tau
        w[m] = tau / np.maximum(a[m], 1e-15)
        return w

    beta, success, n_iter = _irwls_regression(x, y, weight_fn, b_lad)
    return Estimate(beta_hat=beta, success=success, n_iter=n_iter,
                    window_family="adaptive_huber", window_scale=float(tau),
                    diagnostics={"tau": float(tau), "mad_scale": float(sc),
                                 "tau_mult": float(tau_mult)})


def catoni_regression(x: np.ndarray, y: np.ndarray, tau_mult: float = 1.0,
                      beta0: Optional[np.ndarray] = None) -> Estimate:
    """Catoni-type robust regression (Catoni 2012 influence function).

    Solves ``sum_i tau * psi_C(r_i / tau) * X_i = 0`` with the Catoni influence
    ``psi_C(u) = sign(u) * log(1 + |u| + u^2/2)`` and the same adaptive scale
    ``tau = tau_mult * (MAD/0.6745) * sqrt(n / log n)`` as adaptive Huber. The
    influence grows only logarithmically (soft, non-redescending), the
    deviation-optimal alternative to a hard clip.
    """
    b_lad = lad_regression(x, y).beta_hat if beta0 is None else np.asarray(beta0, float)
    X = np.column_stack([np.ones_like(x), x])
    sc = robust_scale_mad(y - X @ b_lad)
    if sc <= 0.0:
        return Estimate(beta_hat=b_lad, success=True,
                        window_family="catoni", window_scale=0.0)
    tau = _adaptive_tau(sc, len(y), tau_mult)

    def weight_fn(r: np.ndarray) -> np.ndarray:
        u = r / tau
        a = np.abs(u)
        psi = np.sign(u) * np.log1p(a + 0.5 * u ** 2)   # Catoni influence
        # IRLS weight w(r) so that w(r)*r = tau*psi_C(u): w = tau*psi/r = psi/u.
        w = np.ones_like(r, dtype=float)
        nz = a > 1e-12
        w[nz] = psi[nz] / u[nz]
        return np.clip(w, 0.0, 1.0)

    beta, success, n_iter = _irwls_regression(x, y, weight_fn, b_lad)
    return Estimate(beta_hat=beta, success=success, n_iter=n_iter,
                    window_family="catoni", window_scale=float(tau),
                    diagnostics={"tau": float(tau), "mad_scale": float(sc),
                                 "tau_mult": float(tau_mult)})
