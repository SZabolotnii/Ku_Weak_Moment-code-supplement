from __future__ import annotations

import numpy as np

from ku_weak_moment.estimators import (
    cauchy_mle_regression,
    huber_regression,
    ols_regression,
    weak_pmm_regression,
)
from ku_weak_moment.moments import raw_moment, weak_moment
from ku_weak_moment.windows import (
    cauchy_like_window,
    gaussian_window,
    hann_value_window,
    tukey_compact_window,
)


def test_windows_are_nonnegative_and_shape_preserving():
    r = np.linspace(-3, 3, 31)
    for fn in [gaussian_window, cauchy_like_window, tukey_compact_window, hann_value_window]:
        w = fn(r, 1.5)
        assert w.shape == r.shape
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0)


def test_wide_gaussian_window_recovers_raw_second_moment():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(weak_moment(x, 2, "gaussian", 1e6) - raw_moment(x, 2)) < 1e-9


def test_regression_estimators_recover_clean_line():
    x = np.linspace(-1.0, 1.0, 41)
    y = 0.5 + 1.0 * x
    for fit in [
        ols_regression(x, y),
        huber_regression(x, y),
        cauchy_mle_regression(x, y, gamma=1.0),
        weak_pmm_regression(x, y, "cauchy_like", scale=1.0),
    ]:
        assert fit.success
        assert np.allclose(fit.beta_hat, np.array([0.5, 1.0]), atol=1e-7)
