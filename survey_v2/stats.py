"""Thống kê cho CV lặp: t-test hiệu chỉnh Nadeau–Bengio, Holm, Bayesian correlated t-test (ROPE), CI.

* Nadeau & Bengio (2003): với J lần đánh giá của CV lặp, phương sai của trung bình chênh lệch d̄ được
  hiệu chỉnh thành (1/J + n_test/n_train)·s², vì các tập train chồng lấn làm các d_j không độc lập.
  t = d̄ / sqrt((1/J + ρ)·s²), bậc tự do J − 1 (ρ = n_test/n_train = 1/4 với 5 fold).
* Holm (1979): hiệu chỉnh đa kiểm định trong từng họ (survey: mỗi phương pháp × 5 mức k).
* Benavoli et al. (2017), Corani & Benavoli (2015): hậu nghiệm của chênh lệch trung bình là Student t
  với vị trí d̄, thang sqrt((1/J + ρ)·s²), bậc tự do J − 1; báo P(left), P(rope), P(right) với
  ROPE = ±r (r = 0,01 AUROC).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st

from . import config as C


def nadeau_bengio_ttest(d: np.ndarray, ratio: float = C.NB_TEST_TRAIN_RATIO, alternative: str = "greater"):
    """d: chênh lệch theo từng lần đánh giá (J,). Trả về dict(mean, sd, se_nb, t, df, p, ci_low, ci_high)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    J = len(d)
    if J < 2:
        return {"n": J, "mean": float(d.mean()) if J else np.nan, "sd": np.nan, "se_nb": np.nan,
                "t": np.nan, "df": J - 1, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = float(np.sqrt((1.0 / J + ratio) * sd**2))
    if se == 0:
        t = np.inf if mean > 0 else (-np.inf if mean < 0 else 0.0)
    else:
        t = mean / se
    df = J - 1
    if alternative == "greater":
        p = float(st.t.sf(t, df))
    elif alternative == "less":
        p = float(st.t.cdf(t, df))
    else:
        p = float(2 * st.t.sf(abs(t), df))
    q = st.t.ppf(0.975, df)
    return {"n": J, "mean": mean, "sd": sd, "se_nb": se, "t": float(t), "df": df, "p": p,
            "ci_low": mean - q * se, "ci_high": mean + q * se}


def holm(pvals) -> np.ndarray:
    """Hiệu chỉnh Holm (step-down); NaN giữ nguyên NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = len(q)
    if m == 0:
        return out
    order = np.argsort(q, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order]).clip(max=1.0)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def bayesian_correlated_ttest(d: np.ndarray, rope: float = C.ROPE_AUROC, ratio: float = C.NB_TEST_TRAIN_RATIO):
    """Trả về dict(p_left, p_rope, p_right, mean, scale, df) với ROPE = [−rope, +rope]."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    J = len(d)
    if J < 2:
        return {"n": J, "p_left": np.nan, "p_rope": np.nan, "p_right": np.nan, "mean": np.nan, "scale": np.nan, "df": J - 1}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    scale = float(np.sqrt((1.0 / J + ratio) * sd**2))
    df = J - 1
    if scale == 0:
        left, right = float(mean < -rope), float(mean > rope)
        return {"n": J, "p_left": left, "p_rope": 1.0 - left - right, "p_right": right,
                "mean": mean, "scale": scale, "df": df}
    dist = st.t(df, loc=mean, scale=scale)
    p_left = float(dist.cdf(-rope))
    p_right = float(dist.sf(rope))
    return {"n": J, "p_left": p_left, "p_rope": 1.0 - p_left - p_right, "p_right": p_right,
            "mean": mean, "scale": scale, "df": df}


def mean_ci(x: np.ndarray, ratio: float = C.NB_TEST_TRAIN_RATIO):
    """Trung bình, SD và CI 95% (phương sai Nadeau–Bengio) của một chỉ số qua các lần đánh giá."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    J = len(x)
    if J == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if J == 1:
        return {"n": 1, "mean": float(x[0]), "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = float(np.sqrt((1.0 / J + ratio) * sd**2))
    q = st.t.ppf(0.975, J - 1)
    return {"n": J, "mean": mean, "sd": sd, "ci_low": mean - q * se, "ci_high": mean + q * se}
