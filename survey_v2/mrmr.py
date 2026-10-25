"""mRMR theo Peng, Long & Ding (2005), IEEE TPAMI 27(8): tiêu chí MID / MIQ trên dữ liệu rời rạc hóa.

Rời rạc hóa từng gen thành 3 mức theo ngưỡng μ ± σ/2 (tính trên dữ liệu được đưa vào, tức phần train
của fold); thông tin tương hỗ tính từ bảng tần suất (không ước lượng mật độ). Chọn tham lam:

    MID:  x* = argmax_{j ∉ S} [ I(X_j; Y) − (1/|S|) Σ_{s ∈ S} I(X_j; X_s) ]
    MIQ:  x* = argmax_{j ∉ S} [ I(X_j; Y) / ((1/|S|) Σ_{s ∈ S} I(X_j; X_s)) ]

Mỗi bước tính I(X_j; X_s) cho mọi j bằng phép nhân ma trận one-hot, nên chi phí O(n·p) mỗi bước và
O(k·n·p) cho k gen (khớp bậc O(n d²) ở Bảng 5 của bài khi k ~ d). Thứ tự chọn là xếp hạng: tiền tố
k gen đầu chính là tập mRMR kích thước k (nested).
"""

from __future__ import annotations

import numpy as np

from . import config as C

_EPS = 1e-12


def discretize(X: np.ndarray, method: str = C.MRMR_DISCRETIZE) -> np.ndarray:
    """Trả về ma trận int8 n × p với giá trị {0, 1, 2}."""
    if method == "mean_half_sd":
        mu, sd = X.mean(0), X.std(0, ddof=1)
        lo, hi = mu - sd / 2.0, mu + sd / 2.0
        return (X > lo).astype(np.int8) + (X > hi).astype(np.int8)
    if method == "tertiles":
        q1, q2 = np.quantile(X, [1 / 3, 2 / 3], axis=0)
        return (X > q1).astype(np.int8) + (X > q2).astype(np.int8)
    raise ValueError(method)


def _onehot(D: np.ndarray, levels: int = 3) -> np.ndarray:
    """n × p × levels (float32) — one-hot của ma trận rời rạc."""
    n, p = D.shape
    out = np.zeros((n, p, levels), dtype=np.float32)
    rows = np.arange(n)[:, None]
    cols = np.arange(p)[None, :]
    out[rows, cols, D] = 1.0
    return out


def _mi_from_counts(counts: np.ndarray) -> np.ndarray:
    """counts: (..., a, b) bảng tần suất; trả về MI (nat) theo trục đầu."""
    n = counts.sum(axis=(-2, -1), keepdims=True)
    pxy = counts / np.maximum(n, 1)
    px = pxy.sum(axis=-1, keepdims=True)
    py = pxy.sum(axis=-2, keepdims=True)
    ratio = pxy / np.maximum(px * py, _EPS)
    term = np.where(pxy > 0, pxy * np.log(np.maximum(ratio, _EPS)), 0.0)
    return term.sum(axis=(-2, -1))


def relevance(D: np.ndarray, y: np.ndarray) -> np.ndarray:
    """I(X_j; Y) cho mọi gen j (D rời rạc n × p; y ∈ {0,1})."""
    H = _onehot(D)                                    # n × p × 3
    Y = np.stack([(y == 0), (y == 1)], axis=1).astype(np.float32)   # n × 2
    counts = np.einsum("npa,nb->pab", H, Y)           # p × 3 × 2
    return _mi_from_counts(counts)


def redundancy_with(D: np.ndarray, H: np.ndarray, s: int) -> np.ndarray:
    """I(X_j; X_s) cho mọi j; H = one-hot của D (n × p × 3)."""
    Hs = H[:, s, :]                                   # n × 3
    counts = np.einsum("npa,nb->pab", H, Hs)          # p × 3 × 3
    return _mi_from_counts(counts)


def mrmr_ranking(X: np.ndarray, y: np.ndarray, k: int, criterion: str = C.MRMR_CRITERION,
                 discretize_method: str = C.MRMR_DISCRETIZE) -> np.ndarray:
    """Thứ tự k gen được chọn tham lam (chỉ số cột của X). Gen đầu tiên là gen có I(X;Y) lớn nhất."""
    n, p = X.shape
    k = min(k, p)
    D = discretize(X, discretize_method)
    H = _onehot(D)
    rel = relevance(D, y)
    selected = [int(np.argmax(rel))]
    red_sum = np.zeros(p)
    mask = np.zeros(p, dtype=bool)
    mask[selected[0]] = True
    while len(selected) < k:
        red_sum += redundancy_with(D, H, selected[-1])
        red_mean = red_sum / len(selected)
        if criterion == "MID":
            score = rel - red_mean
        elif criterion == "MIQ":
            score = rel / np.maximum(red_mean, _EPS)
        else:
            raise ValueError(criterion)
        score[mask] = -np.inf
        j = int(np.argmax(score))
        selected.append(j)
        mask[j] = True
    return np.asarray(selected, dtype=int)
