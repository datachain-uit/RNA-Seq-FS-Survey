"""Chỉ số ổn định: Nogueira (chính, kèm CI), Kuncheva và Jaccard (đối chiếu bản v1), chồng lấn giữa phương pháp.

Nogueira, Sechidis & Brown (2018), JMLR 18(174) — đoạn code H của Review 24/10/2026 (M11):
    Z: ma trận nhị phân M × d (M lần chọn, d gen trong KHÔNG GIAN CHUNG); Z[i, f] = 1 nếu lần i chọn gen f.
    p_f = tần suất chọn gen f; k̄ = số gen chọn trung bình.
    Φ = 1 − (M/(M−1))·mean_f[p_f (1 − p_f)] / ((k̄/d)(1 − k̄/d)).
    CI 95% từ phương sai tiệm cận (chỉ hợp lệ khi các lần chọn độc lập — lấy mẫu con ngẫu nhiên;
    với các fold của CV lặp, CI chỉ để tham khảo).

Không gian chung d: vì lọc sơ bộ nằm trong fold nên tập ứng viên khác nhau giữa các lần chọn; Z được
dựng trên TOÀN BỘ gen của bộ dữ liệu (gen không qua lọc = 0). Điều này đúng với định nghĩa của chỉ số
(d là số đặc trưng của bài toán).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st


def selections_to_matrix(selected_lists, universe) -> np.ndarray:
    """selected_lists: M tập tên gen; universe: danh sách d gen. Trả về Z (M × d, int8)."""
    index = {g: j for j, g in enumerate(universe)}
    Z = np.zeros((len(selected_lists), len(universe)), dtype=np.int8)
    for i, genes in enumerate(selected_lists):
        Z[i, [index[g] for g in genes]] = 1
    return Z


def nogueira(Z: np.ndarray, alpha: float = 0.05) -> dict:
    Z = np.asarray(Z, dtype=float)
    M, d = Z.shape
    if M < 2:
        return {"stability": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "M": M, "d": d}
    p = Z.mean(axis=0)
    kbar = p.sum()
    denom = (kbar / d) * (1 - kbar / d)
    if denom == 0:
        return {"stability": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "M": M, "d": d}
    phi = 1 - (M / (M - 1)) * np.mean(p * (1 - p)) / denom
    k = Z.sum(axis=1)
    phi_i = (1 / denom) * ((Z * p).mean(axis=1) - k * kbar / d**2
                           + (phi / 2) * (2 * k * kbar / d**2 - k / d - kbar / d + 1))
    se = np.sqrt((4 / M**2) * np.sum((phi_i - phi_i.mean()) ** 2))
    z = st.norm.ppf(1 - alpha / 2)
    return {"stability": float(phi), "se": float(se), "ci_low": float(phi - z * se),
            "ci_high": float(phi + z * se), "M": M, "d": d}


def pairwise_mean(sets, func) -> float:
    sets = [set(s) for s in sets]
    vals = [func(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(vals)) if vals else np.nan


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def kuncheva_index(sets, d: int) -> float:
    """Kuncheva (2007) cho các tập CÙNG kích thước k trong không gian d gen (hiệu chỉnh ngẫu nhiên)."""
    sets = [set(s) for s in sets]
    ks = {len(s) for s in sets}
    if len(ks) != 1:
        raise ValueError("Kuncheva yêu cầu các tập cùng kích thước")
    k = ks.pop()
    if k == 0 or k == d:
        return np.nan

    def ku(a, b):
        return (len(a & b) * d - k * k) / (k * (d - k))

    return pairwise_mean(sets, ku)


def mean_jaccard(sets) -> float:
    return pairwise_mean(sets, jaccard)


def overlap_matrix(panels_by_method: dict) -> tuple[list, np.ndarray]:
    """Jaccard trung bình giữa các phương pháp; panels_by_method[m] = danh sách tập gen (một tập mỗi fold)."""
    names = list(panels_by_method)
    n = len(names)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            pa, pb = panels_by_method[a], panels_by_method[b]
            vals = [jaccard(set(x), set(y)) for x, y in zip(pa, pb)]
            M[i, j] = float(np.nanmean(vals)) if vals else np.nan
    return names, M
