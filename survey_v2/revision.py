"""Phần dùng chung của đợt hiệu chỉnh DMKD (B1–B7, QĐ171–QĐ188; đặc tả và quy tắc đọc chốt 12/11/2026).

* Đọc kết quả ĐÃ CÔNG BỐ (tag survey-v3-dmkd.1) từ ``results/survey_v2/`` của kho mã (``PUBLISHED_ROOT``) — không bao giờ ghi vào đó.
* Luật bỏ phiếu gốc của k-TSP (B2): hướng cặp học trên train, điểm = tỉ lệ cặp bỏ phiếu lớp 1, hoà thì cặp đầu quyết định.
* Nhãn cấp lần chia (B1, B6): Wilcoxon dấu hạng hai phía (chính xác), Holm trong từng chiến lược, CI bootstrap theo lần chia.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

from . import config as C
from .stats import holm

REPO = Path(__file__).resolve().parents[1]
# Kết quả của tag survey-v3-dmkd.1 (chỉ đọc). KLTN_PUBLISHED_ROOT chỉ dùng trong test trên dữ liệu giả; đường dẫn thực dùng được ghi
# vào config.json của mọi đầu ra (published_root).
PUBLISHED_ROOT = Path(os.environ.get("KLTN_PUBLISHED_ROOT") or (REPO / "results" / "survey_v2"))
PUBLISHED_TAG = "survey-v3-dmkd.1"

# --- B1/B6: nhãn cấp lần chia (quy tắc chốt 12/11) ----------------------------------------------------------------------
B1_SEEDS = tuple(range(42, 62))                             # 20 lần chia; 42–46 trùng bản chia đã công bố
B1_SPLITS_FILE = "fewshot_splits_20.csv"
SPLIT_LABEL_MARGIN = 0.01
SPLIT_BOOTSTRAP_N = 2000
GATE_TOL_AUROC = 1e-12                                       # cổng bằng nhau của B1, B4 (AUROC)


# --------------------------------------------------------------------------------------------------------------------------
# B2: k-TSP với luật bỏ phiếu gốc
# --------------------------------------------------------------------------------------------------------------------------
def ktsp_pairs_from_panel(cols) -> list[tuple[int, int]]:
    """Panel k-TSP lưu theo thứ tự [i1, j1, i2, j2, ...] (survey_v2.methods.rank_ktsp) -> danh sách cặp theo thứ hạng."""
    cols = list(cols)
    if len(cols) % 2:
        raise ValueError(f"panel k-TSP phải có số gen chẵn, có {len(cols)}")
    return [(int(cols[2 * m]), int(cols[2 * m + 1])) for m in range(len(cols) // 2)]


def ktsp_directions(X: np.ndarray, y: np.ndarray, pairs) -> tuple[np.ndarray, np.ndarray]:
    """Hướng mỗi cặp (i, j) học trên train: +1 nghĩa là "x_i > x_j bỏ phiếu lớp 1", khi P(x_i > x_j | 1) > P(x_i > x_j | 0);
    ngược lại −1. Trả về (hướng, mặt nạ hoà P bằng nhau — khi đó giữ +1 và ghi lại)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    dirs, ties = np.ones(len(pairs), dtype=int), np.zeros(len(pairs), dtype=bool)
    for m, (a, b) in enumerate(pairs):
        gt = X[:, a] > X[:, b]
        p1, p0 = gt[y == 1].mean(), gt[y == 0].mean()
        if p1 < p0:
            dirs[m] = -1
        elif p1 == p0:
            ties[m] = True
    return dirs, ties


def ktsp_vote(X: np.ndarray, pairs, dirs) -> tuple[np.ndarray, np.ndarray]:
    """(điểm = tỉ lệ cặp bỏ phiếu lớp 1, lớp dự đoán). Lớp 1 khi điểm > 0,5; điểm = 0,5 (số cặp chẵn, hoà) thì cặp đầu
    (Δ lớn nhất của k-TSP) quyết định. Không chuẩn hoá: luật chỉ dùng thứ tự trong mẫu."""
    X = np.asarray(X, dtype=float)
    votes = np.zeros((X.shape[0], len(pairs)), dtype=float)
    for m, (a, b) in enumerate(pairs):
        gt = X[:, a] > X[:, b]
        votes[:, m] = gt if dirs[m] > 0 else ~gt
    score = votes.mean(axis=1)
    pred = (score > 0.5).astype(int)
    tie = np.isclose(score, 0.5)
    pred[tie] = votes[tie, 0].astype(int)
    return score, pred


def evaluate_vote(Xtr, ytr, Xte, yte, pairs, dirs=None) -> dict:
    """Hướng học trên (Xtr, ytr) (hoặc dùng `dirs` cho trước), bỏ phiếu trên Xte. AUROC trên điểm; BAcc, MCC trên lớp."""
    if dirs is None:
        dirs, ties = ktsp_directions(Xtr, ytr, pairs)
    else:
        ties = np.zeros(len(pairs), dtype=bool)
    score, pred = ktsp_vote(Xte, pairs, dirs)
    yte = np.asarray(yte, dtype=int)
    auroc = float(roc_auc_score(yte, score)) if len(np.unique(yte)) == 2 else np.nan
    return {"auroc": auroc, "bacc": float(balanced_accuracy_score(yte, pred)), "mcc": float(matthews_corrcoef(yte, pred)),
            "n_pairs": len(pairs), "n_direction_ties": int(ties.sum()), "n_score_ties": int(np.isclose(score, 0.5).sum()),
            "directions": ";".join(str(int(d)) for d in dirs)}


# --------------------------------------------------------------------------------------------------------------------------
# B3: chuẩn hoá theo gen trên từng cohort (dùng dữ liệu đích KHÔNG nhãn)
# --------------------------------------------------------------------------------------------------------------------------
def within_sample_zscore(X: np.ndarray) -> np.ndarray:
    """z-score trong từng mẫu (theo hàng) trên mọi cột của X."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=0, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def per_gene_zscore(X: np.ndarray) -> np.ndarray:
    """z-score từng gen (theo cột) bằng chính các mẫu của X."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


# --------------------------------------------------------------------------------------------------------------------------
# B1/B6: nhãn cấp lần chia
# --------------------------------------------------------------------------------------------------------------------------
def wilcoxon_two_sided(d: np.ndarray) -> float:
    """Wilcoxon dấu hạng hai phía, phân phối chính xác (scipy, method="exact"); mọi Δ = 0 -> p = 1."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan
    if np.all(d == 0):
        return 1.0
    nz = d[d != 0]
    method = "exact" if len(np.unique(np.abs(nz))) == len(nz) else "approx"   # hạng hoà: scipy không cho exact
    return float(st.wilcoxon(d, alternative="two-sided", method=method, zero_method="wilcox").pvalue)


def bootstrap_mean_ci(d: np.ndarray, n_boot: int = SPLIT_BOOTSTRAP_N, seed_parts=(0,), level: float = 0.95) -> tuple[float, float]:
    """CI phân vị của trung bình khi lấy mẫu lại các lần chia (có hoàn lại)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(list(seed_parts))
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def split_label(p_holm: float, mean: float, lo: float, hi: float, margin: float = SPLIT_LABEL_MARGIN) -> str:
    """Quy tắc đọc B1 (12/11). Thứ tự ưu tiên khi nhiều nhãn cùng thoả: positive/negative trước equivalent (ghi choices)."""
    if not np.isnan(p_holm) and p_holm < C.ALPHA and mean > margin:
        return "positive transfer"
    if not np.isnan(p_holm) and p_holm < C.ALPHA and mean < -margin:
        return "negative transfer"
    if not (np.isnan(lo) or np.isnan(hi)) and lo > -margin and hi < margin:
        return "practically equivalent"
    return "inconclusive"


def split_level_labels(per_run: pd.DataFrame, strategy_col: str, reference: str, cell_cols, seed_tag: str,
                       metric: str = "auroc", n_boot: int = SPLIT_BOOTSTRAP_N) -> pd.DataFrame:
    """Giá trị cấp lần chia = trung bình qua tập con (subset); Δ_s = chiến lược − `reference` ở lần chia s; Wilcoxon hai phía
    theo ô `cell_cols`; Holm trong từng chiến lược qua mọi ô của nó; CI bootstrap 95 % theo lần chia (khoá seed `seed_tag`)."""
    cell_cols = list(cell_cols)
    lvl = per_run.groupby(cell_cols + [strategy_col, "split"], as_index=False)[metric].mean()
    ref = lvl[lvl[strategy_col] == reference].drop(columns=strategy_col).rename(columns={metric: "ref"})
    oth = lvl[lvl[strategy_col] != reference]
    m = oth.merge(ref, on=cell_cols + ["split"], how="inner")
    m["delta"] = m[metric] - m["ref"]
    tag = [ord(c) for c in seed_tag]
    rows = []
    for key, g in m.groupby([strategy_col] + cell_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        d = g.sort_values("split")["delta"].to_numpy()
        cell_id = [sum((i + 1) * ord(ch) for i, ch in enumerate(str(v))) for v in key]   # tất định, không dùng hash()
        lo, hi = bootstrap_mean_ci(d, n_boot=n_boot, seed_parts=(*tag, *cell_id))
        rows.append({strategy_col: key[0], **dict(zip(cell_cols, key[1:])), "reference": reference, "n_splits": int(len(d)),
                     "mean_delta": float(np.mean(d)), "median_delta": float(np.median(d)), "ci_low": lo, "ci_high": hi,
                     "p_wilcoxon": wilcoxon_two_sided(d), "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for s, idx in out.groupby(strategy_col).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_wilcoxon"].to_numpy())
    out["label"] = [split_label(p, mu, lo, hi) for p, mu, lo, hi in out[["p_holm", "mean_delta", "ci_low", "ci_high"]].itertuples(index=False)]
    return out


# --------------------------------------------------------------------------------------------------------------------------
# Kết quả đã công bố
# --------------------------------------------------------------------------------------------------------------------------
def published(*parts, root: Path | None = None) -> Path:
    return Path(root or PUBLISHED_ROOT).joinpath(*parts)


def read_published_csv(*parts, root: Path | None = None, **kw) -> pd.DataFrame:
    path = published(*parts, root=root)
    if not path.exists():
        raise FileNotFoundError(f"thiếu kết quả đã công bố: {path}")
    return pd.read_csv(path, **kw)


def compare_rows(new: pd.DataFrame, old: pd.DataFrame, key, cols=("auroc",), tol: float = GATE_TOL_AUROC) -> dict:
    """Cổng bằng nhau: mọi khoá của `old` phải có trong `new`, sai khác tuyệt đối của từng cột ≤ tol (NaN khớp NaN)."""
    key = list(key)
    m = old[key + list(cols)].merge(new[key + list(cols)], on=key, how="left", suffixes=("_old", "_new"), indicator=True)
    missing = int((m["_merge"] != "both").sum())
    report = {"n_reference_rows": int(len(old)), "n_missing": missing, "tol": tol}
    ok = missing == 0
    for c in cols:
        a, b = m[f"{c}_old"].to_numpy(dtype=float), m[f"{c}_new"].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0.0, np.abs(a - b))
        mx = float(np.nanmax(diff)) if len(diff) else 0.0
        report[f"max_abs_diff_{c}"] = mx
        ok = ok and bool(np.all(np.nan_to_num(diff, nan=np.inf) <= tol))
    report["pass"] = bool(ok)
    return report
