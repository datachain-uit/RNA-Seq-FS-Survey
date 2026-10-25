"""Mười phương pháp chọn gen + hai tham chiếu của survey v2 (Bảng 10, Review 24/10/2026).

Quy ước chung
-------------
* Mọi hàm chỉ nhận PHẦN TRAIN của fold (đã lọc sơ bộ p0 gen). Không hàm nào nhận phần test.
* Phương pháp xếp hạng (RANKING_METHODS) trả về một thứ tự đầy đủ/đủ dài; tập k gen = k gen đầu
  (tiền tố), nên các tập lồng nhau theo k (assert trong `prefix_panels`).
* Phương pháp theo k (l1_lr, elastic_net) trả về đúng k gen cho từng k; khi L1 không đạt đủ k gen
  (bão hòa ≤ n_train), bổ sung theo thứ hạng F-test trên train và ghi `n_filled` (KBTN 5.1).
* Chuẩn hóa z-score (fit trên train) được áp dụng TRƯỚC l1_lr, elastic_net, svm_rfe (M7).
* Chỉ số trả về là chỉ số cột của ma trận đưa vào (ma trận đã lọc sơ bộ); người gọi ánh xạ về gen.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from . import config as C
from .mrmr import mrmr_ranking


# ---------------------------------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------------------------------
def zscore_fit(X: np.ndarray):
    mu = X.mean(0)
    sd = X.std(0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def ftest_order(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    F, _ = f_classif(X, y)
    return np.argsort(-np.nan_to_num(F, nan=-np.inf), kind="stable")


@dataclass
class Selection:
    """Kết quả chọn gen của một phương pháp trên một phần train."""
    method: str
    panels: dict                       # k -> np.ndarray chỉ số (kích thước đúng k)
    ranking: np.ndarray | None = None  # thứ tự đầy đủ (phương pháp xếp hạng)
    info: dict = field(default_factory=dict)   # C đã dùng, n_filled, n_eval, ...


def prefix_panels(order: np.ndarray, k_list) -> dict:
    """Tập k gen = k gen đầu của xếp hạng; kiểm tra lồng nhau."""
    ks = sorted(int(k) for k in k_list)
    if max(ks) > len(order):
        raise ValueError(f"xếp hạng chỉ có {len(order)} gen, cần k_max = {max(ks)}")
    panels = {k: np.asarray(order[:k], dtype=int) for k in ks}
    for a, b in zip(ks, ks[1:]):
        assert set(panels[a]) <= set(panels[b])
    return panels


# ---------------------------------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------------------------------
def rank_f_test(X, y, **_) -> np.ndarray:
    """ANOVA F-test (tương đương t-test hai lớp): O(n p)."""
    return ftest_order(X, y)


def rank_mutual_info(X, y, seed: int = 0, n_neighbors: int = 3, **_) -> np.ndarray:
    """Thông tin tương hỗ (ước lượng KNN của scikit-learn): O(n log n · p)."""
    mi = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=seed)
    return np.argsort(-mi, kind="stable")


def rank_mrmr(X, y, k_max: int, **_) -> np.ndarray:
    """mRMR (Peng et al., 2005), MID/MIQ trên dữ liệu rời rạc hóa; chỉ xếp k_max gen đầu (tham lam)."""
    return mrmr_ranking(X, y, k_max, criterion=C.MRMR_CRITERION, discretize_method=C.MRMR_DISCRETIZE)


def rank_relieff(X, y, n_neighbors: int = C.RELIEFF_NEIGHBORS, n_jobs: int = 1, **_) -> np.ndarray:
    """ReliefF (Kononenko 1994; cài đặt skrebate, Urbanowicz et al. 2018): O(n² p)."""
    from skrebate import ReliefF  # import muộn để module vẫn dùng được khi thiếu skrebate
    est = ReliefF(n_features_to_select=X.shape[1], n_neighbors=n_neighbors, n_jobs=n_jobs)
    est.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int))
    return np.argsort(-est.feature_importances_, kind="stable")


# ---------------------------------------------------------------------------------------------------
# Embedded
# ---------------------------------------------------------------------------------------------------
def rank_rf_importance(X, y, seed: int = 0, n_trees: int = C.RF_N_TREES, n_jobs: int = 1, **_) -> np.ndarray:
    """Độ quan trọng Gini của Random Forest (Breiman 2001): O(B n p log n)."""
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=n_jobs,
                                class_weight="balanced_subsample")
    rf.fit(X, y)
    return np.argsort(-rf.feature_importances_, kind="stable")


def rank_svm_rfe(X, y, k_min: int, step: float = 0.1, C_svm: float = 1.0, **_) -> np.ndarray:
    """SVM-RFE (Guyon et al., 2002) với xếp hạng CHẶT: loại dần một tỉ lệ `step` gen có w² nhỏ nhất
    cho tới khi còn k_min gen; thứ tự loại (kèm w² trong mỗi lô) là xếp hạng ngược.

    X phải đã chuẩn hóa z-score trên train (M7). Trả về thứ tự đầy đủ, gen quan trọng nhất trước.
    """
    p = X.shape[1]
    active = np.arange(p)
    removed_order: list[int] = []                      # ít quan trọng nhất trước
    while True:
        svm = LinearSVC(C=C_svm, class_weight="balanced", dual="auto", max_iter=10000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svm.fit(X[:, active], y)
        w2 = np.ravel(svm.coef_) ** 2
        order = np.argsort(w2, kind="stable")          # tăng dần: ít quan trọng trước
        if len(active) <= k_min:
            final = active[np.argsort(-w2, kind="stable")]  # quan trọng nhất trước
            break
        n_remove = max(1, int(step * len(active)))
        n_remove = min(n_remove, len(active) - k_min)
        drop = order[:n_remove]
        removed_order.extend(active[drop].tolist())
        active = np.sort(active[order[n_remove:]])
    ranking = np.concatenate([final, np.asarray(removed_order[::-1], dtype=int)])
    assert len(ranking) == p and len(set(ranking)) == p
    return ranking


def _coef_path(model_factory, X, y, c_grid):
    """Fit dọc lưới C (tăng dần); trả về danh sách (C, hệ số)."""
    out = []
    for c in c_grid:
        m = model_factory(c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        out.append((float(c), np.ravel(m.coef_).copy()))
    return out


def _exact_k_from_path(model_factory, X, y, k, path, refine_steps: int = 6):
    """Tìm C cho đúng k hệ số khác 0 (tinh chỉnh nhị phân giữa hai điểm lưới); nếu không hit đúng,
    lấy C nhỏ nhất có ≥ k hệ số khác 0 rồi giữ k gen |β| lớn nhất (M6)."""
    nnz = [int((coef != 0).sum()) for _, coef in path]
    exact = [i for i, v in enumerate(nnz) if v == k]
    if exact:
        c, coef = path[exact[0]]
        return np.flatnonzero(coef != 0), c, True
    above = [i for i, v in enumerate(nnz) if v > k]
    if not above:
        return None, None, False                          # bão hòa: không C nào đạt k
    i_hi = above[0]                                       # C nhỏ nhất có nnz > k
    c_hi, coef_hi = path[i_hi]
    c_lo = path[i_hi - 1][0] if i_hi > 0 else c_hi / 10.0
    best_c, best_coef = c_hi, coef_hi
    for _ in range(refine_steps):
        c_mid = float(np.sqrt(c_lo * c_hi))
        m = model_factory(c_mid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        coef = np.ravel(m.coef_)
        v = int((coef != 0).sum())
        if v == k:
            return np.flatnonzero(coef != 0), c_mid, True
        if v > k:
            c_hi, best_c, best_coef = c_mid, c_mid, coef
        else:
            c_lo = c_mid
    top = np.argsort(-np.abs(best_coef), kind="stable")[:k]
    return np.sort(top), best_c, False


def _fill_with_ftest(sel: np.ndarray, k: int, X, y) -> tuple[np.ndarray, int]:
    if len(sel) >= k:
        return sel[:k], 0
    order = ftest_order(X, y)
    extra = [j for j in order if j not in set(sel.tolist())][: k - len(sel)]
    return np.concatenate([sel, np.asarray(extra, dtype=int)]), len(extra)


def _l1_factory(seed):
    return lambda c: LogisticRegression(penalty="l1", solver="liblinear", C=c, class_weight="balanced",
                                        max_iter=5000, random_state=seed)


def _enet_factory(seed, l1_ratio=0.5):
    return lambda c: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, C=c,
                                        class_weight="balanced", max_iter=3000, tol=1e-3, random_state=seed)


def select_sparse_exact_k(X, y, k_list, kind: str, seed: int = 0,
                          c_grid=None) -> Selection:
    """L1-LR hoặc Elastic Net (logistic) với đúng k gen cho từng k (X đã z-score trên train).

    Đường hệ số được fit MỘT lần trên lưới C (log), dùng chung cho mọi k; sau đó tinh chỉnh nhị phân
    quanh điểm lưới để hit đúng k. Khi L1 bão hòa (nnz tối đa < k, thường vì k > n_train),
    bổ sung theo F-test và ghi n_filled (KBTN 5.1).
    """
    if c_grid is None:
        c_grid = np.logspace(-3, 2, 26)
    factory = _l1_factory(seed) if kind == "l1_lr" else _enet_factory(seed)
    path = _coef_path(factory, X, y, c_grid)
    panels, info = {}, {"C": {}, "exact": {}, "n_filled": {}}
    for k in sorted(int(k) for k in k_list):
        sel, c, exact = _exact_k_from_path(factory, X, y, k, path)
        if sel is None:
            # bão hòa: lấy nghiệm có nhiều hệ số khác 0 nhất rồi bổ sung
            c, coef = max(path, key=lambda t: int((t[1] != 0).sum()))
            sel = np.flatnonzero(coef != 0)
            sel = sel[np.argsort(-np.abs(coef[sel]), kind="stable")]
            exact = False
        sel, n_filled = _fill_with_ftest(np.asarray(sel, dtype=int), k, X, y)
        panels[k] = np.sort(sel[:k])
        info["C"][k], info["exact"][k], info["n_filled"][k] = c, bool(exact), n_filled
        assert len(panels[k]) == k and len(set(panels[k].tolist())) == k
    return Selection(method=kind, panels=panels, info=info)


# ---------------------------------------------------------------------------------------------------
# Tham chiếu
# ---------------------------------------------------------------------------------------------------
def random_panel(p: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(p, size=k, replace=False))


# ---------------------------------------------------------------------------------------------------
# Giao diện chung
# ---------------------------------------------------------------------------------------------------
def select(method: str, X_tr: np.ndarray, y_tr: np.ndarray, k_list, seed: int = 0, n_jobs: int = 1,
           wrapper_kwargs: dict | None = None) -> Selection:
    """Chọn gen cho một phương pháp trên PHẦN TRAIN (đã lọc sơ bộ). Trả về Selection với panels theo k.

    Chuẩn hóa z-score (fit trên X_tr) được áp dụng bên trong cho các phương pháp ở STANDARDIZE_BEFORE.
    Wrapper (ga, bpso) dùng survey_v2.wrappers; fitness chỉ dùng X_tr/y_tr (CV nội bộ).
    """
    ks = sorted(int(k) for k in k_list)
    k_max, k_min = max(ks), min(ks)
    X_use = X_tr
    if method in C.STANDARDIZE_BEFORE:
        mu, sd = zscore_fit(X_tr)
        X_use = zscore_apply(X_tr, mu, sd)

    if method == "f_test":
        order = rank_f_test(X_use, y_tr)
    elif method == "mutual_info":
        order = rank_mutual_info(X_use, y_tr, seed=seed)
    elif method == "mrmr":
        order = rank_mrmr(X_use, y_tr, k_max=k_max)
    elif method == "relieff":
        order = rank_relieff(X_use, y_tr, n_jobs=n_jobs)
    elif method == "rf_importance":
        order = rank_rf_importance(X_use, y_tr, seed=seed, n_jobs=n_jobs)
    elif method == "svm_rfe":
        order = rank_svm_rfe(X_use, y_tr, k_min=k_min)
    elif method in ("l1_lr", "elastic_net"):
        return select_sparse_exact_k(X_use, y_tr, ks, kind=method, seed=seed)
    elif method in ("ga", "bpso"):
        from .wrappers import run_wrapper
        return run_wrapper(method, X_use, y_tr, ks, seed=seed, **(wrapper_kwargs or {}))
    else:
        raise ValueError(f"phương pháp không hỗ trợ: {method}")
    return Selection(method=method, panels=prefix_panels(order, ks), ranking=np.asarray(order, dtype=int))
