"""E7 — ba chiến lược chuyển giao ĐƠN GIẢN nguồn (TCGA, RNA-seq) → đích (GEO microarray), KBTN v2 Mục 6 (baseline D3).

Không có phương pháp đề xuất ở đây (MVP thuộc bài 2). Ba chiến lược + target-only để ghép cặp:
* source_only  : chọn k gen trên nguồn (TRANSFER_SELECTOR), huấn luyện LR-L2 trên nguồn với chuẩn hoá mức panel φ,
                 áp thẳng lên test_T (zero-shot, không dùng nhãn đích; không phụ thuộc n_T).
* select_on_source : panel chọn trên nguồn, bộ phân loại fit trên L_T (n_T mẫu/lớp) ở thang đo đích.
* pooled       : nguồn (z-score theo nguồn) ⊕ L_T (z-score theo L_T) → chọn k gen trên ma trận gộp → fit LR-L2 trên L_T.
* target_only  : chọn k gen và fit trên L_T (trùng E1b, tính lại để ghép cặp đúng tập con).

Chuẩn hoá mức panel φ (định nghĩa bài toán, Review 11/10 Mục 8.2.1): biến đổi trong từng mẫu chỉ dùng giá trị của chính
mẫu đó trên panel S: z-score trong mẫu trên k gen (k ≥ 3); k = 2: hiệu hai gen; k = 1: không có φ (giữ giá trị gốc —
chỉ dùng được khi fit lại trên đích).
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .classify import EvalResult, evaluate_panel
from .data import Dataset, align_to_genes, common_genes, variance_prefilter
from .methods import select, zscore_apply, zscore_fit


def panel_transform(X: np.ndarray, mode: str = C.PANEL_NORMALISATION) -> np.ndarray:
    """φ trên ma trận mẫu × k gen của panel."""
    X = np.asarray(X, dtype=float)
    k = X.shape[1]
    if mode != "within_sample_zscore":
        raise ValueError(mode)
    if k == 1:
        return X.copy()
    if k == 2:
        return (X[:, 0] - X[:, 1])[:, None]
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


class TransferContext:
    """Nguồn và đích đã căn về không gian gen chung G; lọc sơ bộ trên nguồn; panel nguồn theo k (tính một lần)."""

    def __init__(self, source: Dataset, target: Dataset, k_list, selector: str = C.TRANSFER_SELECTOR,
                 prefilter: int = C.PREFILTER_P0, seed: int = 0):
        self.genes = common_genes(target, source)                 # thứ tự theo đích
        if len(self.genes) < max(k_list):
            raise ValueError(f"không gian gen chung quá nhỏ: {len(self.genes)}")
        self.Xs = align_to_genes(source, self.genes)
        self.ys = source.y
        self.Xt = align_to_genes(target, self.genes)
        self.yt = target.y
        self.pre = variance_prefilter(self.Xs, prefilter)         # lọc sơ bộ KHÔNG nhãn trên nguồn
        self.selector = selector
        self.k_list = sorted(int(k) for k in k_list)
        self.seed = seed
        sel = select(selector, self.Xs[:, self.pre], self.ys, self.k_list, seed=seed)
        self.source_panels = {k: self.pre[np.asarray(sel.panels[k])] for k in self.k_list}   # chỉ số trong G
        self.source_info = sel.info
        self._source_only_models: dict[int, tuple] = {}

    # --- source_only ----------------------------------------------------------------------------------
    def source_only(self, k: int, test_idx: np.ndarray, seed: int | None = None) -> EvalResult:
        """Zero-shot: mô hình nguồn là MỘT mô hình cố định (seed của context), không phụ thuộc split/tập con đích."""
        S = self.source_panels[k]
        Xs_phi = panel_transform(self.Xs[:, S])
        Xt_phi = panel_transform(self.Xt[test_idx][:, S])
        return evaluate_panel(Xs_phi, self.ys, Xt_phi, self.yt[test_idx], classifier="lr_l2", seed=self.seed, small_n=False)

    # --- select_on_source -----------------------------------------------------------------------------
    def select_on_source(self, k: int, train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> EvalResult:
        S = self.source_panels[k]
        return evaluate_panel(self.Xt[train_idx][:, S], self.yt[train_idx], self.Xt[test_idx][:, S], self.yt[test_idx],
                              classifier="lr_l2", seed=seed)

    # --- pooled ----------------------------------------------------------------------------------------
    def pooled(self, k: int, train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> tuple[EvalResult, np.ndarray]:
        Xs_p = self.Xs[:, self.pre]
        Xt_p = self.Xt[train_idx][:, self.pre]
        mu_s, sd_s = zscore_fit(Xs_p)
        mu_t, sd_t = zscore_fit(Xt_p)
        Z = np.vstack([zscore_apply(Xs_p, mu_s, sd_s), zscore_apply(Xt_p, mu_t, sd_t)])
        yz = np.concatenate([self.ys, self.yt[train_idx]])
        sel = select(self.selector, Z, yz, [k], seed=seed)          # select() z-score lại (đã chuẩn) — vô hại
        S = self.pre[np.asarray(sel.panels[k])]
        res = evaluate_panel(self.Xt[train_idx][:, S], self.yt[train_idx], self.Xt[test_idx][:, S], self.yt[test_idx],
                             classifier="lr_l2", seed=seed)
        return res, S

    # --- target_only -----------------------------------------------------------------------------------
    def target_only(self, k: int, train_idx: np.ndarray, test_idx: np.ndarray, seed: int,
                    prefilter: int = C.PREFILTER_P0) -> tuple[EvalResult, np.ndarray, dict]:
        Xt_tr = self.Xt[train_idx]
        pre_t = variance_prefilter(Xt_tr, prefilter)
        sel = select(self.selector, Xt_tr[:, pre_t], self.yt[train_idx], [k], seed=seed)
        S = pre_t[np.asarray(sel.panels[k])]
        res = evaluate_panel(Xt_tr[:, S], self.yt[train_idx], self.Xt[test_idx][:, S], self.yt[test_idx],
                             classifier="lr_l2", seed=seed)
        return res, S, sel.info
