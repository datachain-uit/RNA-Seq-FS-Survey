"""Đọc dữ liệu, chia CV theo bệnh nhân, lọc sơ bộ không dùng nhãn, lấy mẫu con, bản sao tương quan.

Định dạng bộ dữ liệu (experiment-data-perm/<bộ>/): ``expression.h5`` (key ``expression``, mẫu × gen,
index = sample_id), ``metadata.csv`` (index sample_id; cột patient_id, class, label ∈ {0, 1}).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config as C


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Dataset:
    name: str
    X: np.ndarray          # mẫu × gen, float64, log2 gốc (KHÔNG chuẩn hóa)
    y: np.ndarray          # 0 = sớm (I), 1 = muộn (II–IV)
    groups: np.ndarray     # patient_id
    genes: np.ndarray      # tên gen (str)
    sample_ids: np.ndarray
    files_sha256: dict
    data_dir: Path

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def p(self) -> int:
        return self.X.shape[1]


def load_dataset(root: Path, name: str) -> Dataset:
    """Đọc một bộ dữ liệu bài toán phụ; kiểm tra khớp index, nhãn nhị phân, không NaN."""
    d = Path(root) / C.PERM_DATA_DIR / name
    x_df = pd.read_hdf(d / "expression.h5", key="expression")
    meta = pd.read_csv(d / "metadata.csv", index_col=0)
    x_df.index = x_df.index.astype(str)
    meta.index = meta.index.astype(str)
    if list(x_df.index) != list(meta.index):
        raise ValueError(f"{name}: expression.h5 và metadata.csv không cùng thứ tự sample_id")
    y = meta["label"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{name}: nhãn phải là 0/1, gặp {np.unique(y)}")
    X = np.ascontiguousarray(x_df.to_numpy(dtype=np.float64))
    if np.isnan(X).any():
        raise ValueError(f"{name}: có NaN trong biểu hiện")
    groups = meta["patient_id"].astype(str).to_numpy()
    return Dataset(name=name, X=X, y=y, groups=groups, genes=x_df.columns.to_numpy().astype(str),
                   sample_ids=meta.index.to_numpy(), data_dir=d,
                   files_sha256={f: sha256(d / f) for f in ("expression.h5", "metadata.csv")})


# ---------------------------------------------------------------------------------------------------
# Chia CV lặp theo bệnh nhân
# ---------------------------------------------------------------------------------------------------
def repeated_patient_splits(y: np.ndarray, groups: np.ndarray, seeds=C.REPEAT_SEEDS, n_folds: int = C.N_FOLDS):
    """Sinh các lần chia (repeat, fold, train_idx, test_idx) phân tầng theo nhãn, nhóm theo bệnh nhân.

    Nếu mỗi bệnh nhân chỉ có một mẫu (trường hợp của ba bộ giai đoạn) thì StratifiedKFold(shuffle)
    là đủ và trùng với cách chia của RQ0′; nếu có bệnh nhân nhiều mẫu thì dùng StratifiedGroupKFold.
    Luôn assert không có bệnh nhân nằm ở cả train và test.
    """
    one_sample_per_patient = len(np.unique(groups)) == len(groups)
    out = []
    for repeat, seed in enumerate(seeds):
        if one_sample_per_patient:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splitter = cv.split(np.zeros(len(y)), y)
        else:
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splitter = cv.split(np.zeros(len(y)), y, groups)
        for fold, (tr, te) in enumerate(splitter):
            assert set(groups[tr]).isdisjoint(groups[te]), f"repeat {repeat} fold {fold}: trùng bệnh nhân"
            assert len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2
            out.append((repeat, seed, fold, tr, te))
    return out


def splits_to_frame(ds: Dataset, splits) -> pd.DataFrame:
    rows = []
    for repeat, seed, fold, tr, te in splits:
        role = np.full(ds.n, "", dtype=object)
        role[tr], role[te] = "train", "test"
        rows.append(pd.DataFrame({"cohort": ds.name, "repeat": repeat, "seed": seed, "fold": fold,
                                  "sample_id": ds.sample_ids, "patient_id": ds.groups, "label": ds.y, "role": role}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------------
# Lọc sơ bộ không dùng nhãn (trong fold)
# ---------------------------------------------------------------------------------------------------
def variance_prefilter(X_train: np.ndarray, p0: int = C.PREFILTER_P0) -> np.ndarray:
    """Chỉ số của p0 gen có phương sai lớn nhất TRÊN PHẦN TRAIN (không đọc nhãn, không đọc test).

    Trả về mảng chỉ số đã sắp xếp tăng dần để giữ thứ tự cột ổn định.
    """
    if p0 is None or p0 >= X_train.shape[1]:
        return np.arange(X_train.shape[1])
    var = X_train.var(axis=0, ddof=1)
    keep = np.argpartition(-var, p0 - 1)[:p0]
    return np.sort(keep)


# ---------------------------------------------------------------------------------------------------
# Lấy mẫu con độc lập (E4) và bản sao tương quan (E5a)
# ---------------------------------------------------------------------------------------------------
def stratified_subsample(y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Chỉ số của một tập con phân tầng theo nhãn với tỉ lệ `fraction` số mẫu (mỗi mẫu = một bệnh nhân)."""
    idx = []
    for c in (0, 1):
        cand = np.flatnonzero(y == c)
        m = max(2, int(round(fraction * len(cand))))
        idx.append(rng.choice(cand, size=m, replace=False))
    return np.sort(np.concatenate(idx))


def add_correlated_copies(X: np.ndarray, genes: np.ndarray, rho: float, n_copies: int,
                          rng: np.random.Generator):
    """Thêm `n_copies` bản sao tương quan của các gen phương sai lớn nhất (E5a), KHÔNG dùng nhãn.

    x' = ρ·z + sqrt(1 − ρ²)·ε với z = x đã z-score trên chính ma trận X (phần train / tập con),
    ε ~ N(0, 1). Bản sao được đổi về thang của gen gốc (nhân σ, cộng μ) để các phương pháp không
    chuẩn hóa không phân biệt được bản sao chỉ qua thang đo. Trả về (X_mới, genes_mới, chỉ số gốc,
    chỉ số bản sao).
    """
    var = X.var(axis=0, ddof=1)
    src = np.argsort(-var, kind="stable")[:n_copies]
    mu, sd = X[:, src].mean(0), X[:, src].std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    z = (X[:, src] - mu) / sd
    eps = rng.standard_normal(z.shape)
    copies = (rho * z + np.sqrt(1.0 - rho**2) * eps) * sd + mu
    X_new = np.hstack([X, copies])
    copy_names = np.array([f"{genes[j]}__copy_rho{rho}" for j in src])
    genes_new = np.concatenate([genes, copy_names])
    copy_idx = np.arange(X.shape[1], X.shape[1] + n_copies)
    return X_new, genes_new, src, copy_idx
