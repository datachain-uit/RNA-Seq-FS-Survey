"""Dữ liệu tổng hợp theo đúng định dạng của hai layout (perm/ và geo-target/) để chạy thử và test.

KHÔNG dùng cho kết quả bài. Bài toán giả: lớp 0/1 ("LUAD"/"LUSC"); `n_informative` gen tín hiệu chung giữa nguồn và
đích; cohort GEO giả có (i) tập gen = phần lớn gen nguồn + vài gen riêng nền tảng, (ii) lệch nền tảng (affine theo
gen + nhiễu), (iii) ít mẫu, lớp lệch; metadata GEO có cột sample_id, label, class và KHÔNG có patient_id (để kiểm tra
nhánh patient_id = sample_id của loader). Bộ sanity (normal vs tumor) giữ layout perm/ với perm_folds và folds/ như v2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C


def _signal_matrix(rng, n, p, y, informative, effect, n_redundant_blocks):
    X = rng.normal(8.0, 1.0, size=(n, p))
    signs = np.where(np.arange(len(informative)) % 2 == 0, 1.0, -1.0)
    X[:, informative] += effect * y[:, None] * signs
    others = np.setdiff1d(np.arange(p), informative)
    for b in range(n_redundant_blocks):
        src = informative[b % len(informative)]
        block = rng.choice(others, size=5, replace=False)
        X[:, block] = 0.8 * X[:, [src]] + 0.6 * rng.normal(0, 1, size=(n, 5))
    return X


def make_dataset(root: Path, name: str, n: int = 120, p: int = 600, n_informative: int = 15, effect: float = 1.0,
                 seed: int = 0, prevalence: float = 0.45, two_samples_per_patient: bool = False,
                 class_names=("LUAD", "LUSC"), n_redundant_blocks: int = 3, informative=None, genes=None,
                 with_folds: bool = True) -> Path:
    """Layout experiment-data-perm/<name>/ (TCGA v2, sanity)."""
    rng = np.random.default_rng(seed)
    d = Path(root) / C.PERM_DATA_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    n_pat = n // 2 if two_samples_per_patient else n
    patients = [f"P{seed}-{i:04d}" for i in range(n_pat)]
    if two_samples_per_patient:
        pid = np.repeat(patients, 2)
        y = np.tile([0, 1], n_pat)
    else:
        pid = np.array(patients)
        y = (rng.random(n) < prevalence).astype(int)
        y[:2] = [0, 1]
    n = len(y)
    informative = rng.choice(p, size=n_informative, replace=False) if informative is None else np.asarray(informative)
    X = _signal_matrix(rng, n, p, y, informative, effect, n_redundant_blocks)
    X *= rng.uniform(0.5, 2.0, size=p)
    genes = np.array([f"G{j:05d}" for j in range(p)]) if genes is None else np.asarray(genes)
    sample_ids = np.array([f"{name[:6]}-S{i:04d}" for i in range(n)])
    expr = pd.DataFrame(X.astype(np.float32), index=pd.Index(sample_ids, name="sample_id"), columns=genes)
    expr.to_hdf(d / "expression.h5", key="expression", mode="w", complevel=5, complib="blosc")
    meta = pd.DataFrame({"patient_id": pid, "class": np.array(class_names)[y], "label": y,
                         "sample_type": np.where(y == 0, "Solid Tissue Normal", "Primary Tumor") if two_samples_per_patient
                         else "Primary Tumor"}, index=pd.Index(sample_ids, name="sample_id"))
    meta.to_csv(d / "metadata.csv")
    if with_folds:
        parts = np.zeros(n, dtype=int)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for part, (_, idx) in enumerate(sgkf.split(np.zeros(n), y, pid), 1):
            parts[idx] = part
        pf = meta[["patient_id", "class", "label"]].copy()
        pf["part"] = parts
        for r in range(1, 6):
            test, val = r, (r % 5) + 1
            pf[f"fold_{r}"] = np.where(parts == test, "test", np.where(parts == val, "val", "train"))
            for split in ("train", "val", "test"):
                mask = (pf[f"fold_{r}"] == split).to_numpy()
                fp = d / "folds" / f"fold_{r}" / f"{split}.h5"
                fp.parent.mkdir(parents=True, exist_ok=True)
                expr[mask].assign(label=y[mask]).to_hdf(fp, key="data", mode="w", complevel=5, complib="blosc")
        pf.to_csv(d / "perm_folds.csv")
    json.dump({"script": "survey_v2/synthetic.py (SYNTHETIC — not for publication)", "seed": seed,
               "n_informative": int(n_informative), "effect": effect, "informative_genes": [str(genes[j]) for j in informative],
               "commit_hash": None}, open(d / "perm_config.json", "w"), indent=1)
    return d


def make_geo_target(root: Path, name: str, source_genes: np.ndarray, informative_genes, n: int = 80, effect: float = 1.0,
                    seed: int = 0, prevalence: float = 0.35, shared_fraction: float = 0.85, n_platform_only: int = 40,
                    shift: float = 1.0, class_names=("LUAD", "LUSC")) -> Path:
    """Layout experiment-data-geo-target/<thư mục>/ (RQ0′): expression.h5 một node, metadata.csv có sample_id, label, class."""
    rng = np.random.default_rng(seed)
    d = Path(root) / C.dataset_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    keep = rng.random(len(source_genes)) < shared_fraction
    keep[[j for j, g in enumerate(source_genes) if g in set(informative_genes)]] = True   # giữ gen tín hiệu
    genes = np.concatenate([source_genes[keep], np.array([f"PLT{seed}_{j:03d}" for j in range(n_platform_only)])])
    y = (rng.random(n) < prevalence).astype(int)
    y[:4] = [0, 1, 0, 1]
    p = len(genes)
    gi = np.array([j for j, g in enumerate(genes) if g in set(informative_genes)])
    X = _signal_matrix(rng, n, p, y, gi, effect, n_redundant_blocks=2)
    # lệch nền tảng: affine theo gen + nhiễu nền tảng
    X = X * rng.uniform(0.6, 1.4, size=p) + shift * rng.normal(0, 1, size=p) + rng.normal(0, 0.3, size=(n, p))
    sample_ids = np.array([f"GSM{seed}{i:05d}" for i in range(n)])
    expr = pd.DataFrame(X.astype(np.float32), index=pd.Index(sample_ids, name="sample_id"), columns=genes)
    expr.to_hdf(d / "expression.h5", key="expression", mode="w", complevel=5, complib="blosc")
    pd.DataFrame({"sample_id": sample_ids, "label": y, "class": np.array(class_names)[y],
                  "platform": "SYNTHETIC"}).to_csv(d / "metadata.csv", index=False)
    json.dump({"script": "survey_v2/synthetic.py (SYNTHETIC — not for publication)", "seed": seed, "n": int(n), "p": int(p),
               "commit_hash": None}, open(d / "config.json", "w"), indent=1)
    return d


def make_all(root: Path, n: int = 120, p: int = 600, effect_stage: float = 0.7, n_target: int = 80,
             effect: float | None = None) -> dict:
    """Nguồn TCGA giả (perm/), 4 cohort GEO giả (geo-target/), bộ sanity (perm/, 2 mẫu/bệnh nhân)."""
    effect = effect_stage if effect is None else effect
    out = {}
    rng = np.random.default_rng(2026)
    informative = rng.choice(p, size=15, replace=False)
    genes = np.array([f"G{j:05d}" for j in range(p)])
    genes[informative[:4]] = ["NKX2-1", "TP63", "NAPSA", "KRT5"]   # để panel IHC tồn tại trong dữ liệu giả
    out[C.SOURCE_DATASET] = make_dataset(root, C.SOURCE_DATASET, n=max(n, 160), p=p, effect=effect, seed=10,
                                         informative=informative, genes=genes, with_folds=False, prevalence=0.5)
    inf_genes = genes[informative]
    for i, name in enumerate(C.FEWSHOT_COHORTS):
        n_i = n_target if name != "gse10245" else max(40, n_target // 2)
        out[name] = make_geo_target(root, name, genes, inf_genes, n=n_i, effect=effect, seed=20 + i,
                                    prevalence=0.35 if name != "gse10245" else 0.3)
    out[C.SANITY_DATASET] = make_dataset(root, C.SANITY_DATASET, n=n, p=p, n_informative=40, effect=3.0, seed=99,
                                         two_samples_per_patient=True, class_names=("normal", "tumor"))
    return out
