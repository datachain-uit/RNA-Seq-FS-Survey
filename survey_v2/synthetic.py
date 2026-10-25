"""Dữ liệu tổng hợp theo đúng định dạng experiment-data-perm/ để chạy thử và test (KHÔNG dùng cho kết quả bài).

make_dataset(root, name, n, p, ...) tạo: expression.h5 (key "expression", float32), metadata.csv (sample_id,
patient_id, class, label, sample_type), perm_folds.csv (+ fold_1..fold_5 train/val/test) và folds/fold_<r>/*.h5
(key "data") như scripts/build_perm_test_data.py. Tín hiệu: `n_informative` gen có chênh trung bình `effect`
(đơn vị SD) giữa hai lớp; `n_redundant_blocks` khối gen tương quan với gen tín hiệu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C


def make_dataset(root: Path, name: str, n: int = 120, p: int = 600, n_informative: int = 15, effect: float = 1.0,
                 seed: int = 0, prevalence: float = 0.45, two_samples_per_patient: bool = False,
                 class_names=("early", "late"), n_redundant_blocks: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    d = Path(root) / C.PERM_DATA_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    n_pat = n // 2 if two_samples_per_patient else n
    patients = [f"P{seed}-{i:04d}" for i in range(n_pat)]
    if two_samples_per_patient:
        # mỗi bệnh nhân một mẫu lớp 0 và một mẫu lớp 1 (kiểu normal / tumor)
        pid = np.repeat(patients, 2)
        y = np.tile([0, 1], n_pat)
    else:
        pid = np.array(patients)
        y = (rng.random(n) < prevalence).astype(int)
        y[:2] = [0, 1]  # đảm bảo đủ hai lớp
    n = len(y)
    X = rng.normal(8.0, 1.0, size=(n, p))
    informative = rng.choice(p, size=n_informative, replace=False)
    X[:, informative] += effect * y[:, None] * rng.choice([-1, 1], size=n_informative)
    # khối tương quan với gen tín hiệu (dư thừa)
    others = np.setdiff1d(np.arange(p), informative)
    for b in range(n_redundant_blocks):
        src = informative[b % n_informative]
        block = rng.choice(others, size=5, replace=False)
        X[:, block] = 0.8 * X[:, [src]] + 0.6 * rng.normal(0, 1, size=(n, 5))
    # phương sai khác nhau để lọc phương sai có ý nghĩa
    X *= rng.uniform(0.5, 2.0, size=p)
    genes = np.array([f"G{j:05d}" for j in range(p)])
    sample_ids = np.array([f"{name[:6]}-S{i:04d}" for i in range(n)])
    expr = pd.DataFrame(X.astype(np.float32), index=pd.Index(sample_ids, name="sample_id"), columns=genes)
    expr.to_hdf(d / "expression.h5", key="expression", mode="w", complevel=5, complib="blosc")
    meta = pd.DataFrame({"patient_id": pid, "class": np.array(class_names)[y], "label": y,
                         "sample_type": np.where(y == 0, "Solid Tissue Normal", "Primary Tumor") if two_samples_per_patient
                         else "Primary Tumor"}, index=pd.Index(sample_ids, name="sample_id"))
    meta.to_csv(d / "metadata.csv")
    # perm_folds: 5 phần theo bệnh nhân; fold r: test = P_r, val = P_(r+1), train = còn lại
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
    json.dump({"script": "survey_v2/synthetic.py (SYNTHETIC — not for publication)", "seed": seed, "n_informative": int(n_informative),
               "effect": effect, "commit_hash": None}, open(d / "perm_config.json", "w"), indent=1)
    return d


def make_all(root: Path, n: int = 120, p: int = 600, effect_stage: float = 0.6) -> dict:
    """Ba bộ giai đoạn (tín hiệu vừa) + bộ sanity (tín hiệu mạnh, 2 mẫu/bệnh nhân)."""
    out = {}
    for i, name in enumerate(C.DATASETS):
        out[name] = make_dataset(root, name, n=n, p=p, effect=effect_stage, seed=10 + i)
    out[C.SANITY_DATASET] = make_dataset(root, C.SANITY_DATASET, n=n, p=p, n_informative=40, effect=3.0, seed=99,
                                         two_samples_per_patient=True, class_names=("normal", "tumor"))
    return out
