"""Bản chia few-shot dùng chung cho E1b, E7 (survey) và RQ2, RQ4 (bài phương pháp) — giao thức KBTN v2 Mục 6.

Mỗi cohort GEO: FEWSHOT_SPLITS lần chia test_T / pool_T (≈ 1/3 : 2/3, phân tầng, seed 42–46); trong mỗi lần chia,
FEWSHOT_SUBSETS tập con; tập con s ở mức n_T là n_T mẫu MỖI LỚP rút từ pool_T, **lồng nhau theo n_T** trong cùng
(split, subset) (n_T = 5 ⊂ 10 ⊂ 20 ⊂ 40 ⊂ pool) để đường cong học ghép cặp theo n_T. Mọi phương pháp dùng chung
các tập con. Mức n_T vượt số mẫu của lớp nhỏ trong pool_T bị bỏ (GSE10245 chỉ n_T ∈ {5, 10}, FEWSHOT_N_T_LIMIT).

Tệp ``fewshot_splits.csv`` (dạng dài): cohort, split, seed, subset, n_t (số hoặc "pool"), sample_id, patient_id,
label, role ∈ {test, train}. Dòng test lặp lại cho mỗi (subset, n_t) để mỗi lần chạy tự chứa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from . import config as C

FEWSHOT_COLUMNS = ("cohort", "split", "seed", "subset", "n_t", "sample_id", "patient_id", "label", "role")


def _n_t_levels(name: str, pool_counts: dict[int, int]) -> list[int | None]:
    allowed = C.FEWSHOT_N_T_LIMIT.get(name)
    levels: list[int | None] = []
    min_class = min(pool_counts.values())
    for n_t in C.N_T_GRID:
        if n_t is None:
            levels.append(None)
        elif allowed is not None and n_t not in allowed:
            continue
        elif n_t <= min_class:
            levels.append(n_t)
    return levels


def build_fewshot_manifest(name: str, sample_ids, patient_ids, labels, *, seeds=C.REPEAT_SEEDS,
                           n_subsets: int = C.FEWSHOT_SUBSETS, test_fraction: float = C.FEWSHOT_TEST_FRACTION) -> pd.DataFrame:
    sample_ids = np.asarray(sample_ids, dtype=str)
    patient_ids = np.asarray(patient_ids, dtype=str)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(patient_ids)) != len(patient_ids):
        raise ValueError(f"{name}: few-shot giả định mỗi bệnh nhân một mẫu; có bệnh nhân nhiều mẫu")
    cid = C.cohort_id(name)
    rows = []
    for split, seed in enumerate(seeds[: C.FEWSHOT_SPLITS]):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=int(seed))
        pool_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
        pool_counts = {c: int(np.sum(y[pool_idx] == c)) for c in (0, 1)}
        levels = _n_t_levels(name, pool_counts)
        for subset in range(n_subsets):
            rng = np.random.default_rng([cid, int(seed), subset, 7])
            # hoán vị cố định của pool theo lớp -> tập con lồng nhau theo n_T
            perm = {c: rng.permutation(pool_idx[y[pool_idx] == c]) for c in (0, 1)}
            for n_t in levels:
                if n_t is None:
                    train_idx = np.sort(pool_idx)
                    tag = "pool"
                else:
                    train_idx = np.sort(np.concatenate([perm[0][:n_t], perm[1][:n_t]]))
                    tag = str(n_t)
                for role, idx in (("train", train_idx), ("test", np.sort(test_idx))):
                    rows.append(pd.DataFrame({"cohort": name, "split": split, "seed": int(seed), "subset": subset, "n_t": tag,
                                              "sample_id": sample_ids[idx], "patient_id": patient_ids[idx],
                                              "label": y[idx], "role": role}))
    man = pd.concat(rows, ignore_index=True)
    errors = validate_fewshot_manifest(man)
    if errors:
        raise ValueError("bản chia few-shot không hợp lệ: " + "; ".join(errors))
    return man.loc[:, FEWSHOT_COLUMNS]


def validate_fewshot_manifest(man: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = set(FEWSHOT_COLUMNS) - set(man.columns)
    if missing:
        return [f"thiếu cột {sorted(missing)}"]
    for (split, subset, n_t), g in man.groupby(["split", "subset", "n_t"], sort=False):
        tr, te = g[g.role == "train"], g[g.role == "test"]
        if tr.empty or te.empty:
            errors.append(f"split={split} subset={subset} n_t={n_t}: thiếu train hoặc test")
            continue
        if set(tr.patient_id) & set(te.patient_id):
            errors.append(f"split={split} subset={subset} n_t={n_t}: trùng bệnh nhân train/test")
        if set(tr.label) != {0, 1} or set(te.label) != {0, 1}:
            errors.append(f"split={split} subset={subset} n_t={n_t}: thiếu một lớp")
        if n_t != "pool":
            n = int(n_t)
            if not all(int((tr.label == c).sum()) == n for c in (0, 1)):
                errors.append(f"split={split} subset={subset} n_t={n_t}: không đúng n_t mẫu mỗi lớp")
    # test_T giống nhau trong một split; tập con lồng nhau theo n_T
    for split, g in man.groupby("split", sort=False):
        tests = {tuple(sorted(x.sample_id)) for _, x in g[g.role == "test"].groupby(["subset", "n_t"])}
        if len(tests) != 1:
            errors.append(f"split={split}: test_T không cố định")
        for subset, gs in g[g.role == "train"].groupby("subset"):
            levels = sorted({int(v) for v in gs.n_t.unique() if v != "pool"})
            prev = None
            for n_t in levels:
                cur = set(gs[gs.n_t == str(n_t)].sample_id)
                if prev is not None and not prev <= cur:
                    errors.append(f"split={split} subset={subset}: tập con không lồng nhau ở n_t={n_t}")
                prev = cur
    return errors


def fewshot_runs(man: pd.DataFrame):
    """Sinh (split, seed, subset, n_t_tag, train_ids, test_ids) theo thứ tự."""
    for (split, seed, subset, n_t), g in man.groupby(["split", "seed", "subset", "n_t"], sort=False):
        yield (int(split), int(seed), int(subset), str(n_t),
               g.loc[g.role == "train", "sample_id"].to_numpy(dtype=str), g.loc[g.role == "test", "sample_id"].to_numpy(dtype=str))


def n_t_sort_key(tag: str) -> int:
    return 10**9 if tag == "pool" else int(tag)
