"""Run the preregistered RQ0' performance-ceiling checks on GEO cohorts.

All feature selection happens inside each training split.  The result is an
honest repeated stratified 5x5-CV estimate, not a reuse of Phase-1 fold
summaries that were produced by different pipeline variants.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COHORTS = ("gse10245", "gse30219", "gse37745", "gse42127")
REPEATS = (42, 43, 44, 45, 46)
RANDOM_KS = (1, 2, 3, 5, 10, 25, 50)
IHC_2 = ("NKX2-1", "TP63")
IHC_4 = ("NKX2-1", "TP63", "NAPSA", "KRT5")


def sample_balanced_training_indices(labels: list[int] | np.ndarray, n_per_class: int, seed: int) -> np.ndarray:
    """Select `n_per_class` observations for each binary class without replacement."""
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for value in (0, 1):
        candidates = np.flatnonzero(y == value)
        if len(candidates) < n_per_class:
            raise ValueError(f"Class {value} only has {len(candidates)} samples; need {n_per_class}.")
        selected.extend(rng.choice(candidates, size=n_per_class, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def fitted_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000, random_state=0),
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def evaluate_method(
    method: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
    columns: pd.Index, random_seed: int,
) -> tuple[np.ndarray, str]:
    if method == "B0_all_genes":
        return fitted_scores(x_train, y_train, x_test), "all_genes"
    if method == "best_single_gene":
        scores, _ = f_classif(x_train, y_train)
        selected = int(np.nanargmax(np.nan_to_num(scores, nan=-np.inf)))
        return fitted_scores(x_train[:, [selected]], y_train, x_test[:, [selected]]), str(columns[selected])
    if method.startswith("IHC_"):
        genes = IHC_2 if method == "IHC_2" else IHC_4
        available = [gene for gene in genes if gene in columns]
        if len(available) != len(genes):
            return np.array([]), "SKIP: missing " + ", ".join(gene for gene in genes if gene not in columns)
        positions = [int(columns.get_loc(gene)) for gene in available]
        return fitted_scores(x_train[:, positions], y_train, x_test[:, positions]), ",".join(available)
    if method.startswith("random_k_"):
        k = int(method.rsplit("_", 1)[1])
        rng = np.random.default_rng(random_seed)
        positions = np.sort(rng.choice(x_train.shape[1], size=k, replace=False))
        return fitted_scores(x_train[:, positions], y_train, x_test[:, positions]), ",".join(columns[positions].tolist())
    raise ValueError(f"Unknown method: {method}")


def run_cohort(root: Path, cohort: str) -> list[dict[str, object]]:
    data_dir = root / "experiment-data-geo-target" / cohort
    x_df = pd.read_hdf(data_dir / "expression.h5")
    metadata = pd.read_csv(data_dir / "metadata.csv")
    if list(x_df.index.astype(str)) != metadata["sample_id"].astype(str).tolist():
        raise ValueError(f"{cohort}: expression rows do not align with metadata sample_id.")
    x = x_df.to_numpy(dtype=np.float64, copy=True)
    y = metadata["label"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{cohort}: expected binary labels 0/1.")
    methods = ["B0_all_genes", "best_single_gene", "IHC_2", "IHC_4", *[f"random_k_{k}" for k in RANDOM_KS]]
    rows: list[dict[str, object]] = []
    for repeat, split_seed in enumerate(REPEATS, start=1):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
            for n_train_per_class in ("all", 5):
                active_train = train_idx
                if n_train_per_class == 5:
                    local = sample_balanced_training_indices(y[train_idx], 5, seed=split_seed * 100 + fold)
                    active_train = train_idx[local]
                for method in methods:
                    score, selected = evaluate_method(
                        method, x[active_train], y[active_train], x[test_idx], x_df.columns,
                        random_seed=split_seed * 10000 + fold * 100 + len(rows),
                    )
                    if len(score) == 0:
                        rows.append({"cohort": cohort, "repeat": repeat, "seed": split_seed, "fold": fold,
                                     "train_per_class": n_train_per_class, "method": method,
                                     "selected_features": selected, "auroc": np.nan, "balanced_accuracy": np.nan,
                                     "status": "SKIP"})
                        continue
                    prediction = (score >= 0.5).astype(int)
                    rows.append({"cohort": cohort, "repeat": repeat, "seed": split_seed, "fold": fold,
                                 "train_per_class": n_train_per_class, "method": method,
                                 "selected_features": selected, "auroc": roc_auc_score(y[test_idx], score),
                                 "balanced_accuracy": balanced_accuracy_score(y[test_idx], prediction), "status": "OK"})
    return rows


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    usable = metrics.query("status == 'OK'").copy()
    grouped = usable.groupby(["cohort", "train_per_class", "method"], as_index=False).agg(
        n_folds=("auroc", "count"), auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        bacc_mean=("balanced_accuracy", "mean"), bacc_sd=("balanced_accuracy", "std"),
    )
    grouped["ceiling_rule"] = "Không áp dụng"
    subset = (grouped["train_per_class"].eq(5) & grouped["method"].isin(["B0_all_genes", "best_single_gene", "IHC_2", "IHC_4", "random_k_1", "random_k_2"]))
    grouped.loc[subset, "ceiling_rule"] = np.where(
        grouped.loc[subset, "auroc_mean"] >= 0.98,
        "Đạt điều kiện AUROC trung bình ≥ 0.98 (đánh giá tiếp theo ở cấp cohort)",
        "Chưa đạt điều kiện AUROC trung bình ≥ 0.98",
    )
    return grouped.sort_values(["cohort", "train_per_class", "method"]).reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "phase 1" / "rq0_prime"
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        print(f"Running RQ0' for {cohort} ...", flush=True)
        all_rows.extend(run_cohort(root, cohort))
    metrics = pd.DataFrame(all_rows)
    summary = summarize(metrics)
    metrics.to_csv(out / "metrics_per_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    config = {"protocol": "Repeated stratified 5x5 CV on each GEO cohort; all selection inside training folds",
              "cohorts": list(COHORTS), "repeat_seeds": list(REPEATS), "random_k": list(RANDOM_KS),
              "methods": ["B0_all_genes", "best_single_gene", "IHC_2: NKX2-1, TP63", "IHC_4: NKX2-1, TP63, NAPSA, KRT5"],
              "n_train_per_class": ["all", 5], "model": "StandardScaler + LogisticRegression(liblinear, balanced, max_iter=5000)",
              "commit_hash": "unavailable: workspace is not a Git repository"}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# RQ0′ — bảng trần hiệu năng\n\n"
        "`metrics_per_fold.csv` chứa 25 đánh giá độc lập (5 lần lặp × 5 fold) cho từng cohort/method/cỡ train. "
        "`summary.csv` tổng hợp trung bình và độ lệch chuẩn. Quy tắc trần chỉ là cờ sàng lọc khi `n_train_per_class=5`; "
        "không diễn giải AUROC trung bình là bằng chứng mọi fold đều ≥ 0.98.\n", encoding="utf-8")
    print(f"Wrote {len(metrics)} fold-level rows to {out}", flush=True)


if __name__ == "__main__":
    main()
