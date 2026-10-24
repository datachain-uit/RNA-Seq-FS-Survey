"""Run the preregistered RQ0' performance-ceiling checks on GEO cohorts.

All feature selection happens inside each training split.  The result is an
honest repeated stratified 5x5-CV estimate, not a reuse of Phase-1 fold
summaries that were produced by different pipeline variants.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # import khi chạy test (scripts.<x>) hoặc chạy trực tiếp (python scripts/<x>.py)
    from scripts.provenance import DATA_ROOT, require, stamp
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import DATA_ROOT, require, stamp


COHORTS = ("gse10245", "gse30219", "gse37745", "gse42127")
REPEATS = (42, 43, 44, 45, 46)
RANDOM_KS = (1, 2, 3, 5, 10, 25, 50)
IHC_2 = ("NKX2-1", "TP63")
IHC_4 = ("NKX2-1", "TP63", "NAPSA", "KRT5")


def cohort_data_dir(root: Path, cohort: str) -> Path:
    """Return the approved input directory for one RQ0′ cohort."""
    directory = "gse30219_r5_fixed" if cohort == "gse30219" else cohort
    return root / "experiment-data-geo-target" / directory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_provenance(root: Path, cohort: str, data_dir: Path, x_df: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, object]:
    """Capture enough input evidence to distinguish the corrected R5 rerun."""
    counts = metadata["label"].value_counts().sort_index().to_dict()
    result: dict[str, object] = {
        "cohort": cohort,
        "data_directory": str(data_dir.relative_to(root)),
        "n_samples": int(len(metadata)),
        "n_genes": int(x_df.shape[1]),
        "label_counts": {str(label): int(count) for label, count in counts.items()},
        "files_sha256": {
            "expression.h5": sha256(data_dir / "expression.h5"),
            "metadata.csv": sha256(data_dir / "metadata.csv"),
        },
    }
    if "class" in metadata:
        result["class_counts"] = {
            str(label): int(count)
            for label, count in metadata["class"].value_counts().sort_index().items()
        }
    return result


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


def run_cohort(root: Path, cohort: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    data_dir = cohort_data_dir(root, cohort)
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
    return rows, input_provenance(root, cohort, data_dir, x_df, metadata)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    usable = metrics.query("status == 'OK'").copy()
    grouped = usable.groupby(["cohort", "train_per_class", "method"], as_index=False).agg(
        n_folds=("auroc", "count"), auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        bacc_mean=("balanced_accuracy", "mean"), bacc_sd=("balanced_accuracy", "std"),
    )
    return grouped.sort_values(["cohort", "train_per_class", "method"]).reset_index(drop=True)


# Quy tắc trần RQ0′ (Bảng 7, Review 11/10; giảng viên làm rõ 20/10; sửa O14):
# ở n_T = 5 mẫu mỗi lớp, cohort chạm trần nếu MỌI phương pháp chọn gen có k <= 2
# đều đạt AUROC trung bình >= 0,98. B0 (toàn bộ gen), IHC_4 (k = 4) và random-k
# (mức sàn) không thuộc tập xét. Kết luận ở mức cohort.
CEILING_SET = ("best_single_gene", "IHC_2")
CEILING_AUROC = 0.98
CEILING_TRAIN_PER_CLASS = "5"


def ceiling_by_cohort(summary: pd.DataFrame) -> pd.DataFrame:
    """Trả về một dòng cho mỗi cohort với kết luận chạm trần (ceiling_hit)."""
    n_t5 = summary["train_per_class"].astype(str) == CEILING_TRAIN_PER_CLASS
    subset = summary[n_t5 & summary["method"].isin(CEILING_SET)]
    rows = []
    for cohort in sorted(summary["cohort"].unique()):
        group = subset[subset["cohort"] == cohort]
        missing = sorted(set(CEILING_SET) - set(group["method"]))
        if missing:
            raise ValueError(f"{cohort}: thiếu phương pháp {missing} ở n_T = 5")
        auroc = group.set_index("method")["auroc_mean"]
        min_method = str(auroc.idxmin())
        rows.append({
            "cohort": cohort,
            "train_per_class": CEILING_TRAIN_PER_CLASS,
            "methods_considered": "; ".join(CEILING_SET),
            **{f"auroc_{m}": float(auroc[m]) for m in CEILING_SET},
            "min_auroc": float(auroc.min()),
            "min_method": min_method,
            "threshold": CEILING_AUROC,
            "ceiling_hit": bool(auroc.min() >= CEILING_AUROC),
        })
    return pd.DataFrame(rows)


def main() -> None:
    root = DATA_ROOT                             # dữ liệu + kết quả (KLTN_DATA_ROOT)
    prov = require(__file__)                     # dừng ngay nếu không truy vết được
    out = root / "results" / "phase 1" / "rq0_prime"
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    input_datasets: list[dict[str, object]] = []
    for cohort in COHORTS:
        print(f"Running RQ0' for {cohort} ...", flush=True)
        rows, provenance = run_cohort(root, cohort)
        all_rows.extend(rows)
        input_datasets.append(provenance)
    metrics = pd.DataFrame(all_rows)
    summary = summarize(metrics)
    metrics.to_csv(out / "metrics_per_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    ceiling = ceiling_by_cohort(summary)
    ceiling.to_csv(out / "ceiling_by_cohort.csv", index=False, encoding="utf-8-sig")
    print(ceiling.to_string(index=False), flush=True)
    config = {"protocol": "Repeated stratified 5x5 CV on each GEO cohort; all selection inside training folds",
              "cohorts": list(COHORTS), "repeat_seeds": list(REPEATS), "random_k": list(RANDOM_KS),
              "methods": ["B0_all_genes", "best_single_gene", "IHC_2: NKX2-1, TP63", "IHC_4: NKX2-1, TP63, NAPSA, KRT5"],
              "n_train_per_class": ["all", 5],
              "ceiling_rule": {"methods": list(CEILING_SET), "train_per_class": 5,
                               "auroc_mean_threshold": CEILING_AUROC, "level": "cohort",
                               "decision": "ceiling_hit = min AUROC trung bình của các phương pháp trong tập >= ngưỡng"}, "model": "StandardScaler + LogisticRegression(liblinear, balanced, max_iter=5000)",
              "input_datasets": input_datasets,
              "generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    for d in input_datasets:
        prov["data_sha256"].update({f"{d['data_directory']}/{f}": h for f, h in d["files_sha256"].items()})
    stamp(config, prov)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hits = ceiling.loc[ceiling["ceiling_hit"], "cohort"].tolist()
    verdict = ("Có cohort chạm trần: " + ", ".join(hits)) if hits else "Không cohort nào chạm trần; giữ nhiệm vụ chính LUAD vs LUSC."
    table = "\n".join(
        f"| {r.cohort} | {r.auroc_best_single_gene:.3f} | {r.auroc_IHC_2:.3f} | {r.min_auroc:.3f} | {'Có' if r.ceiling_hit else 'Không'} |"
        for r in ceiling.itertuples())
    (out / "README.md").write_text(
        "# RQ0′ — bảng trần hiệu năng\n\n"
        "`metrics_per_fold.csv` chứa 25 đánh giá (5 lần lặp × 5 fold) cho từng cohort/method/cỡ train. "
        "Các đánh giá này không độc lập vì tập train giữa các fold chồng lấn; khi so sánh phương pháp phải dùng "
        "phương sai hiệu chỉnh Nadeau–Bengio. `summary.csv` tổng hợp trung bình và độ lệch chuẩn. "
        "`config.json` lưu đường dẫn, số mẫu/lớp và SHA-256 của dữ liệu đầu vào; riêng GSE30219 phải trỏ đến `gse30219_r5_fixed`.\n\n"
        "## Quy tắc trần (sửa O14, Review 20/10)\n\n"
        "Ở n_T = 5 mẫu mỗi lớp, cohort chạm trần nếu cả `best_single_gene` và `IHC_2` (các phương pháp chọn gen có k ≤ 2) "
        "đều đạt AUROC trung bình ≥ 0,98. B0, IHC_4 và random-k không thuộc tập xét. Kết luận ở mức cohort, lưu trong "
        "`ceiling_by_cohort.csv`.\n\n"
        "| Cohort | Gen đơn tốt nhất | IHC-2 | AUROC nhỏ nhất | Chạm trần |\n|---|---|---|---|---|\n"
        + table + "\n\n**Kết luận:** " + verdict + "\n", encoding="utf-8")
    print(f"Wrote {len(metrics)} fold-level rows to {out}", flush=True)


if __name__ == "__main__":
    main()
