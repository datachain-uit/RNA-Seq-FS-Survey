"""E1 phase B: classify fixed Phase-A gene panels on precomputed 5x5 outer tests.

The feature list is read from ``selected_genes.csv``.  Hyperparameters are
chosen using only a 3-fold CV within the corresponding outer-train partition.
No outer-test label or row is consulted until the final scoring call.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.e1_classification import CLASSIFICATION_MODELS, METRIC_NAMES, tune_and_evaluate  # noqa: E402
from survey_v2.e1_splits import validate_repeated_split_manifest  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from scripts.run_e1_feature_selection import split_indices, split_path  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--models", nargs="+", choices=CLASSIFICATION_MODELS, default=list(CLASSIFICATION_MODELS))
    ap.add_argument("--methods", nargs="+", default=None, help="only panels from these selectors")
    ap.add_argument("--k", nargs="+", type=int, default=None)
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(len(C.REPEAT_SEEDS))))
    ap.add_argument("--selection-root", type=Path, default=None,
                    help="results root containing <cohort>/selected_genes.csv")
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--out", default="E1_classification")
    ap.add_argument("--quick", action="store_true", help="small parameter grids for a smoke run; never use for final table")
    return ap.parse_args(argv)


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    selection_root = args.selection_root or (DATA_ROOT / "results" / "survey_v2" / "E1_feature_selection")
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        panels_path = selection_root / name / "selected_genes.csv"
        panels = pd.read_csv(panels_path)
        manifest_path = split_path(ds, args.splits_root)
        manifest = pd.read_csv(manifest_path)
        errors = validate_repeated_split_manifest(manifest, n_folds=C.N_FOLDS)
        if errors:
            raise ValueError(f"{name}: invalid repeated_splits.csv: {'; '.join(errors)}")
        if args.methods:
            panels = panels[panels.method.isin(args.methods)]
        if args.k:
            panels = panels[panels.k.isin(args.k)]
        panels = panels[panels.repeat.isin(args.repeats)]
        gene_index = {g: i for i, g in enumerate(ds.genes)}
        rows = []
        for key, group in panels.groupby(["repeat", "fold", "method", "k"], sort=True):
            repeat, fold, method, k = key
            run = manifest[(manifest.repeat == repeat) & (manifest.fold == fold)]
            seed = int(run.seed.iloc[0])
            tr, te = split_indices(ds, manifest, repeat, fold)
            genes = group.sort_values("rank").gene.tolist()
            if len(genes) != len(set(genes)) or any(g not in gene_index for g in genes):
                raise ValueError(f"{name} repeat={repeat} fold={fold} {method} k={k}: invalid fixed panel")
            columns = [gene_index[g] for g in genes]
            for model in args.models:
                with threadpool_limits(1):
                    result = tune_and_evaluate(model, ds.X[tr][:, columns], ds.y[tr], ds.X[te][:, columns], ds.y[te],
                                               seed=C.seed_for(C.dataset_seed_id(name), seed, fold, len(columns)), quick=args.quick)
                rows.append({"cohort": name, "repeat": repeat, "seed": seed, "fold": fold, "method": method,
                             "k": k, "n_features": len(columns), **result})
                print(f"[{name}] repeat {repeat} fold {fold} {method} k={k} {model}: AUROC={result['test_roc_auc']:.3f}", flush=True)
        result_df = pd.DataFrame(rows)
        result_df.to_csv(out / "classification_per_fold.csv", index=False)
        summary_cols = [f"test_{m}" for m in METRIC_NAMES] + ["tuning_time_s", "train_time_s", "infer_time_test_s"]
        summary = result_df.groupby(["cohort", "method", "k", "model"], as_index=False)[summary_cols].agg(["mean", "std"])
        summary.columns = ["_".join(c for c in col if c) for col in summary.columns.to_flat_index()]
        summary.to_csv(out / "classification_summary.csv", index=False)
        best = summary.sort_values("test_roc_auc_mean", ascending=False).groupby(["cohort", "method", "k"], as_index=False).first()
        best.to_csv(out / "best_classifier_by_panel.csv", index=False)
        cfg = {"experiment": "E1_classification", "dataset": name, "protocol":
               "fixed Phase-A panels; precomputed patient-grouped repeated 5x5 outer tests; parameter tuning is inner 3-fold CV on outer train only",
               "panel_file": str(panels_path), "panel_sha256": sha256(panels_path), "split_file": str(manifest_path),
               "split_sha256": sha256(manifest_path), "models": args.models, "quick_grid": args.quick,
               "inner_cv_folds": C.INNER_CV_FOLDS, "selection_reused": True,
               "metrics": list(METRIC_NAMES) + ["confusion_matrix", "tuning_time_s", "train_time_s", "infer_time_test_s"],
               "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", panels_path, manifest_path])


if __name__ == "__main__":
    main()
