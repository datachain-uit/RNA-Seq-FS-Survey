"""E1 phase A: select feature panels only, using precomputed 5x5 outer splits.

Each selection call receives only its manifest's outer-train samples.  The
resulting panels are deliberately stored separately from classifier metrics so
that Phase B can evaluate several classifiers without selecting features again.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, sha256, variance_prefilter  # noqa: E402
from survey_v2.e1_splits import validate_repeated_split_manifest  # noqa: E402
from survey_v2.methods import select  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--methods", nargs="+", default=list(C.FILTER_METHODS + C.EMBEDDED_METHODS))
    ap.add_argument("--k", nargs="+", type=int, default=list(C.K_GRID))
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(len(C.REPEAT_SEEDS))))
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--splits-root", type=Path, default=None,
                    help="root containing <cohort>/repeated_splits.csv; defaults to the cohort data directory")
    ap.add_argument("--out", default="E1_feature_selection")
    ap.add_argument("--pop", type=int, default=C.WRAPPER_POP)
    ap.add_argument("--generations", type=int, default=C.WRAPPER_GENERATIONS)
    ap.add_argument("--no-track-emissions", dest="track_emissions", action="store_false")
    ap.set_defaults(track_emissions=True)
    return ap.parse_args(argv)


def split_path(ds, root):
    return (Path(root) / ds.name / "repeated_splits.csv") if root else ds.data_dir / "repeated_splits.csv"


def split_indices(ds, manifest, repeat, fold):
    part = manifest[(manifest.repeat == repeat) & (manifest.fold == fold)]
    index = {sample: i for i, sample in enumerate(ds.sample_ids)}
    if set(part.sample_id) != set(index):
        raise ValueError(f"{ds.name}: manifest samples differ from expression samples")
    expected_patient = dict(zip(ds.sample_ids, ds.groups))
    expected_label = dict(zip(ds.sample_ids, ds.y))
    for row in part[["sample_id", "patient_id", "label"]].itertuples(index=False):
        if expected_patient[row.sample_id] != str(row.patient_id) or expected_label[row.sample_id] != int(row.label):
            raise ValueError(f"{ds.name}: manifest patient_id or label does not match metadata for {row.sample_id}")
    train = np.array([index[s] for s in part.loc[part.role == "train", "sample_id"]], dtype=int)
    test = np.array([index[s] for s in part.loc[part.role == "test", "sample_id"]], dtype=int)
    return train, test


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else float("nan")


def start_emissions_tracker(enabled):
    if not enabled:
        return None
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(project_name="e1_feature_selection", country_iso_code="VNM", save_to_file=False, log_level="error")
        tracker.start()
        return tracker
    except Exception as exc:
        print(f"[emissions] unavailable: {exc}", flush=True)
        return None


def checkpoint_paths(out: Path, repeat: int, fold: int, method: str) -> tuple[Path, Path]:
    """Two complete files are the atomic completion marker for one selection unit."""
    base = out / "checkpoints" / f"repeat_{repeat}_fold_{fold}_{method}"
    return base.with_suffix(".metrics.csv"), base.with_suffix(".genes.csv")


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        path = split_path(ds, args.splits_root)
        manifest = pd.read_csv(path)
        errors = validate_repeated_split_manifest(manifest, n_folds=C.N_FOLDS)
        if errors:
            raise ValueError(f"{name}: invalid repeated_splits.csv: {'; '.join(errors)}")
        process = psutil.Process()
        runs = manifest[["repeat", "seed", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
        for run in runs.itertuples(index=False):
            if run.repeat not in args.repeats:
                continue
            tr, _ = split_indices(ds, manifest, run.repeat, run.fold)
            pre = variance_prefilter(ds.X[tr], args.prefilter)
            for method in args.methods:
                metrics_path, genes_path = checkpoint_paths(out, int(run.repeat), int(run.fold), method)
                if metrics_path.is_file() and genes_path.is_file():
                    print(f"[{name}] repeat {run.repeat} fold {run.fold} {method}: checkpoint exists, skip", flush=True)
                    continue
                method_id = (C.FILTER_METHODS + C.EMBEDDED_METHODS + C.WRAPPER_METHODS).index(method)
                seed = C.seed_for(list(C.DATASETS).index(name), int(run.seed), int(run.fold), method_id)
                rss_before, cpu_before, t0 = process.memory_info().rss, time.process_time(), time.perf_counter()
                tracker = start_emissions_tracker(args.track_emissions)
                with threadpool_limits(1):
                    selected = select(method, ds.X[tr][:, pre], ds.y[tr], args.k, seed=seed, n_jobs=1,
                                      wrapper_kwargs={"pop": args.pop, "generations": args.generations})
                emissions_kg = tracker.stop() if tracker else float("nan")
                elapsed, cpu_s = time.perf_counter() - t0, time.process_time() - cpu_before
                fold_rows, fold_genes = [], []
                for k in args.k:
                    panel = selected.panels[int(k)]
                    info = selected.info
                    fold_rows.append({"cohort": name, "repeat": run.repeat, "seed": run.seed, "fold": run.fold,
                                 "method": method, "k": int(k), "n_train": len(tr), "p0": len(pre),
                                 "n_selected": len(panel), "exact_k": info.get("exact", {}).get(int(k), True),
                                 "n_filled": info.get("n_filled", {}).get(int(k), 0),
                                 "selector_param": info.get("C", {}).get(int(k)),
                                 "n_eval": info.get("n_eval", {}).get(int(k)),
                                 "best_fitness": info.get("best_fitness", {}).get(int(k)),
                                 "selection_time_s": elapsed, "cpu_time_s": cpu_s,
                                 "rss_delta_mb": (process.memory_info().rss - rss_before) / 2**20,
                                 "emissions_kg_co2": emissions_kg})
                    fold_genes.extend({"cohort": name, "repeat": run.repeat, "fold": run.fold, "method": method,
                                  "k": int(k), "rank": rank, "gene": ds.genes[pre[col]]}
                                 for rank, col in enumerate(panel, 1))
                write_csv_atomic(pd.DataFrame(fold_rows), metrics_path)
                write_csv_atomic(pd.DataFrame(fold_genes), genes_path)
                print(f"[{name}] repeat {run.repeat} fold {run.fold} {method}: checkpoint saved", flush=True)
        checkpoint_dir = out / "checkpoints"
        run_df = pd.concat([pd.read_csv(p) for p in checkpoint_dir.glob("*.metrics.csv")], ignore_index=True)
        gene_df = pd.concat([pd.read_csv(p) for p in checkpoint_dir.glob("*.genes.csv")], ignore_index=True)
        run_df.to_csv(out / "selection_per_fold.csv", index=False)
        gene_df.to_csv(out / "selected_genes.csv", index=False)
        stability = []
        for key, group in gene_df.groupby(["cohort", "method", "k"], sort=False):
            panels = [set(x.gene) for _, x in group.groupby(["repeat", "fold"])]
            pairs = [jaccard(a, b) for i, a in enumerate(panels) for b in panels[i + 1:]]
            stability.append({"cohort": key[0], "method": key[1], "k": key[2], "n_panels": len(panels),
                              "mean_pairwise_jaccard": float(np.mean(pairs)) if pairs else float("nan")})
        pd.DataFrame(stability).to_csv(out / "selection_stability.csv", index=False)
        gene_df.groupby(["cohort", "method", "k", "gene"], as_index=False).size().rename(
            columns={"size": "selected_in_outer_folds"}).to_csv(out / "selection_gene_frequency.csv", index=False)
        cfg = {"experiment": "E1_feature_selection", "dataset": name, "protocol":
               "precomputed patient-grouped repeated 5x5 outer train/test; selection and variance prefilter use outer train only",
               "split_file": str(path), "split_sha256": sha256(path), "methods": args.methods, "k_grid": args.k,
               "prefilter_p0": args.prefilter, "metrics": ["n_selected", "exact_k", "n_filled", "selection_time_s", "cpu_time_s", "rss_delta_mb", "emissions_kg_co2", "n_eval", "best_fitness", "pairwise_jaccard"],
               "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", path])


if __name__ == "__main__":
    main()
