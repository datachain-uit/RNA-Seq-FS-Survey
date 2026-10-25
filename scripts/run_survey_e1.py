"""E1 — Hiệu năng theo ngân sách k (survey v2, Bảng 11 Review 24/10/2026): filter + embedded + B0 + random-k.

Giao thức: CV 5×5 phân tầng theo bệnh nhân (seed 42–46), lọc sơ bộ không dùng nhãn 5.000 gen trong fold,
10 phương pháp (wrapper chạy bằng scripts/run_survey_e1_wrappers.py), LR-L2 chính (C chọn bằng CV nội bộ),
RF và SVM tuyến tính phụ ở k ∈ {50, 200}; AUROC chính, BAcc, MCC. Mọi lần chạy ghi provenance.

    $env:KLTN_DATA_ROOT = "D:\\KLTN"
    python scripts/run_survey_e1.py                                   # 3 bộ, filter + embedded
    python scripts/run_survey_e1.py --datasets gse68465_stage_early_late --methods f_test mrmr --repeats 0 --n-jobs 4
    python scripts/run_survey_e1.py --secondary                        # thêm RF, SVM tuyến tính ở k = 50, 200

Đầu ra: $KLTN_DATA_ROOT/results/survey_v2/E1/<bộ>/{e1_per_fold.csv, e1_selected_genes.csv, e1_random_k.csv,
e1_scores.csv, cv_splits.csv, config.json}. Chạy lại sẽ bỏ qua các (repeat, fold, method) đã có (resume).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import load_dataset, repeated_patient_splits, sha256, splits_to_frame, variance_prefilter  # noqa: E402
from survey_v2.methods import random_panel, select  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402

COHORT_ID = {name: i for i, name in enumerate(C.DATASETS)}
METHOD_ID = {m: i for i, m in enumerate(C.FILTER_METHODS + C.EMBEDDED_METHODS + C.WRAPPER_METHODS)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--methods", nargs="+", default=list(C.FILTER_METHODS + C.EMBEDDED_METHODS),
                    help="filter/embedded; wrapper dùng run_survey_e1_wrappers.py")
    ap.add_argument("--k", nargs="+", type=int, default=list(C.K_GRID))
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(len(C.REPEAT_SEEDS))),
                    help="chỉ số lần lặp (0..4) cần chạy")
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--random-draws", type=int, default=C.N_RANDOM_DRAWS)
    ap.add_argument("--no-b0", action="store_true")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--secondary", action="store_true", help="chạy thêm RF và SVM tuyến tính ở k ∈ SECONDARY_K")
    ap.add_argument("--n-jobs", type=int, default=1, help="số fold chạy song song (mỗi fold 1 luồng BLAS)")
    ap.add_argument("--out", default="E1", help="thư mục con trong results/survey_v2/")
    ap.add_argument("--pop", type=int, default=C.WRAPPER_POP, help="(wrapper) kích thước quần thể")
    ap.add_argument("--generations", type=int, default=C.WRAPPER_GENERATIONS, help="(wrapper) số thế hệ")
    return ap.parse_args(argv)


def run_fold(ds, repeat, seed, fold, tr, te, args, done: set):
    """Một fold: lọc sơ bộ, B0, random-k, các phương pháp; trả về (rows, gene_rows, random_rows, score_rows)."""
    rows, gene_rows, random_rows, score_rows = [], [], [], []
    cid = COHORT_ID.get(ds.name, 99)
    Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
    base = {"cohort": ds.name, "repeat": repeat, "seed": seed, "fold": fold, "n_train": len(tr), "n_test": len(te)}
    with threadpool_limits(1):
        pre = variance_prefilter(Xtr, args.prefilter)
        Xtr_p, Xte_p = Xtr[:, pre], Xte[:, pre]
        p0 = len(pre)
        classifiers_for = lambda k: ["lr_l2"] + (["rf", "svm_linear"] if args.secondary and k in C.SECONDARY_K else [])

        def add_eval(method, k, sel_idx, clf, extra=None):
            t0 = time.perf_counter()
            r = evaluate_panel(Xtr_p[:, sel_idx], ytr, Xte_p[:, sel_idx], yte, classifier=clf,
                               seed=C.seed_for(cid, seed, fold, 1000 + k))
            row = {**base, "method": method, "k": int(k), "classifier": clf, "auroc": r.auroc, "bacc": r.bacc,
                   "mcc": r.mcc, "threshold": r.threshold, "hyperparam": r.hyperparam, "n_selected": len(sel_idx),
                   "p0": p0, "eval_time_s": time.perf_counter() - t0, **(extra or {})}
            rows.append(row)
            for sid, yy, s in zip(ds.sample_ids[te], yte, r.scores_test):
                score_rows.append({"cohort": ds.name, "repeat": repeat, "fold": fold, "method": method, "k": int(k),
                                   "classifier": clf, "sample_id": sid, "y": int(yy), "score": float(s)})

        # B0: toàn bộ gen, không lọc
        if not args.no_b0 and ("b0" not in done):
            t0 = time.perf_counter()
            r = evaluate_panel(Xtr, ytr, Xte, yte, classifier="lr_l2", seed=C.seed_for(cid, seed, fold, 0))
            rows.append({**base, "method": "b0", "k": ds.p, "classifier": "lr_l2", "auroc": r.auroc, "bacc": r.bacc,
                         "mcc": r.mcc, "threshold": r.threshold, "hyperparam": r.hyperparam, "n_selected": ds.p,
                         "p0": ds.p, "eval_time_s": time.perf_counter() - t0})
            for sid, yy, s in zip(ds.sample_ids[te], yte, r.scores_test):
                score_rows.append({"cohort": ds.name, "repeat": repeat, "fold": fold, "method": "b0", "k": ds.p,
                                   "classifier": "lr_l2", "sample_id": sid, "y": int(yy), "score": float(s)})

        # random-k: 100 lần rút mỗi (fold, k) trong KHÔNG GIAN ĐÃ LỌC SƠ BỘ (seed theo Y14)
        if not args.no_random and ("random_k" not in done):
            for k in args.k:
                for draw in range(args.random_draws):
                    rng = np.random.default_rng([cid, seed, fold, k, draw])
                    sel = random_panel(p0, k, rng)
                    r = evaluate_panel(Xtr_p[:, sel], ytr, Xte_p[:, sel], yte, classifier="lr_l2",
                                       seed=C.seed_for(cid, seed, fold, 2000 + k, draw))
                    random_rows.append({**base, "method": "random_k", "k": int(k), "draw": draw, "classifier": "lr_l2",
                                        "auroc": r.auroc, "bacc": r.bacc, "mcc": r.mcc, "hyperparam": r.hyperparam})

        # các phương pháp chọn gen
        for method in args.methods:
            if method in done:
                continue
            t0 = time.perf_counter()
            selection = select(method, Xtr_p, ytr, args.k, seed=C.seed_for(cid, seed, fold, METHOD_ID.get(method, 50)),
                               n_jobs=1, wrapper_kwargs={"pop": args.pop, "generations": args.generations})
            sel_time = time.perf_counter() - t0
            for k in args.k:
                sel_idx = selection.panels[int(k)]
                assert len(sel_idx) == k
                info = {"select_time_s": sel_time,
                        "n_filled": int(selection.info.get("n_filled", {}).get(int(k), 0)),
                        "exact_k": bool(selection.info.get("exact", {}).get(int(k), True)),
                        "sel_hyperparam": selection.info.get("C", {}).get(int(k)),
                        "n_eval": selection.info.get("n_eval", {}).get(int(k)),
                        "best_fitness": selection.info.get("best_fitness", {}).get(int(k))}
                for clf in classifiers_for(k):
                    add_eval(method, k, sel_idx, clf, info)
                for rank, j in enumerate(sel_idx):
                    gene_rows.append({"cohort": ds.name, "repeat": repeat, "fold": fold, "method": method, "k": int(k),
                                      "rank": rank + 1, "gene": ds.genes[pre[j]]})
    return rows, gene_rows, random_rows, score_rows


def append_csv(path: Path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists(), encoding="utf-8-sig")


def main(argv=None, script=__file__) -> None:
    args = parse_args(argv)
    params = {k: v for k, v in vars(args).items()}
    prov = start_run(script, params=params)
    t_start = time.time()
    for name in args.datasets:
        ds = load_dataset(DATA_ROOT, name)
        out = results_dir(args.out, name)
        all_splits = repeated_patient_splits(ds.y, ds.groups)
        splits = [s for s in all_splits if s[0] in set(args.repeats)]
        splits_to_frame(ds, all_splits).to_csv(out / "cv_splits.csv", index=False)
        per_fold = out / "e1_per_fold.csv"
        done_map: dict[tuple, set] = {}
        for fname, label in ((per_fold, None), (out / "e1_random_k.csv", "random_k")):
            if fname.exists():
                prev = pd.read_csv(fname)
                for (r, f), g in prev.groupby(["repeat", "fold"]):
                    done_map.setdefault((int(r), int(f)), set()).update({label} if label else set(g["method"].unique()))
        print(f"[{name}] {ds.n} mẫu × {ds.p} gen | {len(splits)} fold cần xét | đã có {len(done_map)} fold", flush=True)
        jobs = [(repeat, seed, fold, tr, te, done_map.get((repeat, fold), set())) for repeat, seed, fold, tr, te in splits]
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(run_fold)(ds, repeat, seed, fold, tr, te, args, done) for repeat, seed, fold, tr, te, done in jobs)
        for rows, gene_rows, random_rows, score_rows in results:
            append_csv(per_fold, rows)
            append_csv(out / "e1_selected_genes.csv", gene_rows)
            append_csv(out / "e1_random_k.csv", random_rows)
            append_csv(out / "e1_scores.csv", score_rows)
        cfg = {"experiment": "E1", "dataset": name, "role": C.DATASETS.get(name), "n_samples": ds.n, "n_genes": ds.p,
               "class_counts": {int(c): int((ds.y == c).sum()) for c in (0, 1)},
               "protocol": "repeated stratified 5x5 CV at patient level; label-free variance prefilter inside fold",
               "repeat_seeds": list(C.REPEAT_SEEDS), "repeats_run": args.repeats, "k_grid": args.k,
               "prefilter_p0": args.prefilter, "methods": args.methods, "b0": not args.no_b0,
               "random_k_draws": 0 if args.no_random else args.random_draws,
               "random_k_space": "prefiltered p0 genes of the fold", "random_k_seed": "default_rng([cohort_id, split_seed, fold, k, draw])",
               "classifier_main": "StandardScaler + LogisticRegression(L2, liblinear, balanced); C by inner 3-fold CV (AUROC); threshold by inner OOF BAcc",
               "classifiers_secondary": ["rf", "svm_linear"] if args.secondary else [], "secondary_k": list(C.SECONDARY_K),
               "mrmr": {"criterion": C.MRMR_CRITERION, "discretize": C.MRMR_DISCRETIZE},
               "standardize_before": list(C.STANDARDIZE_BEFORE), "versions": all_versions(),
               "cv_splits_sha256": sha256(out / "cv_splits.csv"), "elapsed_s": time.time() - t_start}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv"])
        print(f"[{name}] xong, ghi vào {out}", flush=True)


if __name__ == "__main__":
    main()
