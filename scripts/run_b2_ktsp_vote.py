"""B2 (QĐ172, đặc tả 12/11): k-TSP với luật bỏ phiếu gốc, trên đúng các cặp gen mà bộ chọn k-TSP đã chọn.

Hướng mỗi cặp (i, j) học trên phần train: dự đoán lớp 1 (LUSC) khi x_i > x_j nếu P(x_i > x_j | 1) > P(x_i > x_j | 0), ngược lại
đảo. Điểm của một mẫu = tỉ lệ cặp bỏ phiếu lớp 1 (dùng cho AUROC); lớp 1 khi điểm > 0,5; hoà thì cặp đầu (Δ lớn nhất) quyết
định. Không chuẩn hoá (survey_v2/revision.py: ktsp_directions, ktsp_vote).

* e1  : 5 bộ × 25 fold, k ∈ {2, 4, 10}; cặp = panel k-TSP đã công bố (E1_feature_selection/<bộ>/selected_genes.csv); không chọn lại.
* e1b : 4 cohort GEO, 5 lần chia đã công bố, mọi (tập con, n_T), k ∈ {2, 4, 10}; cặp = e1b_selected_genes.csv đã công bố.
* e7  : source_only (cặp và hướng học trên TCGA, áp lên test_T) và select_on_source (cặp từ TCGA, hướng học lại trên L_T);
        cặp k-TSP chọn trên TCGA sau lọc phương sai 5.000 gen của nguồn (như E7), n_T ∈ {5, 10, 20}. Mỗi dòng kèm AUROC của
        k-TSP + LR-L2 (φ cho source_only, như E7) trên cùng cặp và cùng lần chạy để tính Δ.

    python scripts/run_b2_ktsp_vote.py --datasets gse37745
    python scripts/run_b2_ktsp_vote.py --parts e1 --datasets tcga_luad_lusc_primary_tumor

Đầu ra: results/survey_v2/B2_ktsp_vote/<bộ>/{b2_e1_per_fold.csv, b2_e1b_per_run.csv, b2_e7_per_run.csv, config.json}.
Đối chiếu k-TSP + LR-L2: E1, E1b lấy dòng `ktsp` đã công bố (cùng panel); E7 tính lại trên cùng cặp vì E7 công bố dùng L1-LR.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.fewshot import fewshot_runs, validate_fewshot_manifest  # noqa: E402
from survey_v2.revision import PUBLISHED_ROOT, PUBLISHED_TAG, evaluate_vote, ktsp_directions, ktsp_pairs_from_panel  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.transfer import TransferContext, panel_transform  # noqa: E402
from scripts.build_fewshot_splits import fewshot_path  # noqa: E402
from scripts.run_e1_feature_selection import split_indices, split_path  # noqa: E402

B2_K = (2, 4, 10)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN_DATASETS))
    ap.add_argument("--parts", nargs="+", default=["e1", "e1b", "e7"], choices=["e1", "e1b", "e7"])
    ap.add_argument("--k", nargs="+", type=int, default=list(B2_K))
    ap.add_argument("--n-t", nargs="+", default=[str(v) for v in C.TRANSFER_N_T], help="E7: mức n_T")
    ap.add_argument("--published-root", type=Path, default=PUBLISHED_ROOT)
    ap.add_argument("--source", default=C.SOURCE_DATASET)
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--out", default="B2_ktsp_vote")
    return ap.parse_args(argv)


def _panel_cols(genes_df: pd.DataFrame, gene_index: dict) -> list[int]:
    genes = genes_df.sort_values("rank").gene.tolist()
    if len(genes) != len(set(genes)) or any(g not in gene_index for g in genes):
        raise ValueError("panel k-TSP đã công bố không hợp lệ với không gian gen của bộ")
    return [gene_index[g] for g in genes]


def run_e1(ds, args, out: Path) -> tuple[pd.DataFrame, dict]:
    pub = args.published_root
    sel = pd.read_csv(pub / "E1_feature_selection" / ds.name / "selected_genes.csv")
    sel = sel[(sel.method == "ktsp") & sel.k.isin(args.k)]
    lr = pd.read_csv(pub / "E1_classification" / ds.name / "classification_per_fold.csv")
    lr = lr[(lr.method == "ktsp") & (lr.model == "lr_l2") & lr.k.isin(args.k)][["repeat", "fold", "k", "auroc", "bacc", "mcc"]]
    manifest_path = split_path(ds, args.splits_root)
    manifest = pd.read_csv(manifest_path)
    gene_index = {g: i for i, g in enumerate(ds.genes)}
    rows = []
    for (repeat, fold), g in sel.groupby(["repeat", "fold"], sort=True):
        tr, te = split_indices(ds, manifest, int(repeat), int(fold))
        seed = int(manifest.loc[manifest.repeat == repeat, "seed"].iloc[0])
        for k, gk in g.groupby("k", sort=True):
            pairs = ktsp_pairs_from_panel(_panel_cols(gk, gene_index))
            r = evaluate_vote(ds.X[tr], ds.y[tr], ds.X[te], ds.y[te], pairs)
            rows.append({"cohort": ds.name, "repeat": int(repeat), "seed": seed, "fold": int(fold), "k": int(k), "n_test": len(te),
                         "auroc_vote": r["auroc"], "bacc_vote": r["bacc"], "mcc_vote": r["mcc"], "n_pairs": r["n_pairs"],
                         "n_direction_ties": r["n_direction_ties"], "n_score_ties": r["n_score_ties"], "directions": r["directions"]})
    df = pd.DataFrame(rows).merge(lr.rename(columns={"auroc": "auroc_lr", "bacc": "bacc_lr", "mcc": "mcc_lr"}),
                                  on=["repeat", "fold", "k"], how="left")
    df["delta_auroc"] = df.auroc_vote - df.auroc_lr
    df.to_csv(out / "b2_e1_per_fold.csv", index=False, encoding="utf-8-sig")
    return df, {"split_file": str(manifest_path), "split_sha256": sha256(manifest_path),
                "published_panels_sha256": sha256(pub / "E1_feature_selection" / ds.name / "selected_genes.csv")}


def run_e1b(ds, args, out: Path) -> tuple[pd.DataFrame, dict]:
    pub = args.published_root
    sel = pd.read_csv(pub / "E1b" / ds.name / "e1b_selected_genes.csv", dtype={"n_t": str})
    sel = sel[(sel.method == "ktsp") & sel.k.isin(args.k)]
    lr = pd.read_csv(pub / "E1b" / ds.name / "e1b_per_run.csv", dtype={"n_t": str})
    lr = lr[(lr.method == "ktsp") & lr.k.isin(args.k)][["split", "subset", "n_t", "k", "auroc", "bacc", "mcc"]]
    path = fewshot_path(ds, args.splits_root)
    man = pd.read_csv(path, dtype={"n_t": str})
    errors = validate_fewshot_manifest(man)
    if errors:
        raise ValueError(f"{ds.name}: fewshot_splits.csv không hợp lệ: {errors}")
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    gene_index = {g: i for i, g in enumerate(ds.genes)}
    panels = {key: g for key, g in sel.groupby(["split", "subset", "n_t", "k"], sort=False)}
    rows = []
    for split, seed, subset, n_t, train_ids, test_ids in fewshot_runs(man):
        tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
        for k in args.k:
            g = panels.get((split, subset, n_t, k))
            if g is None:
                continue
            pairs = ktsp_pairs_from_panel(_panel_cols(g, gene_index))
            r = evaluate_vote(ds.X[tr], ds.y[tr], ds.X[te], ds.y[te], pairs)
            rows.append({"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t, "k": int(k), "n_train": len(tr),
                         "n_test": len(te), "auroc_vote": r["auroc"], "bacc_vote": r["bacc"], "mcc_vote": r["mcc"],
                         "n_pairs": r["n_pairs"], "n_direction_ties": r["n_direction_ties"], "n_score_ties": r["n_score_ties"]})
    df = pd.DataFrame(rows).merge(lr.rename(columns={"auroc": "auroc_lr", "bacc": "bacc_lr", "mcc": "mcc_lr"}),
                                  on=["split", "subset", "n_t", "k"], how="left")
    df["delta_auroc"] = df.auroc_vote - df.auroc_lr
    df.to_csv(out / "b2_e1b_per_run.csv", index=False, encoding="utf-8-sig")
    return df, {"fewshot_splits_file": str(path), "fewshot_splits_sha256": sha256(path)}


def run_e7(ds, source, args, out: Path) -> tuple[pd.DataFrame, dict]:
    cid = C.cohort_id(ds.name)
    path = fewshot_path(ds, args.splits_root)
    man = pd.read_csv(path, dtype={"n_t": str})
    with threadpool_limits(1):
        ctx = TransferContext(source, ds, args.k, selector="ktsp", prefilter=args.prefilter, seed=C.seed_for(cid, 0, 0, 7))
    src_pairs = {k: ktsp_pairs_from_panel(ctx.source_panels[k]) for k in args.k}
    src_dirs = {k: ktsp_directions(ctx.Xs, ctx.ys, src_pairs[k])[0] for k in args.k}
    rows, cache_so = [], {}
    for split, seed, subset, n_t, train_ids, test_ids in fewshot_runs(man):
        if n_t not in args.n_t:
            continue
        index = {s: i for i, s in enumerate(ds.sample_ids)}
        tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
        n_t_int = 10**6 if n_t == "pool" else int(n_t)
        base = {"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t, "n_train": len(tr), "n_test": len(te)}
        with threadpool_limits(1):
            for k in args.k:
                seed_k = C.seed_for(cid, seed, subset, n_t_int, k, 77)
                pairs = src_pairs[k]
                # source_only: hướng từ nguồn; đối chiếu = LR-L2 trên φ của cặp (như E7), không phụ thuộc tập con
                v = evaluate_vote(None, None, ctx.Xt[te], ctx.yt[te], pairs, dirs=src_dirs[k])
                if (split, k) not in cache_so:
                    cache_so[(split, k)] = ctx.source_only(k, te)
                lr = cache_so[(split, k)]
                rows.append({**base, "strategy": "source_only", "k": int(k), "auroc_vote": v["auroc"], "bacc_vote": v["bacc"],
                             "mcc_vote": v["mcc"], "auroc_lr": lr.auroc, "bacc_lr": lr.bacc, "mcc_lr": lr.mcc,
                             "lr_normalisation": C.PANEL_NORMALISATION, "directions": v["directions"]})
                # select_on_source: hướng học lại trên L_T; đối chiếu = LR-L2 fit trên L_T (như E7)
                v = evaluate_vote(ctx.Xt[tr], ctx.yt[tr], ctx.Xt[te], ctx.yt[te], pairs)
                lr = ctx.select_on_source(k, tr, te, seed_k)
                rows.append({**base, "strategy": "select_on_source", "k": int(k), "auroc_vote": v["auroc"], "bacc_vote": v["bacc"],
                             "mcc_vote": v["mcc"], "auroc_lr": lr.auroc, "bacc_lr": lr.bacc, "mcc_lr": lr.mcc, "lr_normalisation": "none",
                             "directions": v["directions"], "n_direction_ties": v["n_direction_ties"]})
    df = pd.DataFrame(rows)
    df["delta_auroc"] = df.auroc_vote - df.auroc_lr
    df.to_csv(out / "b2_e7_per_run.csv", index=False, encoding="utf-8-sig")
    return df, {"fewshot_splits_file": str(path), "fewshot_splits_sha256": sha256(path), "gene_space_size": int(len(ctx.genes)),
                "source_pairs": {int(k): [[str(ctx.genes[a]), str(ctx.genes[b])] for a, b in src_pairs[k]] for k in args.k},
                "source_directions": {int(k): src_dirs[k].tolist() for k in args.k}, "prefilter_p0_on_source": int(len(ctx.pre)),
                "source_sha256": source.files_sha256}


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t_all = time.time()
    source = None
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        info, counts = {}, {}
        if "e1" in args.parts:
            df, info["e1"] = run_e1(ds, args, out)
            counts["e1"] = len(df)
        geo = name in C.FEWSHOT_COHORTS
        if "e1b" in args.parts and geo:
            df, info["e1b"] = run_e1b(ds, args, out)
            counts["e1b"] = len(df)
        if "e7" in args.parts and geo:
            if source is None:
                source = load_dataset(DATA_ROOT, args.source, gene_space=None)
            df, info["e7"] = run_e7(ds, source, args, out)
            counts["e7"] = len(df)
        cfg = {"experiment": "B2_ktsp_vote", "decision": "QĐ172 (Review 12/11), spec YEU-CAU-DOI-CODE-2026-11-12-GVHD-B1-B7 §B2",
               "dataset": name, "parts": [p for p in args.parts if p in counts], "k_grid": args.k, "n_t_e7": args.n_t,
               "rule": "pair direction learned on train (P(x_i>x_j|1) vs P(x_i>x_j|0)); score = share of pairs voting class 1; "
                       "class 1 if score > 0.5; tie -> top pair decides; no normalisation",
               "published_root": str(args.published_root), "published_tag": PUBLISHED_TAG, "rows": counts, **info,
               "choices_to_confirm": ["direction tie P(x_i>x_j|1) == P(x_i>x_j|0): keep 'x_i > x_j votes class 1' (count in n_direction_ties)",
                                      "E7 k-TSP + LR-L2 reference recomputed on the same source pairs (published E7 used L1-LR panels)"],
               "elapsed_s": time.time() - t_all, "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv"])
        print(f"[{name}] B2 xong: {counts}", flush=True)


if __name__ == "__main__":
    main()
