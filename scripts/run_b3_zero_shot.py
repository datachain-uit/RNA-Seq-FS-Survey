"""B3 (QĐ173, đặc tả 12/11): hai biến thể zero-shot (source-only) của E7, trên đúng các tập test_T của E7 (5 lần chia).

* e7_phi      : source_only của E7 tính lại (panel L1-LR trên nguồn, φ = z-score trong mẫu trên panel) — mốc đối chiếu, phải
                trùng E7 đã công bố.
* (i)  consistent_phi : áp z-score TRONG MẪU trên 5.000 gen đã lọc của nguồn, chọn gen bằng L1-LR trên dữ liệu đó (đúng k),
                rồi phân loại trên φ của panel như E7 (LR-L2 fit trên φ của nguồn, áp lên φ của test_T).
* (ii) target_gene_zscore : panel của E7 (L1-LR trên nguồn); z-score TỪNG GEN của panel trên TCGA (huấn luyện) và trên TOÀN BỘ
                mẫu của cohort đích (áp dụng). Không dùng nhãn đích; CÓ dùng dữ liệu đích không nhãn (ghi trong config).

LR-L2 như source_only của E7: CV nội bộ trên nguồn chọn C, seed của context E7. k ∈ {2, 3, 4, 5, 10}, bốn cohort GEO.

    python scripts/run_b3_zero_shot.py --datasets gse37745

Đầu ra: results/survey_v2/B3_zero_shot/<cohort>/{b3_per_split.csv, b3_panels.csv, config.json}.
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
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.methods import select  # noqa: E402
from survey_v2.revision import PUBLISHED_ROOT, PUBLISHED_TAG, per_gene_zscore, within_sample_zscore  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.transfer import TransferContext, panel_transform  # noqa: E402
from scripts.build_fewshot_splits import fewshot_path  # noqa: E402

VARIANTS = ("e7_phi", "consistent_phi", "target_gene_zscore")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--k", nargs="+", type=int, default=list(C.TRANSFER_K_GRID))
    ap.add_argument("--source", default=C.SOURCE_DATASET)
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--published-root", type=Path, default=PUBLISHED_ROOT)
    ap.add_argument("--out", default="B3_zero_shot")
    return ap.parse_args(argv)


def test_sets(man: pd.DataFrame) -> list[tuple[int, int, np.ndarray]]:
    """(split, seed, sample_id của test_T) — test_T cố định trong một lần chia (validate_fewshot_manifest)."""
    out = []
    for (split, seed), g in man[man.role == "test"].groupby(["split", "seed"], sort=True):
        first = g[(g.subset == g.subset.min())]
        first = first[first.n_t == first.n_t.iloc[0]]
        out.append((int(split), int(seed), np.sort(first.sample_id.to_numpy(dtype=str))))
    return out


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t_all = time.time()
    source = load_dataset(DATA_ROOT, args.source, gene_space=None)
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        cid = C.cohort_id(name)
        path = fewshot_path(ds, args.splits_root)
        man = pd.read_csv(path, dtype={"n_t": str})
        with threadpool_limits(1):
            ctx = TransferContext(source, ds, args.k, selector=C.TRANSFER_SELECTOR, prefilter=args.prefilter, seed=C.seed_for(cid, 0, 0, 7))
            # (i) z-score trong mẫu trên 5.000 gen đã lọc của nguồn, rồi L1-LR đúng k trên dữ liệu đó (cùng seed với E7)
            Zs = within_sample_zscore(ctx.Xs[:, ctx.pre])
            sel_i = select(C.TRANSFER_SELECTOR, Zs, ctx.ys, ctx.k_list, seed=ctx.seed)
            panels_i = {k: ctx.pre[np.asarray(sel_i.panels[k])] for k in ctx.k_list}
        index = {s: i for i, s in enumerate(ds.sample_ids)}
        rows = []
        all_t = np.arange(len(ctx.yt))
        for split, seed, test_ids in test_sets(man):
            te = np.array([index[s] for s in test_ids])
            with threadpool_limits(1):
                for k in ctx.k_list:
                    S = ctx.source_panels[k]
                    res = {"e7_phi": ctx.source_only(k, te)}
                    Si = panels_i[k]
                    res["consistent_phi"] = evaluate_panel(panel_transform(ctx.Xs[:, Si]), ctx.ys, panel_transform(ctx.Xt[te][:, Si]),
                                                           ctx.yt[te], classifier="lr_l2", seed=ctx.seed, small_n=False)
                    Zt_all = per_gene_zscore(ctx.Xt[all_t][:, S])          # toàn bộ mẫu đích, không nhãn
                    res["target_gene_zscore"] = evaluate_panel(per_gene_zscore(ctx.Xs[:, S]), ctx.ys, Zt_all[te], ctx.yt[te],
                                                               classifier="lr_l2", seed=ctx.seed, small_n=False)
                    for v in VARIANTS:
                        r = res[v]
                        rows.append({"cohort": name, "split": split, "seed": seed, "n_test": len(te), "variant": v, "k": int(k),
                                     "auroc": r.auroc, "bacc": r.bacc, "mcc": r.mcc, "logloss": r.logloss, "brier": r.brier, "C": r.hyperparam,
                                     "uses_unlabelled_target": v == "target_gene_zscore"})
        df = pd.DataFrame(rows)
        df.to_csv(out / "b3_per_split.csv", index=False, encoding="utf-8-sig")
        prow = [{"cohort": name, "variant": v, "k": int(k), "rank": r + 1, "gene": str(ctx.genes[j])}
                for v, P in (("e7_phi", ctx.source_panels), ("consistent_phi", panels_i), ("target_gene_zscore", ctx.source_panels))
                for k in ctx.k_list for r, j in enumerate(P[k])]
        pd.DataFrame(prow).to_csv(out / "b3_panels.csv", index=False, encoding="utf-8-sig")
        # đối chiếu panel nguồn với E7 đã công bố (cùng context, cùng seed -> phải trùng)
        gate = {}
        pub_cfg = args.published_root / "E7" / name / "config.json"
        if pub_cfg.exists():
            import json
            pub = json.loads(pub_cfg.read_text(encoding="utf-8")).get("source_panels", {})
            gate = {"source_panels_match_published_e7": all(pub.get(str(k)) == ctx.genes[ctx.source_panels[k]].tolist() for k in ctx.k_list)}
        cfg = {"experiment": "B3_zero_shot", "decision": "QĐ173 (Review 12/11), spec §B3", "dataset": name, "variants": list(VARIANTS),
               "k_grid": ctx.k_list, "selector": C.TRANSFER_SELECTOR, "source": args.source, "source_sha256": source.files_sha256,
               "gene_space_size": int(len(ctx.genes)), "prefilter_p0_on_source": int(len(ctx.pre)),
               "variant_i": "within-sample z-score over the 5,000 filtered source genes, L1-LR exact-k on it, classify on phi of the panel (as E7)",
               "variant_ii": "per-gene z-score of the E7 panel on TCGA (train) and on ALL samples of the target cohort (apply); "
                             "no target labels; USES UNLABELLED TARGET DATA",
               "panels_i": {int(k): ctx.genes[v].tolist() for k, v in panels_i.items()},
               "panels_e7": {int(k): ctx.genes[v].tolist() for k, v in ctx.source_panels.items()},
               "panels_i_info": {kk: {int(k): (vv.get(k) if isinstance(vv, dict) else None) for k in ctx.k_list}
                                 for kk, vv in sel_i.info.items() if isinstance(vv, dict)},
               "fewshot_splits_file": str(path), "fewshot_splits_sha256": sha256(path), "published_root": str(args.published_root),
               "published_tag": PUBLISHED_TAG, **gate, "n_rows": int(len(df)), "elapsed_s": time.time() - t_all, "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", path,
                                                 source.data_dir / "expression.h5", source.data_dir / "metadata.csv"])
        print(f"[{name}] B3 xong: {len(df)} dòng; {gate}", flush=True)


if __name__ == "__main__":
    main()
