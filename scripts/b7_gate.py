"""B7 — cổng G1, G2 (QĐ177, QĐ188; quy tắc chốt 12/11, Review 13/11 Phụ lục A.4). Chỉ đọc kết quả đã chạy, không fit mô hình.

* G1: E0 (F-test, k = 10, 200 hoán vị) cho p_perm = (b + 1)/(B + 1) ≤ 0,05 — đọc E0/<bộ>/e0_signal_check.csv.
* G2: AUROC trung bình của B0 trừ AUROC trung bình của random-k (100 lần rút mỗi fold) ở k = 10, trên đủ 25 fold, ≥ 0,05 —
  đọc <E1 dir>/<bộ>/classification_per_fold.csv (run_e1_classification.py --k 10, không có panel nào).
* AUPRC của B0 (trung bình 25 fold) chỉ để tham khảo, từ e1_scores.csv (--save-scores); mức ngẫu nhiên = tỉ lệ dương.

TCGA-LUAD trượt G1 hoặc G2 → bước tiếp theo là nhãn TP53 với cùng thủ tục; GSE31210 trượt → chỉ báo E0 và cổng.

    python scripts/b7_gate.py --e1-dir E1_classification_b7

Đầu ra: results/survey_v2/B7_gate/{b7_gate_table.csv, config.json}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402

N_FOLDS_EXPECTED = len(C.REPEAT_SEEDS) * C.N_FOLDS
NEXT_IF_FAIL = {"tcga_luad_egfr": "build TP53 labels on TCGA-LUAD and apply the same gate (QĐ177)",
                "gse31210_egfr": "report E0 and the gate only (no E1)"}


def gate_row(name: str, e0: pd.DataFrame, per_fold: pd.DataFrame, scores: pd.DataFrame | None, k: int = C.B7_GATE_K) -> dict:
    """Một dòng của bảng cổng cho bộ `name` (hàm thuần để test)."""
    e = e0[e0.cohort == name]
    if len(e) != 1 or int(e.k.iloc[0]) != k:
        raise ValueError(f"{name}: E0 phải có đúng một dòng ở k = {k}")
    e = e.iloc[0]
    pf = per_fold[(per_fold.cohort == name) & (per_fold.model == "lr_l2")]
    b0 = pf[pf.method == "b0"]
    rk = pf[(pf.method == "random_k") & (pf.k == k)]
    for what, t in (("B0", b0), ("random-k", rk)):
        n = len(t[["repeat", "fold"]].drop_duplicates())
        if n != N_FOLDS_EXPECTED or len(t) != N_FOLDS_EXPECTED:
            raise ValueError(f"{name}: {what} có {len(t)} dòng / {n} fold, cần đúng {N_FOLDS_EXPECTED}")
    p_perm = float(e.p_perm)
    b0_mean, rk_mean = float(b0.auroc.mean()), float(rk.auroc.mean())
    g1, g2 = p_perm <= C.B7_G1_P_PERM_MAX, (b0_mean - rk_mean) >= C.B7_G2_B0_MINUS_RANDOM_MIN
    row = {"cohort": name, "n": int(e.n), "p": int(e.p), "k": k, "e0_auroc_real": float(e.auroc_real), "e0_null_mean": float(e.null_mean),
           "p_perm": p_perm, "n_perm": int(e.n_perm), "G1_threshold": C.B7_G1_P_PERM_MAX, "G1_pass": bool(g1),
           "b0_auroc_mean": b0_mean, "b0_auroc_sd": float(b0.auroc.std(ddof=1)), "random_k_auroc_mean": rk_mean,
           "random_k_auroc_sd": float(rk.auroc.std(ddof=1)), "b0_minus_random_k": b0_mean - rk_mean,
           "G2_threshold": C.B7_G2_B0_MINUS_RANDOM_MIN, "G2_pass": bool(g2), "n_folds": N_FOLDS_EXPECTED,
           "random_k_draws_per_fold": float(rk.n_draws.iloc[0]) if "n_draws" in rk else np.nan,
           "gate_pass": bool(g1 and g2), "next_step": "E1 (and E4 for TCGA-LUAD) per QĐ177" if g1 and g2 else NEXT_IF_FAIL.get(name, "report only")}
    if scores is not None and len(scores):
        s = scores[(scores.cohort == name) & (scores.method == "b0")]
        ap = [average_precision_score(g.label, g.score) for _, g in s.groupby(["repeat", "fold"]) if g.label.nunique() == 2]
        first = s[s.repeat == s.repeat.min()]                      # mỗi mẫu đúng một lần trong một lần lặp
        row.update({"b0_auprc_mean": float(np.mean(ap)) if ap else np.nan, "b0_auprc_n_folds": len(ap),
                    "positive_rate": float(first.label.mean()) if len(first) else np.nan})
    return row


def main(argv=None, script=__file__):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.B7_DATASETS))
    ap.add_argument("--e0-dir", default="E0_b7")
    ap.add_argument("--e1-dir", default="E1_classification_b7")
    ap.add_argument("--out", default="B7_gate")
    args = ap.parse_args(argv)
    prov = start_run(script, params=vars(args))
    base = DATA_ROOT.joinpath(*C.RESULT_DIR)
    rows, inputs = [], []
    for name in args.datasets:
        e0_path = base / args.e0_dir / name / "e0_signal_check.csv"
        pf_path = base / args.e1_dir / name / "classification_per_fold.csv"
        sc_path = base / args.e1_dir / name / "e1_scores.csv"
        e0, pf = pd.read_csv(e0_path), pd.read_csv(pf_path)
        sc = pd.read_csv(sc_path) if sc_path.exists() else None
        rows.append(gate_row(name, e0, pf, sc))
        inputs += [p for p in (e0_path, pf_path, sc_path) if p.exists()]
        r = rows[-1]
        print(f"[{name}] G1: p_perm = {r['p_perm']:.4f} ({'đạt' if r['G1_pass'] else 'TRƯỢT'}); G2: B0 − random-k = "
              f"{r['b0_auroc_mean']:.3f} − {r['random_k_auroc_mean']:.3f} = {r['b0_minus_random_k']:.3f} ({'đạt' if r['G2_pass'] else 'TRƯỢT'}) "
              f"→ {r['next_step']}", flush=True)
    out = results_dir(args.out)
    table = pd.DataFrame(rows)
    table.to_csv(out / "b7_gate_table.csv", index=False)
    write_config(out, prov, {"experiment": "B7_gate", "rule": "G1: E0 p_perm <= 0.05 (F-test, k = 10, 200 permutations); "
                             "G2: mean AUROC(B0) - mean AUROC(random-k, k = 10, 100 draws/fold) over 25 folds >= 0.05 (fixed 12/11)",
                             "datasets": args.datasets, "e0_dir": args.e0_dir, "e1_dir": args.e1_dir,
                             "auprc_note": "B0 AUPRC from e1_scores.csv, reference only; random-k AUPRC not stored by run_e1_classification.py",
                             "n_rows": len(table), "versions": all_versions()}, data_files=inputs)
    print("wrote:", out / "b7_gate_table.csv", flush=True)


if __name__ == "__main__":
    main()
