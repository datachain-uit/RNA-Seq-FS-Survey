"""Y11 — Hợp nhất kết quả P0 (hoán vị + random-k) thành một bảng quyết định có cờ "bền" (robust).

Đầu vào: một hoặc nhiều summary.csv của perm_test_v3 (bản mới: results/survey_v2/P0/perm_test_v3/<bộ>/summary.csv
hoặc all_summary.csv; bản cũ: all_summary_permutation.csv + all_summary_random_k.csv chạy tách hai notebook).
Mỗi (dataset, class_weight, pipeline, k) lấy cột hoán vị từ dòng có n_perm lớn nhất và cột random-k từ dòng có
n_random lớn nhất; kiểm tra bacc_real khớp giữa hai nguồn (cùng nhãn thật, cùng fold). Sau đó:

* Holm (1979) riêng trong từng họ (dataset, task, class_weight, loại kiểm định) — họ đầy đủ = 3 pipeline × |k|.
* Quy tắc đặt trước (Review 20/10):
    signal  = p_perm_holm < 0,05
    useful  = signal và p_random < 0,05 và gain_vs_random ≥ 0,02   (p_random gốc, vì độ phân giải 1/(R+1))
    robust  = (một pipeline có useful ở hai mức k LIỀN KỀ) hoặc (một mức k có ≥ 2 pipeline useful)
* Verdict theo (dataset, class_weight): "KHONG co bang chung" / "Co tin hieu, chon gen KHONG hon random-k" /
  "Dat (co ich)" / "Dat ben".

    python scripts/aggregate_p0_decision.py --summaries D:/KLTN/results/survey_v2/P0/perm_test_v3/*/summary.csv
    python scripts/aggregate_p0_decision.py --summaries all_summary_permutation.csv all_summary_random_k.csv --out-dir P0_old

Đầu ra (mặc định results/survey_v2/P0/): p0_summary_merged.csv (mọi dòng, p đã hiệu chỉnh, cờ), p0_decision.csv
(một dòng mỗi dataset × class_weight), p0_decision.md (bảng Markdown để dán vào biên bản), config.json (provenance).
Khi gọi từ test: dùng hàm merge_summaries() và decide().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2.stats import holm  # noqa: E402

ALPHA = 0.05
MIN_GAIN = 0.02
ALL_PIPELINES = ("ftest_lr", "svm_rfe", "pam")
KEY = ["dataset", "branch", "task", "class_weight", "analysis", "pipeline", "k"]
PERM_COLS = ["bacc_real", "bacc_real_sd_folds", "bacc_real_val", "chance", "n_perm", "null_mean", "null_sd", "null_q95",
             "p_perm", "gain_vs_null"]
RAND_COLS = ["n_random", "random_mean", "random_q95", "p_random", "gain_vs_random"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", nargs="+", required=True, help="các summary.csv / all_summary*.csv")
    ap.add_argument("--out-dir", default="P0", help="thư mục con trong results/survey_v2/ (hoặc đường dẫn tuyệt đối)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--min-gain", type=float, default=MIN_GAIN)
    ap.add_argument("--no-provenance", action="store_true", help="chỉ để chạy thử ngoài kho (không ghi config.json)")
    return ap.parse_args(argv)


def merge_summaries(frames, tol: float = 1e-9) -> tuple[pd.DataFrame, list[str]]:
    """Ghép các summary; trả về (bảng hợp nhất theo KEY, danh sách cảnh báo)."""
    df = pd.concat([f.copy() for f in frames], ignore_index=True)
    for c in ("n_perm", "n_random"):
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)
    for c in PERM_COLS + RAND_COLS:
        if c not in df.columns:
            df[c] = np.nan
    warnings = []
    rows = []
    for key, g in df.groupby(KEY, sort=True, dropna=False):
        perm = g.sort_values("n_perm", ascending=False).iloc[0]
        rand = g.sort_values("n_random", ascending=False).iloc[0]
        if perm["n_perm"] > 0 and rand["n_random"] > 0 and not np.isclose(perm["bacc_real"], rand["bacc_real"], atol=tol):
            warnings.append(f"{key}: bacc_real khác giữa nguồn hoán vị ({perm['bacc_real']:.6f}) và random-k "
                            f"({rand['bacc_real']:.6f}) -> hai lượt không cùng nhãn thật/fold")
        row = dict(zip(KEY, key))
        row.update({c: perm[c] for c in PERM_COLS})
        row.update({c: rand[c] for c in RAND_COLS})
        if perm["n_perm"] == 0:
            for c in ("null_mean", "null_sd", "null_q95", "p_perm", "gain_vs_null"):
                row[c] = np.nan
        if rand["n_random"] == 0:
            for c in ("random_mean", "random_q95", "p_random", "gain_vs_random"):
                row[c] = np.nan
        # commit của từng nguồn (để truy vết)
        row["commit_perm"] = perm.get("commit_hash", np.nan)
        row["commit_random"] = rand.get("commit_hash", np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["k"] = out["k"].astype(int)
    return out, warnings


def adjust_and_flag(merged: pd.DataFrame, alpha: float = ALPHA, min_gain: float = MIN_GAIN) -> pd.DataFrame:
    """Holm trong từng họ + cờ signal / beats_random / useful. Họ đầy đủ = 3 pipeline × |k| của bộ."""
    df = merged.copy()
    df["p_perm_holm"] = np.nan
    df["p_random_holm"] = np.nan
    df["family_size"] = 0
    df["family_complete"] = False
    for (ds, task, cw), g in df.groupby(["dataset", "task", "class_weight"]):
        n_k = g["k"].nunique()
        expected = len(ALL_PIPELINES) * n_k
        for pcol in ("p_perm", "p_random"):
            idx = g.index[g[pcol].notna()]
            if len(idx):
                df.loc[idx, f"{pcol}_holm"] = holm(df.loc[idx, pcol].to_numpy())
        idx = g.index[g["p_perm"].notna()]
        df.loc[g.index, "family_size"] = len(idx)
        df.loc[g.index, "family_complete"] = (len(idx) == expected) and (set(g.loc[idx, "pipeline"]) == set(ALL_PIPELINES))
    df["signal"] = df["p_perm_holm"] < alpha
    df["beats_random"] = df["p_random"] < alpha
    df["useful"] = df["signal"] & df["beats_random"] & (df["gain_vs_random"] >= min_gain)
    return df


def robust_evidence(g: pd.DataFrame) -> tuple[bool, str]:
    """g: các dòng của một (dataset, class_weight). Bền nếu useful ở hai k liền kề của một pipeline, hoặc ≥ 2 pipeline tại một k."""
    ks = sorted(g["k"].unique())
    reasons = []
    for p, gp in g[g["useful"]].groupby("pipeline"):
        got = sorted(gp["k"].unique())
        for a, b in zip(ks, ks[1:]):
            if a in got and b in got:
                reasons.append(f"{p}: k={a},{b}")
    for k, gk in g[g["useful"]].groupby("k"):
        if gk["pipeline"].nunique() >= 2:
            reasons.append(f"k={k}: {','.join(sorted(gk['pipeline']))}")
    return bool(reasons), "; ".join(reasons)


def decide(flagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, br, task, cw, an), g in flagged.groupby(["dataset", "branch", "task", "class_weight", "analysis"], dropna=False):
        robust, why = robust_evidence(g)
        n_signal, n_useful = int(g["signal"].sum()), int(g["useful"].sum())
        has_perm = g["p_perm"].notna().any()
        if not has_perm:
            verdict = "SANITY OK: BAcc nhan that >= 0.9" if g["bacc_real"].max() >= 0.9 else "SANITY CAN XEM LAI: BAcc < 0.9"
        elif robust:
            verdict = "Dat ben (co ich o >= 2 k lien ke hoac >= 2 pipeline)"
        elif n_useful > 0:
            verdict = "Dat theo quy tac dat truoc (co ich) nhung KHONG ben"
        elif n_signal > 0:
            verdict = "Co tin hieu nhung chon gen KHONG hon random-k"
        else:
            verdict = "KHONG co bang chung ve tin hieu"
        useful_list = "; ".join(f"{r.pipeline}@k={int(r.k)} (gain {r.gain_vs_random:+.3f})"
                                for r in g[g["useful"]].sort_values(["pipeline", "k"]).itertuples())
        gains = g.loc[g["useful"], "gain_vs_random"]
        rows.append({"dataset": ds, "branch": br, "task": task, "class_weight": cw, "analysis": an,
                     "n_tests": int(len(g)), "family_complete": bool(g["family_complete"].all()),
                     "n_perm": int(g["n_perm"].max()), "n_random": int(g["n_random"].max()),
                     "best_bacc": float(g["bacc_real"].max()),
                     "best_bacc_at": f"{g.loc[g['bacc_real'].idxmax(), 'pipeline']}@k={int(g.loc[g['bacc_real'].idxmax(), 'k'])}",
                     "min_p_perm_holm": float(g["p_perm_holm"].min()) if has_perm else np.nan,
                     "min_p_random": float(g["p_random"].min()) if g["p_random"].notna().any() else np.nan,
                     "n_signal": n_signal, "n_beats_random": int(g["beats_random"].sum()), "n_useful": n_useful,
                     "gain_useful_min": float(gains.min()) if len(gains) else np.nan,
                     "gain_useful_max": float(gains.max()) if len(gains) else np.nan,
                     "robust": robust, "robust_evidence": why, "useful_tests": useful_list, "verdict": verdict})
    return pd.DataFrame(rows)


def to_markdown(decision: pd.DataFrame) -> str:
    cols = ["dataset", "class_weight", "n_tests", "best_bacc", "n_signal", "n_useful", "gain_useful_min", "gain_useful_max",
            "robust", "verdict"]
    d = decision[cols].copy()
    for c in ("best_bacc", "gain_useful_min", "gain_useful_max"):
        d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main(argv=None, script=__file__) -> None:
    args = parse_args(argv)
    frames = [pd.read_csv(f) for f in args.summaries]
    merged, warns = merge_summaries(frames)
    for w in warns:
        print("CẢNH BÁO:", w, file=sys.stderr)
    flagged = adjust_and_flag(merged, args.alpha, args.min_gain)
    decision = decide(flagged)
    out_dir = Path(args.out_dir)
    if not args.no_provenance:
        from survey_v2.runlog import results_dir, start_run, write_config
        prov = start_run(script, params=vars(args))
        if not out_dir.is_absolute():
            out_dir = results_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(out_dir / "p0_summary_merged.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(out_dir / "p0_decision.csv", index=False, encoding="utf-8-sig")
    (out_dir / "p0_decision.md").write_text(to_markdown(decision), encoding="utf-8")
    if not args.no_provenance:
        commits = sorted({str(c) for c in pd.concat([merged["commit_perm"], merged["commit_random"]]).dropna().unique()})
        write_config(out_dir, prov, {"experiment": "P0_decision", "inputs": [str(Path(f).resolve()) for f in args.summaries],
                                     "input_commit_hashes": commits, "alpha": args.alpha, "min_gain": args.min_gain,
                                     "rules": {"signal": "p_perm_holm < alpha", "useful": "signal & p_random < alpha & gain >= min_gain",
                                               "robust": "useful at two adjacent k of one pipeline, or >= 2 pipelines at one k"},
                                     "warnings": warns, "citable": all(c not in ("unknown", "None", "nan") for c in commits) and bool(commits)},
                     data_files=[Path(f) for f in args.summaries])
    print(to_markdown(decision))
    print("ghi:", out_dir)


if __name__ == "__main__":
    main()
