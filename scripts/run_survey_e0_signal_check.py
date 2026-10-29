"""E0 — kiểm tra tín hiệu của từng cohort (bước 1 của khung quyết định, survey v3): hoán vị nhãn + random-k.

Pipeline kiểm tra: lọc phương sai (không nhãn) trong fold → E0_SELECTOR (F-test) lấy k = E0_K gen → LR-L2 (C = 1,
balanced) → AUROC trung bình CV 5 fold (StratifiedKFold, seed 42; chia theo bệnh nhân khi có nhiều mẫu/bệnh nhân).
* Hoán vị: E0_N_PERM lần xáo nhãn (seed cố định), cùng pipeline → p_perm = (#null ≥ thật + 1)/(B + 1) (Phipson & Smyth 2010).
* Random-k: E0_N_RANDOM panel k gen ngẫu nhiên trong vùng lọc (seed [cohort, 0, fold, k, draw]) → p_random cùng công thức,
  gain = AUROC thật − trung bình random-k.
Quy tắc đặt trước (từ P0, Review 11/10): có tín hiệu nếu p_perm < 0,05; chọn gen có ích nếu p_random < 0,05 và gain ≥ 0,02.
Trên bài toán LUAD vs LUSC đây là thủ tục (kỳ vọng đạt ở mọi cohort); nó ghi lại bằng chứng cho Mục 4.4 và Fig. 9.

    python scripts/run_survey_e0_signal_check.py
    python scripts/run_survey_e0_signal_check.py --datasets gse37745 --n-perm 50 --n-random 20

Đầu ra: results/survey_v2/E0/<bộ>/{e0_signal_check.csv, e0_perm_scores.csv, e0_random_k_scores.csv, config.json}.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, variance_prefilter  # noqa: E402
from survey_v2.methods import random_panel, select  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--k", type=int, default=C.E0_K)
    ap.add_argument("--selector", default=C.E0_SELECTOR)
    ap.add_argument("--n-perm", type=int, default=C.E0_N_PERM)
    ap.add_argument("--n-random", type=int, default=C.E0_N_RANDOM)
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--seed", type=int, default=C.REPEAT_SEEDS[0])
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="E0")
    return ap.parse_args(argv)


def _folds(y, groups, seed):
    one = len(np.unique(groups)) == len(groups)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed) if one else StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, None if one else groups))


def _auc_lr(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=seed)
    m.fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1]))


def cv_auc(X, y, folds, k, selector, prefilter, seed, fixed_cols=None):
    """AUROC trung bình qua fold; chọn gen trong fold (hoặc dùng cột cố định fixed_cols trong vùng lọc)."""
    aucs = []
    for f, (tr, te) in enumerate(folds):
        pre = variance_prefilter(X[tr], prefilter)
        if fixed_cols is None:
            sel = select(selector, X[tr][:, pre], y[tr], [k], seed=seed)
            cols = pre[np.asarray(sel.panels[k])]
        else:
            cols = pre[fixed_cols[f]]
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            aucs.append(np.nan)
            continue
        aucs.append(_auc_lr(X[tr][:, cols], y[tr], X[te][:, cols], y[te], seed))
    return float(np.nanmean(aucs)), aucs


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params=vars(args))
    t_all = time.time()
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        cid = C.cohort_id(name)
        folds = _folds(ds.y, ds.groups, args.seed)
        with threadpool_limits(1):
            real, real_folds = cv_auc(ds.X, ds.y, folds, args.k, args.selector, args.prefilter, args.seed)

            def perm(i):
                with threadpool_limits(1):
                    rng = np.random.default_rng([cid, 1, i])
                    y_perm = rng.permutation(ds.y)
                    # chạy lại TOÀN BỘ pipeline trên nhãn đã xáo, kể cả chia fold phân tầng theo nhãn xáo
                    # (để mỗi fold test đủ hai lớp; Ojala & Garriga 2010)
                    folds_perm = _folds(y_perm, ds.groups, args.seed)
                    return cv_auc(ds.X, y_perm, folds_perm, args.k, args.selector, args.prefilter, args.seed)[0]

            def rand(d):
                with threadpool_limits(1):
                    cols = [random_panel(min(args.prefilter, ds.p), args.k, np.random.default_rng([cid, 0, f, args.k, d]))
                            for f in range(len(folds))]
                    return cv_auc(ds.X, ds.y, folds, args.k, args.selector, args.prefilter, args.seed, fixed_cols=cols)[0]

            null = list(Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(perm)(i) for i in range(args.n_perm)))
            rnd = list(Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(rand)(d) for d in range(args.n_random)))
        null, rnd = np.asarray(null), np.asarray(rnd)
        p_perm = (np.sum(null >= real) + 1) / (len(null) + 1)
        p_rand = (np.sum(rnd >= real) + 1) / (len(rnd) + 1)
        gain = real - float(rnd.mean())
        row = {"cohort": name, "role": C.DATASETS.get(name), "n": ds.n, "p": ds.p, "k": args.k, "selector": args.selector,
               "auroc_real": real, "auroc_real_folds": ";".join(f"{a:.4f}" for a in real_folds),
               "n_perm": len(null), "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)), "p_perm": float(p_perm),
               "n_random": len(rnd), "random_mean": float(rnd.mean()), "random_q95": float(np.quantile(rnd, 0.95)), "p_random": float(p_rand),
               "gain_vs_random": gain, "signal_present": bool(p_perm < C.ALPHA),
               "selection_useful": bool(p_rand < C.ALPHA and gain >= 0.02), "min_p": 1.0 / (len(null) + 1)}
        pd.DataFrame([row]).to_csv(out / "e0_signal_check.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"perm": np.arange(len(null)), "auroc": null}).to_csv(out / "e0_perm_scores.csv", index=False)
        pd.DataFrame({"draw": np.arange(len(rnd)), "auroc": rnd}).to_csv(out / "e0_random_k_scores.csv", index=False)
        print(f"[{name}] AUROC {real:.3f} | p_perm {p_perm:.4f} (null {null.mean():.3f}) | p_random {p_rand:.4f} "
              f"(random {rnd.mean():.3f}, gain {gain:+.3f}) | signal {row['signal_present']} useful {row['selection_useful']}", flush=True)
        write_config(out, prov, {"experiment": "E0_signal_check", "dataset": name, **{k: v for k, v in row.items() if k != "auroc_real_folds"},
                                 "rule": "signal if p_perm < 0.05; selection useful if p_random < 0.05 and gain >= 0.02 (P0 rule, Review 11/10)",
                                 "p_formula": "(b + 1) / (B + 1), Phipson & Smyth 2010", "cv": "5-fold, seed 42, patient-grouped if needed",
                                 "gene_space": ds.notes, "elapsed_s": time.time() - t_all, "versions": all_versions()},
                     data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv"])


if __name__ == "__main__":
    main()
