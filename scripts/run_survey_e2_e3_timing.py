"""E2 (chi phí tại một điểm) và E3 (đường cong độ phức tạp theo n, p; wrapper theo số lần đánh giá).

Chạy trên MỘT máy cố định, BLAS 1 luồng (script tự đặt OMP/OPENBLAS/MKL_NUM_THREADS = 1 trước khi import
numpy). Dữ liệu: TCGA-LUAD giai đoạn (dải p rộng nhất); phần train của repeat 0, fold 0 của CV 5×5.

E2: n = toàn bộ phần train, p = 5.000 (lọc phương sai), k = 50; wall time, CPU time, bộ nhớ đỉnh; trung vị 3 lần.
E3: theo n ∈ {100, 200, 300, toàn bộ} (p = 5.000) và theo p ∈ {1.000, 2.000, 5.000, 10.000, toàn bộ} (n toàn bộ);
    số mũ log–log của thời gian theo n và theo p (so với Bảng 5 của bản v1); wrapper theo (quần thể × thế hệ)
    ∈ {10×20, 30×50} (--wrappers).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_survey_e2_e3_timing.py
    python scripts/run_survey_e2_e3_timing.py --methods f_test mrmr --repeats 1      # chạy thử nhanh
    python scripts/run_survey_e2_e3_timing.py --wrappers                              # thêm GA, BPSO theo N_eval

Đầu ra: results/survey_v2/E2_E3/{e2_cost_point.csv, e3_scaling_n.csv, e3_scaling_p.csv, e3_wrapper_evals.csv,
e3_exponents.csv, machine_config.json, config.json}.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, repeated_patient_splits  # noqa: E402
from survey_v2.methods import select  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.timing import empirical_exponent, machine_info, measure, subsample_rows, top_variance_columns  # noqa: E402

NON_WRAPPER = list(C.FILTER_METHODS + C.EMBEDDED_METHODS)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=C.TIMING_DATASET)
    ap.add_argument("--methods", nargs="+", default=NON_WRAPPER)
    ap.add_argument("--k", type=int, default=C.TIMING_K)
    ap.add_argument("--repeats", type=int, default=C.TIMING_REPEATS, help="số lần đo, lấy trung vị")
    ap.add_argument("--n-grid", nargs="+", default=list(C.E3_N_GRID), help="dùng 'all' cho toàn bộ phần train")
    ap.add_argument("--p-grid", nargs="+", default=list(C.E3_P_GRID), help="dùng 'all' cho toàn bộ gen")
    ap.add_argument("--wrappers", action="store_true", help="đo GA, BPSO theo ngân sách (quần thể × thế hệ)")
    ap.add_argument("--skip-e3", action="store_true")
    ap.add_argument("--out", default="E2_E3")
    return ap.parse_args()


def _grid(values):
    out = []
    for v in values:
        out.append(None if v in (None, "all", "None") else int(v))
    return out


def selector(method: str, seed: int, pop=None, generations=None):
    def run(X, y, k):
        kw = {"pop": pop, "generations": generations} if pop else None
        return select(method, X, y, [k], seed=seed, n_jobs=1, wrapper_kwargs=kw).panels[k]
    return run


def main() -> None:
    args = parse_args()
    prov = start_run(__file__, params=vars(args))
    out = results_dir(args.out)
    ds = load_dataset(DATA_ROOT, args.dataset)
    repeat, seed, fold, tr, _ = repeated_patient_splits(ds.y, ds.groups)[0]
    Xtr, ytr = ds.X[tr], ds.y[tr]
    rng = np.random.default_rng([7, seed, fold])
    minfo = machine_info()
    (out / "machine_config.json").write_text(json.dumps(minfo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Máy:", minfo.get("cpu_model", minfo.get("processor")), "| RAM GB:", minfo.get("ram_gb"), flush=True)
    t_all = time.time()
    with threadpool_limits(1):
        # ---- E2: điểm cố định ---------------------------------------------------------------------
        p_fixed = min(C.PREFILTER_P0, Xtr.shape[1])
        cols = top_variance_columns(Xtr, p_fixed)
        X2 = Xtr[:, cols]
        rows = []
        for m in args.methods:
            print(f"E2 {m} ...", end=" ", flush=True)
            r = measure(selector(m, seed), X2, ytr, args.k, repeats=args.repeats)
            rows.append({"method": m, "family": C.METHOD_FAMILY[m], "n": X2.shape[0], "p": X2.shape[1], "k": args.k, **r})
            print(f"wall {r['wall_s']:.2f}s cpu {r['cpu_s']:.2f}s peak {r['peak_mb']:.0f}MB", flush=True)
        if args.wrappers:
            for m in C.WRAPPER_METHODS:
                pop, gen = C.WRAPPER_POP, C.WRAPPER_GENERATIONS
                print(f"E2 {m} ({pop}x{gen}) ...", end=" ", flush=True)
                r = measure(selector(m, seed, pop, gen), X2, ytr, args.k, repeats=args.repeats)
                rows.append({"method": m, "family": "wrapper", "n": X2.shape[0], "p": X2.shape[1], "k": args.k,
                             "pop": pop, "generations": gen, **r})
                print(f"wall {r['wall_s']:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "e2_cost_point.csv", index=False, encoding="utf-8-sig")

        if not args.skip_e3:
            # ---- E3 theo n (p = 5.000) ---------------------------------------------------------------
            rows_n = []
            for n in _grid(args.n_grid):
                idx = subsample_rows(ytr, n, rng)
                for m in args.methods:
                    r = measure(selector(m, seed), X2[idx], ytr[idx], args.k, repeats=args.repeats)
                    rows_n.append({"method": m, "family": C.METHOD_FAMILY[m], "n": len(idx), "p": X2.shape[1], "k": args.k, **r})
                    print(f"E3 n={len(idx):4d} {m:14s} wall {r['wall_s']:.2f}s", flush=True)
            pd.DataFrame(rows_n).to_csv(out / "e3_scaling_n.csv", index=False, encoding="utf-8-sig")
            # ---- E3 theo p (n toàn bộ) ---------------------------------------------------------------
            rows_p = []
            for p in _grid(args.p_grid):
                cols_p = top_variance_columns(Xtr, p)
                Xp = Xtr[:, cols_p]
                for m in args.methods:
                    r = measure(selector(m, seed), Xp, ytr, args.k, repeats=args.repeats)
                    rows_p.append({"method": m, "family": C.METHOD_FAMILY[m], "n": Xp.shape[0], "p": Xp.shape[1], "k": args.k, **r})
                    print(f"E3 p={Xp.shape[1]:6d} {m:14s} wall {r['wall_s']:.2f}s", flush=True)
            pd.DataFrame(rows_p).to_csv(out / "e3_scaling_p.csv", index=False, encoding="utf-8-sig")
            # ---- E3 wrapper theo số lần đánh giá ------------------------------------------------------
            rows_w = []
            if args.wrappers:
                for m in C.WRAPPER_METHODS:
                    for pop, gen in C.E3_WRAPPER_BUDGETS:
                        r = measure(selector(m, seed, pop, gen), X2, ytr, args.k, repeats=args.repeats)
                        rows_w.append({"method": m, "pop": pop, "generations": gen, "n_eval_nominal": pop * gen,
                                       "n": X2.shape[0], "p": X2.shape[1], "k": args.k, **r})
                        print(f"E3 {m} {pop}x{gen} wall {r['wall_s']:.1f}s", flush=True)
            pd.DataFrame(rows_w).to_csv(out / "e3_wrapper_evals.csv", index=False, encoding="utf-8-sig")
            # ---- số mũ log–log ---------------------------------------------------------------------------
            exps = []
            dn, dp = pd.DataFrame(rows_n), pd.DataFrame(rows_p)
            for m in args.methods:
                gn, gp = dn[dn.method == m], dp[dp.method == m]
                gp_large = gp[gp.p >= 2000]  # dải lớn: tránh chi phí cố định làm số mũ thấp (code I)
                exps.append({"method": m, "family": C.METHOD_FAMILY[m],
                             "alpha_n": empirical_exponent(gn, "n"), "n_range": f"{gn.n.min()}-{gn.n.max()}",
                             "alpha_p": empirical_exponent(gp, "p"), "alpha_p_large": empirical_exponent(gp_large, "p"),
                             "p_range": f"{gp.p.min()}-{gp.p.max()}",
                             "wall_spread_max_rel": float(max((max(w) - min(w)) / max(np.median(w), 1e-9)
                                                              for w in gn.wall_s_all.tolist() + gp.wall_s_all.tolist()))})
            pd.DataFrame(exps).to_csv(out / "e3_exponents.csv", index=False, encoding="utf-8-sig")
    cfg = {"experiment": "E2_E3", "dataset": args.dataset, "split": {"repeat": repeat, "seed": seed, "fold": fold,
           "n_train": int(len(tr))}, "k": args.k, "p_fixed": int(p_fixed), "repeats": args.repeats,
           "n_grid": args.n_grid, "p_grid": args.p_grid, "methods": args.methods, "wrappers": args.wrappers,
           "blas_threads": 1, "machine": minfo, "versions": all_versions(), "elapsed_s": time.time() - t_all,
           "note": "thời gian chỉ gồm bước chọn gen (không gồm huấn luyện bộ phân loại sau chọn)"}
    write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv"])
    print("xong:", out, flush=True)


if __name__ == "__main__":
    main()
