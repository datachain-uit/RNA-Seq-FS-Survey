"""P0 — Permutation test v3 (ba pipeline, hoán vị theo bệnh nhân + random-k) CHẠY TỪ KHO có provenance.

Bản chuyển từ notebooks/phase_0/04_analysis/perm_test_v3_{permutation,random_k}.ipynb sang script để
kết quả P0 có commit hash (kết quả nhận ngày 24/10/2026 ghi commit_hash = "unknown", không trích được).
Thiết kế, seed và quy tắc quyết định GIỮ NGUYÊN so với notebook (so sánh được với bản cũ):

* Dữ liệu: $KLTN_DATA_ROOT/experiment-data-perm/<bộ>/ (expression.h5, metadata.csv, perm_folds.csv,
  folds/fold_<r>/{train,val,test}.h5 key "data"); fit trên train, p-value trên test, val chỉ ghi lại.
* Ba pipeline (chọn gen TRONG training fold, chạy lại ở mọi hoán vị): P1 F-test + LR-L2 (C = 1);
  P2 SVM-RFE (loại một nửa mỗi vòng) + SVM tuyến tính (C = 1); P3 PAM / nearest shrunken centroids.
* class_weight: balanced (chính); none (sensitivity, tùy chọn --class-weights balanced none).
* Hoán vị: xáo nhãn theo bệnh nhân, seed default_rng([42, pid]); random-k: gen theo default_rng([42, 7, rep]).
  p = (#null ≥ thật + 1)/(B + 1) (Phipson & Smyth 2010). Holm riêng trong từng họ (bộ, tác vụ, class_weight,
  loại kiểm định) gồm 3 pipeline × |k|.
* Quyết định đặt trước: có tín hiệu nếu p_perm_holm < 0,05; chọn gen có ích nếu thêm p_random < 0,05 và
  gain_vs_random ≥ 0,02. (Mức "bền": scripts/aggregate_p0_decision.py.)

    python scripts/run_p0_perm_v3.py                                   # 3 bộ giai đoạn, balanced, 1000 hoán vị, 1000 random-k
    python scripts/run_p0_perm_v3.py --datasets gse68465_stage_early_late --n-perm 100 --n-random 100 --n-jobs 8
    python scripts/run_p0_perm_v3.py --class-weights balanced none      # thêm sensitivity như notebook gốc
    python scripts/run_p0_perm_v3.py --datasets tcga_normal_vs_tumor_sanity --n-perm 0 --n-random 0   # nhánh sanity

Đầu ra: results/survey_v2/P0/perm_test_v3/<bộ>/{fold_assignment.csv, real_scores_per_fold.csv, real_confusion.csv,
real_selected_genes.csv, null_scores_per_fold.csv, random_k_scores_per_fold.csv, timing.csv, summary.csv,
pvalues_adjusted.csv, decision.csv, config.json}. Chạy lại tiếp tục từ phần đã có (resume); --resume-dirs nhận thư mục
kết quả cũ (chứa <bộ>/...) để gộp.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.stats import holm  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
LR_C = 1.0
SVM_C = 1.0
ALPHA = 0.05
MIN_GAIN = 0.02
N_SPLITS = 5
MAIN_CW = "balanced"
ALL_PIPELINES = ["ftest_lr", "svm_rfe", "pam"]
CW_SKLEARN = {"balanced": "balanced", "none": None}
K_STAGE = [10, 50, 100, 200, 500]
K_HISTO = [1, 2, 3, 5, 10, 25, 50]
DATASET_SPECS = {
    "tcga_luad_stage_early_late": {"branch": "1_stage", "task": "early_vs_late", "k": K_STAGE},
    "tcga_lusc_stage_early_late": {"branch": "1_stage", "task": "early_vs_late", "k": K_STAGE},
    "gse68465_stage_early_late": {"branch": "1_stage", "task": "early_vs_late", "k": K_STAGE},
    "tcga_luad_lusc_primary_tumor": {"branch": "2_histology", "task": "luad_vs_lusc", "k": K_HISTO},
    "tcga_normal_vs_tumor_sanity": {"branch": "3_sanity", "task": "normal_vs_tumor", "k": [10]},
}
STAGE_DATASETS = ["gse68465_stage_early_late", "tcga_luad_stage_early_late", "tcga_lusc_stage_early_late"]

DTYPES = {"pid": int, "rep": int, "k": int, "fold": int, "true": int, "pred": int, "count": int, "rank": int,
          "bacc": float, "sec_per_run": float}
FOLD_KEYS = ["class_weight", "pipeline", "k", "fold"]
FILES = {  # tên file -> (khóa bỏ trùng, cột)
    "real": ("real_scores_per_fold.csv", FOLD_KEYS + ["split"], FOLD_KEYS + ["split", "bacc"]),
    "conf": ("real_confusion.csv", ["class_weight", "pipeline", "k", "true", "pred"],
             ["class_weight", "pipeline", "k", "true", "pred", "count"]),
    "gene": ("real_selected_genes.csv", FOLD_KEYS + ["rank"], FOLD_KEYS + ["rank", "gene"]),
    "null": ("null_scores_per_fold.csv", ["pid"] + FOLD_KEYS, ["pid"] + FOLD_KEYS + ["bacc"]),
    "rand": ("random_k_scores_per_fold.csv", ["rep"] + FOLD_KEYS, ["rep"] + FOLD_KEYS + ["bacc"]),
    "time": ("timing.csv", ["pipeline"], ["pipeline", "sec_per_run"]),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=STAGE_DATASETS, choices=list(DATASET_SPECS))
    ap.add_argument("--pipelines", nargs="+", default=ALL_PIPELINES, choices=ALL_PIPELINES)
    ap.add_argument("--class-weights", nargs="+", default=["balanced"], choices=["balanced", "none"])
    ap.add_argument("--k", nargs="+", type=int, default=None, help="chỉ để chạy thử; mặc định theo DATASET_SPECS")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--batch-per-worker", type=int, default=4)
    ap.add_argument("--time-budget-h", type=float, default=1e9, help="dừng an toàn sau lô đang chạy khi sắp hết")
    ap.add_argument("--resume-dirs", nargs="*", default=[], help="thư mục kết quả cũ chứa <bộ>/ để gộp")
    ap.add_argument("--out", default="P0/perm_test_v3")
    ap.add_argument("--exclude-diagnosis-mismatch", action="store_true", default=True)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------
# Ba pipeline (giữ nguyên notebook)
# ---------------------------------------------------------------------------------------------------
def ftest_rank(X, y):
    F, _ = f_classif(X, y)
    return np.argsort(-np.nan_to_num(F, nan=-np.inf), kind="stable")


def make_lr(cw):
    return LogisticRegression(C=LR_C, max_iter=5000, class_weight=CW_SKLEARN[cw])


class LinearSVM:
    """SVM tuyến tính 2 lớp qua kernel tính sẵn X·Xᵀ; giữ w để xếp hạng gen (nhanh khi n ≪ p)."""

    def __init__(self, cw):
        self.cw = cw

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        svc = SVC(kernel="precomputed", C=SVM_C, class_weight=CW_SKLEARN[self.cw]).fit(X @ X.T, y)
        self.w = svc.dual_coef_[0] @ X[svc.support_]
        self.b = svc.intercept_[0]
        return self

    def predict(self, X):
        return self.classes_[(X @ self.w + self.b > 0).astype(int)]


def svm_rfe_panels(X, y, k_list, cw):
    """Loại một nửa số gen mỗi vòng, dừng đúng tại từng k -> {k: (gen, SVM đã fit trên k gen)}."""
    targets = sorted(k_list, reverse=True)
    active = np.arange(X.shape[1])
    out = {}
    while True:
        m = LinearSVM(cw).fit(X[:, active], y)
        if len(active) in targets:
            out[len(active)] = (active.copy(), m)
        if len(active) <= targets[-1]:
            break
        n_next = max(len(active) // 2, max(t for t in targets if t < len(active)))
        active = active[np.argsort(-(m.w ** 2), kind="stable")[:n_next]]
    return out


class NSCStats:
    """Thống kê PAM của một fold; model(k, cw) = PAM với ngưỡng Δ giữ lại đúng k gen (Tibshirani et al. 2002)."""

    def __init__(self, X, y):
        self.classes_ = np.unique(y)
        idx = np.searchsorted(self.classes_, y)
        n, K = len(y), len(self.classes_)
        nk = np.bincount(idx)
        self.xbar = X.mean(0)
        self.mu = np.stack([X[idx == j].mean(0) for j in range(K)])
        self.s = np.sqrt(((X - self.mu[idx]) ** 2).sum(0) / max(n - K, 1))
        self.s0 = np.median(self.s)
        self.mk = np.sqrt(1.0 / nk - 1.0 / n)
        self.d = (self.mu - self.xbar) / (self.mk[:, None] * (self.s + self.s0))
        self.score = np.abs(self.d).max(0)
        self.order = np.argsort(-self.score, kind="stable")
        self.logprior = {"none": np.log(nk / n), "balanced": np.full(K, -np.log(K))}

    def model(self, k, cw, genes=None):
        if genes is None:
            sel = self.order[:k]
            delta = self.score[self.order[k]] if k < len(self.order) else 0.0
        else:
            sel, delta = np.asarray(genes), 0.0
        d = self.d[:, sel]
        d_sh = np.sign(d) * np.maximum(np.abs(d) - delta, 0.0)
        scale = self.s[sel] + self.s0
        return _NSCModel(self.classes_, sel, self.xbar[sel] + self.mk[:, None] * scale * d_sh, scale ** 2, self.logprior[cw])


class _NSCModel:
    def __init__(self, classes, sel, mu, scale2, logprior):
        self.classes_, self.sel, self.mu, self.scale2, self.logprior = classes, sel, mu, scale2, logprior

    def predict(self, X):
        Xs = X[:, self.sel]
        d = -(((Xs[:, None, :] - self.mu[None]) ** 2) / self.scale2).sum(2) + 2 * self.logprior
        return self.classes_[d.argmax(1)]


def fold_predictions(Xtr, ytr, Xte, pipeline, cws, k_list):
    """-> {(cw, k): (dự đoán trên test, chỉ số gen đã chọn)}."""
    out = {}
    if pipeline == "ftest_lr":
        order = ftest_rank(Xtr, ytr)
        for cw in cws:
            for k in k_list:
                g = order[:k]
                out[(cw, k)] = (make_lr(cw).fit(Xtr[:, g], ytr).predict(Xte[:, g]), g)
    elif pipeline == "svm_rfe":
        for cw in cws:
            for k, (g, m) in svm_rfe_panels(Xtr, ytr, k_list, cw).items():
                out[(cw, k)] = (m.predict(Xte[:, g]), g)
    elif pipeline == "pam":
        st = NSCStats(Xtr, ytr)
        for cw in cws:
            for k in k_list:
                out[(cw, k)] = (st.model(k, cw).predict(Xte), st.order[:k])
    else:
        raise ValueError(pipeline)
    return out


def random_predictions(Xtr, ytr, Xte, genes, pipeline, cw):
    a, b = Xtr[:, genes], Xte[:, genes]
    if pipeline == "ftest_lr":
        return make_lr(cw).fit(a, ytr).predict(b)
    if pipeline == "svm_rfe":
        return LinearSVM(cw).fit(a, ytr).predict(b)
    if pipeline == "pam":
        return NSCStats(a, ytr).model(None, cw, genes=np.arange(len(genes))).predict(b)
    raise ValueError(pipeline)


# ---------------------------------------------------------------------------------------------------
# Hoán vị / random-k trong tiến trình con
# ---------------------------------------------------------------------------------------------------
def permute_labels(y, groups, rng):
    s = pd.Series(y, index=groups)
    if (s.groupby(level=0).nunique() == 1).all():
        lab = s.groupby(level=0).first()
        mapping = dict(zip(lab.index, rng.permutation(lab.to_numpy())))
        return np.array([mapping[g] for g in groups])
    return rng.permutation(y)


def labels_for(pid, y, groups):
    return y if pid < 0 else permute_labels(y, groups, np.random.default_rng([SEED, pid]))


def run_pid(pid, pipelines, fold_data, folds, y, groups, k_list, cws):
    warnings.filterwarnings("ignore")
    yp = labels_for(pid, y, groups)
    rows = []
    with threadpool_limits(1):
        for f, ((Xtr, _, Xte), (tr, _, te)) in enumerate(zip(fold_data, folds), 1):
            for p in pipelines:
                for (cw, k), (pred, _) in fold_predictions(Xtr, yp[tr], Xte, p, cws, k_list).items():
                    rows.append({"pid": pid, "class_weight": cw, "pipeline": p, "k": k, "fold": f,
                                 "bacc": balanced_accuracy_score(yp[te], pred)})
    return rows


def run_random_rep(rep, pipelines, fold_data, folds, y, k_list, cws):
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng([SEED, 7, rep])
    rows = []
    with threadpool_limits(1):
        for f, ((Xtr, _, Xte), (tr, _, te)) in enumerate(zip(fold_data, folds), 1):
            for k in k_list:
                genes = rng.choice(Xtr.shape[1], size=k, replace=False)
                for p in pipelines:
                    for cw in cws:
                        pred = random_predictions(Xtr, y[tr], Xte, genes, p, cw)
                        rows.append({"rep": rep, "class_weight": cw, "pipeline": p, "k": k, "fold": f,
                                     "bacc": balanced_accuracy_score(y[te], pred)})
    return rows


def run_chunk(kind, items, fold_data, folds, y, groups, k_list, cws):
    if kind == "null":
        return [r for i, pipes in items for r in run_pid(i, pipes, fold_data, folds, y, groups, k_list, cws)]
    return [r for i, pipes in items for r in run_random_rep(i, pipes, fold_data, folds, y, k_list, cws)]


# ---------------------------------------------------------------------------------------------------
# Đọc/ghi kết quả, resume
# ---------------------------------------------------------------------------------------------------
def typed(d):
    return d.astype({c: t for c, t in DTYPES.items() if c in d.columns})


def load_prev(D, key, resume_dirs):
    name, keys, cols = FILES[key]
    path = D.result_dir / name
    srcs = [Path(r) / D.name / name for r in resume_dirs] + [path]
    frames = [pd.read_csv(f) for f in srcs if f.is_file() and f.stat().st_size > 0]
    if not frames:
        return typed(pd.DataFrame(columns=cols))
    d = typed(pd.concat(frames, ignore_index=True).drop_duplicates(keys, keep="last"))
    d.to_csv(path, index=False)
    return d


def append_csv(rows, D, key):
    if rows:
        path = D.result_dir / FILES[key][0]
        pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


def done_ids(df, id_col, cws):
    if not len(df):
        return {}
    d = df[df.class_weight.isin(cws)].groupby(["pipeline", id_col]).class_weight.nunique()
    d = d[d == len(cws)].reset_index()
    return d.groupby("pipeline")[id_col].apply(set).to_dict()


# ---------------------------------------------------------------------------------------------------
def load_dataset(name, args, out_root: Path):
    spec = DATASET_SPECS[name]
    D = SimpleNamespace(name=name, branch=spec["branch"], task=spec["task"], k_list=list(args.k or spec["k"]),
                        n_perm=args.n_perm if spec["branch"] != "3_sanity" else 0,
                        n_random=args.n_random if spec["branch"] != "3_sanity" else 0,
                        ds_dir=DATA_ROOT / C.PERM_DATA_DIR / name, result_dir=out_root / name, stopped=False, t0=time.time())
    D.result_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(D.ds_dir / "metadata.csv", index_col="sample_id")
    pf = pd.read_csv(D.ds_dir / "perm_folds.csv", index_col="sample_id")
    meta.index = meta.index.astype(str)
    pf.index = pf.index.astype(str)
    assert list(pf.index) == list(meta.index) and (pf["label"] == meta["label"]).all()
    D.data_sha256 = {f: sha256(D.ds_dir / f) for f in ["metadata.csv", "perm_folds.csv"]}
    cfg = "perm_config.json" if (D.ds_dir / "perm_config.json").is_file() else "config.json"
    D.data_config = json.load(open(D.ds_dir / cfg, encoding="utf-8")) if (D.ds_dir / cfg).is_file() else {}
    D.excluded = []
    if D.branch == "2_histology" and args.exclude_diagnosis_mismatch and "diagnosis_mismatch" in meta.columns:
        bad = meta["diagnosis_mismatch"].astype(str).str.lower().eq("true")
        D.excluded = meta.index[bad].tolist()
        meta, pf = meta.loc[~bad], pf.loc[~bad]
    if "sample_type" in meta.columns and D.branch != "3_sanity":
        assert (meta["sample_type"] == "Primary Tumor").all(), "nhánh 1/2 chỉ được có mẫu u"
    D.y = meta["label"].to_numpy().astype(int)
    D.groups = meta["patient_id"].astype(str).to_numpy()
    D.sample_ids = meta.index.to_numpy()
    pos = pd.Series(np.arange(len(D.sample_ids)), index=D.sample_ids)
    D.class_names = {int(k): v for k, v in meta.groupby("label")["class"].first().items()}
    assert sorted(D.class_names) == [0, 1], f"chỉ hỗ trợ tác vụ 2 lớp: {D.class_names}"
    D.chance = 0.5
    D.folds, D.fold_data, D.genes = [], [], None
    for r in range(1, N_SPLITS + 1):
        idx, mats = [], []
        for s in ["train", "val", "test"]:
            fp = D.ds_dir / "folds" / f"fold_{r}" / f"{s}.h5"
            h = pd.read_hdf(fp, key="data")
            h.index = h.index.astype(str)
            h = h.loc[~h.index.isin(D.excluded)]
            assert set(h.index) == set(pf.index[pf[f"fold_{r}"] == s]), f"fold {r} {s}: khác perm_folds.csv"
            assert (h["label"].to_numpy() == pf.loc[h.index, "label"].to_numpy()).all(), f"fold {r} {s}: lệch nhãn"
            g = h.columns.drop("label")
            D.genes = g if D.genes is None else D.genes
            assert g.equals(D.genes), f"fold {r} {s}: danh sách gen khác"
            Xs = np.ascontiguousarray(h[g].to_numpy(), dtype=np.float64)
            assert not np.isnan(Xs).any()
            idx.append(pos.loc[h.index].to_numpy())
            mats.append(Xs)
            D.data_sha256[f"folds/fold_{r}/{s}.h5"] = sha256(fp)
        tr, va, te = idx
        assert not (set(D.groups[tr]) & set(D.groups[va]) or set(D.groups[tr]) & set(D.groups[te])
                    or set(D.groups[va]) & set(D.groups[te])), f"fold {r}: trùng bệnh nhân giữa các split"
        for s_idx in idx:
            assert len(np.unique(D.y[s_idx])) == 2, f"fold {r}: thiếu lớp"
        D.folds.append((tr, va, te))
        D.fold_data.append(tuple(mats))
    D.genes = D.genes.to_numpy().astype(str)
    D.n_genes = len(D.genes)
    assert max(D.k_list) < D.n_genes
    test_of = np.zeros(len(D.y), dtype=int)
    for r, (_, _, te) in enumerate(D.folds, 1):
        test_of[te] = r
    assert (test_of > 0).all(), "có mẫu không làm test ở fold nào"
    pf.assign(test_fold=test_of).to_csv(D.result_dir / "fold_assignment.csv")
    D.df = {key: load_prev(D, key, args.resume_dirs) for key in FILES}
    print(f"[{name}] {len(D.y)} mẫu × {D.n_genes} gen | {len(set(D.groups))} bệnh nhân | "
          f"{ {D.class_names[c]: int((D.y == c).sum()) for c in (0, 1)} } | đã có {D.df['null'].pid.nunique()} hoán vị, "
          f"{D.df['rand'].rep.nunique()} random-k", flush=True)
    return D


def run_real(D, args):
    cws = args.class_weights
    done = done_ids(D.df["real"].assign(pid=-1), "pid", cws)
    real_rows, conf_rows, gene_rows, time_rows = [], [], [], []
    for p in [p for p in args.pipelines if -1 not in done.get(p, set())]:
        t0 = time.perf_counter()
        pooled = {(cw, k): np.empty_like(D.y) for cw in cws for k in D.k_list}
        for f, ((Xtr, Xva, Xte), (tr, va, te)) in enumerate(zip(D.fold_data, D.folds), 1):
            preds = fold_predictions(Xtr, D.y[tr], np.vstack([Xva, Xte]), p, cws, D.k_list)
            for (cw, k), (pred, g) in preds.items():
                pv, pt = pred[:len(va)], pred[len(va):]
                pooled[(cw, k)][te] = pt
                real_rows += [{"class_weight": cw, "pipeline": p, "k": k, "fold": f, "split": s,
                               "bacc": balanced_accuracy_score(D.y[i], pr)} for s, i, pr in [("val", va, pv), ("test", te, pt)]]
                gene_rows += [{"class_weight": cw, "pipeline": p, "k": k, "fold": f, "rank": r + 1, "gene": D.genes[j]}
                              for r, j in enumerate(g)]
        for (cw, k), pred in pooled.items():
            cm = confusion_matrix(D.y, pred, labels=[0, 1])
            conf_rows += [{"class_weight": cw, "pipeline": p, "k": k, "true": a, "pred": b, "count": int(cm[a, b])}
                          for a in (0, 1) for b in (0, 1)]
        sec = time.perf_counter() - t0
        time_rows.append({"pipeline": p, "sec_per_run": sec})
        print(f"  {p:10s}: 1 lần chạy (mọi cấu hình, mọi k, 5 fold) = {sec:.1f}s", flush=True)
    for key, rows in [("real", real_rows), ("conf", conf_rows), ("gene", gene_rows), ("time", time_rows)]:
        append_csv(rows, D, key)
        D.df[key] = load_prev(D, key, [])


def run_resamples(D, kind, args, t_start):
    cws = args.class_weights
    id_col, n_total = ("pid", D.n_perm) if kind == "null" else ("rep", D.n_random)
    done = done_ids(D.df[kind], id_col, cws)
    items = [(i, [p for p in args.pipelines if i not in done.get(p, set())]) for i in range(n_total)]
    items = [it for it in items if it[1]]
    import os
    n_workers = os.cpu_count() if args.n_jobs == -1 else args.n_jobs
    batch = n_workers * args.batch_per_worker
    print(f"  {'Permutation' if kind == 'null' else 'Random-k'}: cần chạy {len(items)} (lô {batch})", flush=True)
    for s in range(0, len(items), batch):
        lot = items[s:s + batch]
        t0 = time.time()
        chunks = [lot[i::n_workers] for i in range(n_workers) if lot[i::n_workers]]
        res = Parallel(n_jobs=args.n_jobs)(delayed(run_chunk)(kind, c, D.fold_data, D.folds, D.y, D.groups, D.k_list, cws)
                                           for c in chunks)
        rows = [r for part in res for r in part]
        append_csv(rows, D, kind)
        D.df[kind] = typed(pd.concat([D.df[kind], pd.DataFrame(rows)], ignore_index=True))
        lot_h = (time.time() - t0) / 3600
        n_done = s + len(lot)
        hours_left = args.time_budget_h - (time.time() - t_start) / 3600
        print(f"    {n_done}/{len(items)} | lô {lot_h * 60:.1f} phút | còn ~{lot_h * (len(items) - n_done) / len(lot):.2f} giờ",
              flush=True)
        if n_done < len(items) and hours_left < 1.5 * lot_h + 0.25:
            D.stopped = True
            print("  DỪNG AN TOÀN: sắp hết ngân sách thời gian; chạy lại script để tiếp tục.", flush=True)
            return


# ---------------------------------------------------------------------------------------------------
def run_means(df, id_col):
    if not len(df):
        return pd.DataFrame(columns=[id_col, "class_weight", "pipeline", "k", "bacc"])
    g = df.groupby([id_col, "class_weight", "pipeline", "k"])["bacc"].agg(["mean", "size"]).reset_index()
    return g[g["size"] == N_SPLITS].rename(columns={"mean": "bacc"}).drop(columns="size")


def summarize(D, args, commit_hash):
    rdf = D.df["real"]
    real_mean = rdf[rdf.split == "test"].groupby(["class_weight", "pipeline", "k"])["bacc"].agg(["mean", "std"])
    val_mean = rdf[rdf.split == "val"].groupby(["class_weight", "pipeline", "k"])["bacc"].mean()
    null_m, rand_m = run_means(D.df["null"], "pid"), run_means(D.df["rand"], "rep")
    rows = []
    for (cw, p, k), (real, sd) in real_mean.iterrows():
        null = null_m[(null_m.class_weight == cw) & (null_m.pipeline == p) & (null_m.k == k)]["bacc"].to_numpy()
        rk = rand_m[(rand_m.class_weight == cw) & (rand_m.pipeline == p) & (rand_m.k == k)]["bacc"].to_numpy()
        rows.append({
            "dataset": D.name, "branch": D.branch, "task": D.task, "class_weight": cw,
            "analysis": "main" if cw == MAIN_CW else "sensitivity", "pipeline": p, "k": int(k), "chance": D.chance,
            "bacc_real": real, "bacc_real_sd_folds": sd, "bacc_real_val": val_mean.get((cw, p, k), np.nan),
            "n_perm": len(null), "null_mean": null.mean() if len(null) else np.nan,
            "null_sd": null.std(ddof=1) if len(null) > 1 else np.nan,
            "null_q95": np.quantile(null, 0.95) if len(null) else np.nan,
            "p_perm": (np.sum(null >= real) + 1) / (len(null) + 1) if len(null) else np.nan,
            "n_random": len(rk), "random_mean": rk.mean() if len(rk) else np.nan,
            "random_q95": np.quantile(rk, 0.95) if len(rk) else np.nan,
            "p_random": (np.sum(rk >= real) + 1) / (len(rk) + 1) if len(rk) else np.nan,
            "gain_vs_null": real - null.mean() if len(null) else np.nan,
            "gain_vs_random": real - rk.mean() if len(rk) else np.nan,
        })
    summary = pd.DataFrame(rows)
    summary["p_perm_holm"] = np.nan
    summary["p_random_holm"] = np.nan
    family_size = len(ALL_PIPELINES) * len(D.k_list)
    parts = []
    for cw in summary.class_weight.unique():
        for test, pcol, ncol, n_target in [("perm", "p_perm", "n_perm", D.n_perm), ("random_k", "p_random", "n_random", D.n_random)]:
            g = summary[(summary.class_weight == cw) & summary[pcol].notna()]
            if not len(g):
                continue
            complete = (len(g) == family_size) and (g[ncol].min() >= n_target)
            adj = holm(g[pcol].to_numpy())
            summary.loc[g.index, f"{pcol}_holm"] = adj
            parts.append(pd.DataFrame({
                "dataset": D.name, "branch": D.branch, "task": D.task, "class_weight": cw,
                "analysis": "main" if cw == MAIN_CW else "sensitivity", "test": test,
                "family": f"{D.name}|{D.task}|{cw}|{test}", "family_size": len(g), "family_size_expected": family_size,
                "family_complete": complete, "pipeline": g.pipeline, "k": g.k, "n_resamples": g[ncol], "p_raw": g[pcol],
                "p_holm": adj, "reject_holm": adj < ALPHA, "seed": SEED, "commit_hash": commit_hash}))
    pvals = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    pvals.to_csv(D.result_dir / "pvalues_adjusted.csv", index=False)
    summary["signal"] = summary.p_perm_holm < ALPHA
    summary["useful"] = summary.signal & (summary.p_random < ALPHA) & (summary.gain_vs_random >= MIN_GAIN)
    summary["seed"], summary["commit_hash"] = SEED, commit_hash
    summary.to_csv(D.result_dir / "summary.csv", index=False)
    decision = summary.groupby(["class_weight", "analysis"]).agg(
        n_tests=("bacc_real", "size"), best_bacc=("bacc_real", "max"), min_p_perm=("p_perm", "min"),
        min_p_perm_holm=("p_perm_holm", "min"), min_p_random=("p_random", "min"),
        n_signal=("signal", "sum"), n_useful=("useful", "sum")).reset_index()
    decision.insert(0, "dataset", D.name)
    decision.insert(1, "branch", D.branch)
    decision.insert(2, "task", D.task)
    decision["complete"] = bool(len(pvals) == 0 or pvals.family_complete.all()) and not D.stopped
    if D.n_perm == 0:
        decision["verdict"] = np.where(decision.best_bacc >= 0.9, "SANITY OK: BAcc nhan that >= 0.9",
                                       "SANITY CAN XEM LAI: BAcc nhan that < 0.9")
    else:
        decision["verdict"] = np.where(decision.n_useful > 0, "Co tin hieu va chon gen co ich",
                                       np.where(decision.n_signal > 0, "Co tin hieu nhung chon gen KHONG hon random-k",
                                                "KHONG co bang chung ve tin hieu"))
    decision["commit_hash"] = commit_hash
    decision.to_csv(D.result_dir / "decision.csv", index=False)
    if not decision.complete.all():
        print("  CẢNH BÁO: họ kiểm định chưa đủ (thiếu pipeline hoặc chưa đủ hoán vị) -> p hiệu chỉnh CHƯA phải kết quả cuối.")
    cols = ["class_weight", "pipeline", "k", "bacc_real", "null_mean", "p_perm", "p_perm_holm", "random_mean", "p_random",
            "gain_vs_random", "signal", "useful"]
    print(summary[cols].round(4).to_string(index=False), flush=True)
    return summary, pvals, decision


# ---------------------------------------------------------------------------------------------------
def main(argv=None, script=__file__) -> None:
    args = parse_args(argv)
    prov = start_run(script, params=vars(args))
    commit = prov.get("commit_hash") or prov.get("commit") or "unknown"
    t_start = time.time()
    out_root = results_dir(args.out)
    all_summary, all_pvals, all_decision = [], [], []
    for name in args.datasets:
        D = load_dataset(name, args, out_root)
        run_real(D, args)
        if D.n_perm > 0:
            run_resamples(D, "null", args, t_start)
        if D.n_random > 0 and not D.stopped:
            run_resamples(D, "rand", args, t_start)
        D.stopped = D.stopped or (D.n_perm > D.df["null"].pid.nunique()) or (D.n_random > D.df["rand"].rep.nunique())
        summary, pvals, decision = summarize(D, args, commit)
        all_summary.append(summary)
        all_pvals.append(pvals)
        all_decision.append(decision)
        cfg = {"experiment": "P0_perm_test_v3", "dataset": D.name, "branch": D.branch, "task": D.task, "k_list": D.k_list,
               "seed": SEED, "class_names": D.class_names, "n_samples": int(len(D.y)), "n_patients": int(len(set(D.groups))),
               "class_counts": {D.class_names[c]: int((D.y == c).sum()) for c in (0, 1)}, "n_genes": int(D.n_genes),
               "excluded_samples": D.excluded, "scaling": "none (raw log2)", "pipelines_this_run": args.pipelines,
               "all_pipelines": ALL_PIPELINES, "class_weights": args.class_weights, "main_class_weight": MAIN_CW,
               "pam_prior": {"balanced": "equal", "none": "class proportions"}, "lr_c": LR_C, "svm_c": SVM_C,
               "n_splits": N_SPLITS, "fit_split": "train", "eval_split": "test (p-value); val chỉ ghi lại",
               "fold_sizes_train_val_test": [[int(len(i)) for i in f] for f in D.folds],
               "n_permutations": D.n_perm, "n_random_k": D.n_random,
               "permutation_seed": "np.random.default_rng([42, pid]), xáo theo bệnh nhân",
               "random_k_seed": "np.random.default_rng([42, 7, rep])",
               "multiple_testing": "Holm trong từng họ (dataset, task, class_weight, test in {perm, random_k})",
               "alpha": ALPHA, "min_gain": MIN_GAIN, "n_perm_done": int(D.df["null"].pid.nunique()),
               "n_random_done": int(D.df["rand"].rep.nunique()), "stopped_early": D.stopped,
               "data_sha256_folds": D.data_sha256, "data_build": {"script": D.data_config.get("script") or D.data_config.get("notebook"),
                                                                  "commit_hash": D.data_config.get("commit_hash")},
               "versions": all_versions(), "elapsed_hours_this_dataset": (time.time() - D.t0) / 3600}
        write_config(D.result_dir, prov, cfg, data_files=[D.ds_dir / "metadata.csv", D.ds_dir / "perm_folds.csv"])
        print(f"[{name}] {'CHƯA XONG' if D.stopped else 'xong'} | {(time.time() - D.t0) / 60:.1f} phút", flush=True)
        if args.time_budget_h - (time.time() - t_start) / 3600 < 0.25:
            print("Hết ngân sách thời gian; các bộ còn lại chạy ở lần sau.")
            break
    for frames, fn in [(all_summary, "all_summary.csv"), (all_pvals, "all_pvalues_adjusted.csv"), (all_decision, "all_decisions.csv")]:
        frames = [f for f in frames if len(f)]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_root / fn, index=False)
    print("xong:", out_root, flush=True)


if __name__ == "__main__":
    main()
