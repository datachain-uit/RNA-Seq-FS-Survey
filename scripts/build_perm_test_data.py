"""Tao thu muc du lieu thuc nghiem cho permutation test (3 nhanh) tu v1 + v2.

Dau ra: experiment-data-perm/<bo>/{expression.h5, metadata.csv, config.json, label_mapping.json}
  Nhanh 1 - som (I) vs muon (II-IV), chi mau u:
    tcga_luad_stage_early_late   <- experiment-data/h5/tcga_luad (v1), nhan = stage_group cua v2 da va
    tcga_lusc_stage_early_late   <- experiment-data/h5/tcga_lusc (v1), nhan = stage_group cua v2 da va
    gse68465_stage_early_late    <- experiment-data-v2/datasets/gse68465_stage_tumor_only (da bo 19 Stratagene)
  Nhanh 2 - LUAD vs LUSC:   tcga_luad_lusc_primary_tumor   (chep tu v2, ca folds/)
  Nhanh 3 - Normal vs Tumor: tcga_normal_vs_tumor_sanity   (chep tu v2, ca folds/)

Fold train/val/test giong phan feature selection (v2): 5 phan P1..P5 theo patient_id; fold r: test = P_r,
val = P_(r+1), train = 3 phan con lai.
  - bo chep tu v2: giu nguyen folds/ va fold_assignment.csv cua v2 (dung fold FS da dung)
  - 3 bo giai doan: StratifiedGroupKFold(5, shuffle, seed), thu 100 seed (42..141), chon cach chia lech lop it
    nhat (giong lung_datasets_v2_patient_folds.ipynb), phan tang theo early/late
  perm_folds.csv                         sample_id, patient_id, class, label, part, fold_1..fold_5 (train/val/test)
  folds/fold_<r>/{train,val,test}.h5     key "data": gene... + label (cung dinh dang v2)
Khong chuan hoa (giu log2 goc). Can chay scripts/patch_v2_tcga_stage.py truoc (metadata v2 phai co cot stage_group).

    python scripts/build_perm_test_data.py --commit <hash>
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

try:  # import khi chạy test (scripts.<x>) hoặc chạy trực tiếp (python scripts/<x>.py)
    from scripts.provenance import DATA_ROOT, require
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import DATA_ROOT, require

REPO = DATA_ROOT  # thu muc du lieu (KLTN_DATA_ROOT); code nam trong kho Git
V1 = REPO / "experiment-data" / "h5"
V2 = REPO / "experiment-data-v2" / "datasets"
OUT = REPO / "experiment-data-perm"

EARLY_LATE = {"early": 0, "late": 1}                      # early = I, late = II-IV
STAGE_TO_EL = {"I": "early", "II": "late", "III": "late", "IV": "late"}
V1_CODE_TO_STAGE = {0: "Normal", 1: "I", 2: "II", 3: "III", 4: "IV"}
COPY_FILES = ["expression.h5", "metadata.csv", "config.json", "DATA_CARD.md", "filter_log.csv",
              "class_stats.csv", "fold_assignment.csv"]
SEED = 42
SEED_SEARCH = list(range(SEED, SEED + 100))   # nhu v2: thu 100 seed, chon cach chia can bang lop nhat
N_FOLDS = 5
SPLITS = ["train", "val", "test"]
# Xoay vong nhu v2: fold r -> test = phan r, val = phan r+1 (5 -> 1), train = 3 phan con lai
FOLD_PLAN = {r: {"test": r, "val": r % N_FOLDS + 1,
                 "train": [p for p in range(1, N_FOLDS + 1) if p not in (r, r % N_FOLDS + 1)]}
             for r in range(1, N_FOLDS + 1)}
V2_FOLD_DATASETS = ["tcga_luad_lusc_primary_tumor", "tcga_normal_vs_tumor_sanity"]   # giu fold cua v2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_commit(arg):
    """Lay commit qua provenance (dung neu khong phai git clone, xem scripts/provenance.py)."""
    if arg:
        raise SystemExit("--commit nhap tay da bo (khong kiem chung duoc). Chay trong git clone cua kho; neu buoc phai chay tam, dat KLTN_ALLOW_UNTRACKED=1.")
    return require(__file__)


def class_counts(meta):
    return {c: {"n_samples": int((meta["class"] == c).sum()),
                "n_patients": int(meta.loc[meta["class"] == c, "patient_id"].nunique())}
            for c in sorted(meta["class"].unique(), key=lambda c: meta.loc[meta["class"] == c, "label"].iloc[0])}


def write_dataset(name, X, meta, cfg):
    d = OUT / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    assert list(X.index) == list(meta.index) and X.index.is_unique and meta["patient_id"].notna().all()
    assert not X.isna().any().any(), f"{name}: co NaN trong bieu hien"
    X.astype(np.float32).to_hdf(d / "expression.h5", key="expression", mode="w", complevel=9, complib="blosc")
    meta.to_csv(d / "metadata.csv")
    labels = {c: int(meta.loc[meta["class"] == c, "label"].iloc[0]) for c in meta["class"].unique()}
    json.dump(dict(sorted(labels.items(), key=lambda kv: kv[1])), open(d / "label_mapping.json", "w"), indent=2)
    cfg.update({"n_samples": int(len(meta)), "n_patients": int(meta["patient_id"].nunique()),
                "n_genes": int(X.shape[1]), "class_counts": class_counts(meta),
                "files_sha256": {f: sha256(d / f) for f in ["expression.h5", "metadata.csv"]}})
    json.dump(cfg, open(d / "config.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  {name}: {X.shape[0]} mau x {X.shape[1]} gene | {meta['class'].value_counts().to_dict()}")
    return cfg


def tcga_stage(project, v2_meta, base_cfg):
    # v1: gop train/val/test cua fold_1 (= toan bo), bo Normal, gan lai nhan theo stage_group v2 (theo patient_id)
    src = V1 / f"tcga_{project}"
    files = [src / "fold_1" / f"{s}.h5" for s in ["train", "val", "test"]]
    df = pd.concat([pd.read_hdf(f, key="data") for f in files])
    assert df.index.is_unique
    n_all = len(df)
    df = df[df["label"] != 0]
    n_tumor = len(df)
    X = df.drop(columns="label")

    v2 = v2_meta[v2_meta["project"] == f"TCGA-{project.upper()}"].set_index("patient_id")
    meta = pd.DataFrame(index=X.index)
    meta["patient_id"] = X.index.str[:12]
    assert meta["patient_id"].is_unique, "v1 co >1 mau u / benh nhan"
    meta["stage_v1"] = df["label"].map(V1_CODE_TO_STAGE)
    for c in ["stage_group", "stage_raw", "stage_rule", "stage_conflict", "primary_diagnosis"]:
        meta[c] = meta["patient_id"].map(v2[c])
    meta["v2_sample_id"] = meta["patient_id"].map(v2["sample_id"])
    meta["in_v2"] = meta["patient_id"].isin(v2.index)
    meta["stage_changed_vs_v1"] = meta["stage_group"].notna() & (meta["stage_group"] != meta["stage_v1"])

    keep = meta["stage_group"].isin(STAGE_TO_EL)
    dropped = meta.loc[~keep].index.tolist()
    meta, X = meta.loc[keep], X.loc[keep]
    meta.insert(1, "class", meta["stage_group"].map(STAGE_TO_EL))
    meta.insert(2, "label", meta["class"].map(EARLY_LATE).astype(int))

    changed = meta.loc[meta["stage_changed_vs_v1"], ["patient_id", "stage_v1", "stage_group"]]
    print(f"  TCGA-{project.upper()}: doi nhan so voi v1 o {len(changed)} benh nhan:",
          changed.apply(lambda r: f"{r.patient_id} {r.stage_v1}->{r.stage_group}", axis=1).tolist())
    cfg = dict(base_cfg, **{
        "dataset": f"tcga_{project}_stage_early_late",
        "branch": "1_stage_early_late",
        "task": f"Giai doan som (I) vs muon (II-IV), chi mau u nguyen phat TCGA-{project.upper()}",
        "label_mapping": EARLY_LATE,
        "expression": "log2(TPM+1), GDC STAR - Counts, protein_coding, gene_name (bieu hien cua v1), khong chuan hoa",
        "source": {"expression": [str(f.relative_to(REPO)) for f in files],
                   "expression_sha256": {str(f.relative_to(REPO)): sha256(f) for f in files},
                   "stage_labels": "experiment-data-v2/datasets/tcga_luad_lusc_primary_tumor/metadata.csv "
                                   "(cot stage_group, da va bang scripts/patch_v2_tcga_stage.py; GDC moi ban ghi "
                                   "diagnoses, uu tien u chi diem khi conflict), ghep theo patient_id"},
        "sample_counts": [
            {"buoc": "v1 fold_1 train+val+test", "n_samples": n_all},
            {"buoc": "bo mau Normal (label 0)", "n_samples": n_tumor},
            {"buoc": "chi giu benh nhan co stage_group I-IV trong v2 da va", "n_samples": int(len(meta)),
             "removed_sample_ids": dropped},
        ],
        "stage_changed_vs_v1": changed.set_index("patient_id").to_dict(orient="index"),
        "notes": ["Fold KHONG di kem: notebook permutation tao lai StratifiedGroupKFold theo patient_id.",
                  "Cot v2_sample_id khac sample_id khi v1 va v2 chon mau/aliquot khac cho cung benh nhan."],
    })
    return write_dataset(cfg["dataset"], X, meta, cfg)


def gse_stage(base_cfg):
    src = V2 / "gse68465_stage_tumor_only"
    X = pd.read_hdf(src / "expression.h5", key="expression")
    meta = pd.read_csv(src / "metadata.csv", index_col="sample_id").loc[X.index]
    assert not meta["source_name"].str.contains("stratagene", case=False).any()
    assert meta["stage"].isin(STAGE_TO_EL).all()
    meta = meta.rename(columns={"class": "class_stage", "label": "label_stage"})
    meta.insert(1, "class", meta["stage"].map(STAGE_TO_EL))
    meta.insert(2, "label", meta["class"].map(EARLY_LATE).astype(int))
    cfg = dict(base_cfg, **{
        "dataset": "gse68465_stage_early_late",
        "branch": "1_stage_early_late",
        "task": "Giai doan som (I) vs muon (II-III; khong co mau IV), mau u LUAD GSE68465, da bo 19 mau Stratagene",
        "label_mapping": EARLY_LATE,
        "expression": "log2(MAS5+1), GPL96, probe -> gene symbol (giu probe phuong sai lon nhat), khong chuan hoa",
        "source": {"dataset": str(src.relative_to(REPO)),
                   "expression_sha256": sha256(src / "expression.h5"),
                   "stage_labels": "cot stage (pTNM -> AJCC, gia dinh M0) cua v2"},
        "notes": ["Nhan giai doan goc giu o cot class_stage / label_stage.",
                  "Fold KHONG di kem: notebook permutation tao lai StratifiedGroupKFold theo patient_id."],
    })
    return write_dataset(cfg["dataset"], X, meta, cfg)


def copy_v2(name, branch, base_cfg):
    src, d = V2 / name, OUT / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    fold_files = [f"folds/fold_{r}/{s}.h5" for r in FOLD_PLAN for s in SPLITS]
    for f in COPY_FILES + fold_files:
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, d / f)
        assert sha256(src / f) == sha256(d / f)
    cfg = json.load(open(src / "config.json", encoding="utf-8"))
    json.dump(cfg["label_mapping"], open(d / "label_mapping.json", "w"), indent=2)
    perm = dict(base_cfg, branch=branch, copied_from=str(src.relative_to(REPO)),
                copied_files_sha256={f: sha256(d / f) for f in COPY_FILES + fold_files},
                notes=["Chep nguyen tu v2, ke ca folds/ train/val/test (fold da dung cho feature selection)."])
    json.dump(perm, open(d / "perm_config.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  {name}: chep {len(COPY_FILES)} file | {cfg['class_counts']}")
    return {"dataset": name, "branch": branch, "n_samples": cfg["n_samples"], "class_counts": cfg["class_counts"]}


def make_parts(y, groups):
    # Nhu make_parts cua v2: thu SEED_SEARCH, chon cach chia co tong do lech so mau/lop so voi ly tuong nho nhat
    ideal = pd.Series(y).value_counts().sort_index() / N_FOLDS
    best = (None, np.inf, None)
    for s in SEED_SEARCH:
        part = np.empty(len(y), dtype=int)
        for k, (_, idx) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=s)
                                     .split(np.zeros(len(y)), y, groups), start=1):
            part[idx] = k
        ct = pd.crosstab(part, y).reindex(columns=ideal.index, fill_value=0)
        score = float((ct - ideal.values).abs().values.sum())
        if score < best[1]:
            best = (part, score, s)
    return best


def make_folds(name):
    # Fold train/val/test giong FS -> perm_folds.csv (+ folds/fold_<r>/{train,val,test}.h5 cho bo giai doan)
    d = OUT / name
    X = pd.read_hdf(d / "expression.h5", key="expression")
    meta = pd.read_csv(d / "metadata.csv", index_col="sample_id").loc[X.index]
    y = meta["label"].to_numpy().astype(int)
    groups = meta["patient_id"].astype(str).to_numpy()
    pf = meta[["patient_id", "class", "label"]].copy()

    if name in V2_FOLD_DATASETS:
        fa = pd.read_csv(d / "fold_assignment.csv", index_col="sample_id").loc[X.index]
        assert (fa["label"].to_numpy() == y).all()
        pf["part"] = fa["part"].astype(int)
        cfg_v2 = json.load(open(d / "config.json", encoding="utf-8"))
        method = (f"Giu nguyen fold v2 ({cfg_v2['split_method']}; selected_seed={cfg_v2['selected_seed']}, "
                  f"stratify_by={cfg_v2['stratify_by']})")
    else:
        part, score, seed = make_parts(y, groups)
        pf["part"] = part
        method = (f"StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state=selected_seed), "
                  f"groups=patient_id, strata=label (early/late); seed {SEED_SEARCH[0]}..{SEED_SEARCH[-1]} -> "
                  f"selected_seed={seed}, split_imbalance_score={score:.1f}; fold r: test=P_r, val=P_(r+1), "
                  f"train=con lai")
    for r, plan in FOLD_PLAN.items():
        role = {p: "train" for p in plan["train"]}
        role[plan["val"]], role[plan["test"]] = "val", "test"
        pf[f"fold_{r}"] = pf["part"].map(role)

    # Kiem tra nhu verify_no_patient_leak cua v2 + moi split du 2 lop
    assert (pf.groupby("patient_id")["part"].nunique() == 1).all(), "benh nhan nam o nhieu phan"
    roles = pf[[f"fold_{r}" for r in FOLD_PLAN]]
    assert ((roles == "test").sum(axis=1) == 1).all() and ((roles == "val").sum(axis=1) == 1).all()
    stats = []
    for r in FOLD_PLAN:
        split = pf[f"fold_{r}"].to_numpy()
        pats = {s: set(groups[split == s]) for s in SPLITS}
        assert not (pats["train"] & pats["val"] or pats["train"] & pats["test"] or pats["val"] & pats["test"])
        fdir = d / "folds" / f"fold_{r}"
        fdir.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            mask = split == s
            assert len(np.unique(y[mask])) == 2, f"{name} fold {r} {s}: thieu lop"
            fp = fdir / f"{s}.h5"
            if name in V2_FOLD_DATASETS:   # file v2 da chep -> chi kiem tra khop
                h = pd.read_hdf(fp, key="data")
                assert set(h.index) == set(X.index[mask]) and (h["label"] == pf.loc[h.index, "label"]).all()
            else:
                X[mask].assign(label=y[mask]).to_hdf(fp, key="data", mode="w", complevel=9, complib="blosc")
            stats += [{"fold": r, "split": s, "class": c, "n_samples": int((mask & (y == lab)).sum()),
                       "n_patients": int(len(set(groups[mask & (y == lab)])))}
                      for lab, c in sorted(meta.groupby("label")["class"].first().items())]
    pf.to_csv(d / "perm_folds.csv")
    pd.DataFrame(stats).to_csv(d / "perm_fold_stats.csv", index=False)

    info = {"method": method,
            "splits": "train/val/test nhu feature selection; fold r: test = P_r, val = P_(r+1), train = 3 phan con lai",
            "fold_plan": {f"fold_{r}": {"train_parts": p["train"], "val_part": p["val"], "test_part": p["test"]}
                          for r, p in FOLD_PLAN.items()},
            "files": ["perm_folds.csv", "perm_fold_stats.csv", "folds/fold_<r>/{train,val,test}.h5 (key 'data')"],
            "test_sizes": [int((pf[f"fold_{r}"] == "test").sum()) for r in FOLD_PLAN],
            "perm_folds_sha256": sha256(d / "perm_folds.csv")}
    cfg_path = d / ("perm_config.json" if (d / "perm_config.json").exists() else "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["perm_folds"] = info
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  {name}: test moi fold {info['test_sizes']}")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", help="DA BO: commit lay tu git qua provenance()")
    args = ap.parse_args()
    prov = resolve_commit(args.commit)

    v2_meta = pd.read_csv(V2 / "tcga_luad_lusc_primary_tumor" / "metadata.csv")
    assert "stage_group" in v2_meta.columns, "Chua va metadata v2: chay scripts/patch_v2_tcga_stage.py truoc"

    base_cfg = {"script": "scripts/build_perm_test_data.py", "commit_hash": prov["commit_hash"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
    OUT.mkdir(exist_ok=True)
    print(f"Ghi vao {OUT}")
    summary = [tcga_stage("luad", v2_meta, base_cfg), tcga_stage("lusc", v2_meta, base_cfg), gse_stage(base_cfg),
               copy_v2("tcga_luad_lusc_primary_tumor", "2_luad_vs_lusc", base_cfg),
               copy_v2("tcga_normal_vs_tumor_sanity", "3_normal_vs_tumor_sanity", base_cfg)]
    print("Chia fold train/val/test theo benh nhan (giong feature selection)")
    folds = {s["dataset"]: make_folds(s["dataset"]) for s in summary}
    manifest = dict(base_cfg, provenance=prov, datasets=[{**{k: s[k] for k in ["dataset", "branch", "n_samples", "class_counts"]},
                                         "perm_folds": folds[s["dataset"]]["method"]} for s in summary])
    json.dump(manifest, open(OUT / "manifest.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("XONG.")


if __name__ == "__main__":
    main()
