"""Sap xep ket qua feature selection phase 1 (moi zip chua k_<k>/ cho moi k).

    results/phase 1/fs_<method>_<bo>[_p<i>].zip
 -> results/phase 1/fs/<nhom>/fs_<method>_<bo>[_p<i>].zip          (zip goc, chuyen vao nhom)
    results/phase 1/fs/<nhom>/fs_<method>_<bo>/k_<k>/...            (giai nen, bo population_state_history/)
    selected_features_phase1/<nhom>/fs_<method>_<bo>/k_<k>/fold_r.csv

Zip chia nhieu phan (_p1, _p2, ...: chay lai cac k con thieu) duoc gop vao 1 thu muc; 2 file tong hop
metrics_per_fold_all_k.csv / metrics_summary_by_k.csv duoc dung lai tu cac k_<k>/ sau khi gop.
Nhom (filter/wrapper/embedded) lay tu thu muc notebook phase 1 cua phuong phap.

    python scripts/organize_fs_phase1.py [--dry-run]
"""
import argparse, glob, io, json, os, re, shutil, sys, zipfile
import numpy as np
import pandas as pd

try:  # import khi chạy test (scripts.<x>) hoặc chạy trực tiếp (python scripts/<x>.py)
    from scripts.provenance import CODE_ROOT, DATA_ROOT, require, save_provenance
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import CODE_ROOT, DATA_ROOT, require, save_provenance

ROOT = str(DATA_ROOT)  # du lieu/ket qua (KLTN_DATA_ROOT); notebook doc tu kho code
RES = os.path.join(ROOT, "results", "phase 1")
NB_DIR = os.path.join(str(CODE_ROOT), "notebooks", "phase_1", "01_feature_selection")
SEL = os.path.join(ROOT, "selected_features_phase1")
N_FOLDS = 5
HEAVY = "population_state_history/"
K_DIR = re.compile(r"(?:^|/)(k_(\d+))/")
AGG = ("metrics_per_fold_all_k.csv", "metrics_summary_by_k.csv")


def method_groups():
    out = {}
    for p in glob.glob(os.path.join(NB_DIR, "*", "fs_*_phase1.ipynb")):
        out[os.path.basename(p)[3:-len("_phase1.ipynb")]] = os.path.basename(os.path.dirname(p))
    return out


def zip_prefix(zf):
    # Thu muc chua k_<k>/ ben trong zip ("" hoac vd "feature_selection/scfsnn/<bo>/")
    for n in zf.namelist():
        m = K_DIR.search(n)
        if m:
            return n[:m.start(1)]
    raise ValueError("zip khong co thu muc k_<k>/")


def jaccard_mean(sel):
    sets = [set(g.gene) for _, g in sel.groupby("fold")]
    pairs = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(pairs))


def rebuild_aggregates(run_dir):
    ks = sorted(int(d[2:]) for d in os.listdir(run_dir) if re.fullmatch(r"k_\d+", d))
    per_fold, rows = [], []
    for k in ks:
        d = os.path.join(run_dir, f"k_{k}")
        m = pd.read_csv(os.path.join(d, "metrics_per_fold.csv"))
        sel = pd.read_csv(os.path.join(d, "selected_genes_all_folds.csv"))
        per_fold.append(m)
        rows.append({"k": k, "n_selected_mean": m["n_selected"].mean(),
                     "time_sec_total": m["time_sec"].sum(), "time_sec_mean": m["time_sec"].mean(),
                     "peak_memory_mb_max": m["peak_memory_mb"].max(),
                     "energy_kwh_total": m["energy_kwh"].sum(),
                     "carbon_g_co2eq_total": m["carbon_g_co2eq"].sum(), "jaccard_mean": jaccard_mean(sel)})
    pd.concat(per_fold, ignore_index=True).to_csv(os.path.join(run_dir, AGG[0]), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, AGG[1]), index=False)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    prov = None if args.dry_run else require(__file__)

    groups = method_groups()
    runs = {}   # stem -> [zip path, ...]
    for z in sorted(glob.glob(os.path.join(RES, "*.zip")) + glob.glob(os.path.join(RES, "fs", "*", "*.zip"))):
        stem = re.sub(r"_p\d+$", "", os.path.basename(z)[:-4])
        runs.setdefault(stem, []).append(z)

    errors, report = [], []
    for stem, zips in sorted(runs.items()):
        method = next((m for m in sorted(groups, key=len, reverse=True) if stem.startswith(f"fs_{m}_")), None)
        if method is None:
            errors.append(f"{stem}: khong tim thay notebook phase 1 cua phuong phap")
            continue
        group = groups[method]
        gdir = os.path.join(RES, "fs", group)
        run_dir = os.path.join(gdir, stem)
        print(f"{stem}: nhom={group}, {len(zips)} zip -> {os.path.relpath(run_dir, ROOT)}")
        if args.dry_run:
            continue
        os.makedirs(gdir, exist_ok=True)

        # 1) Giai nen (gop neu nhieu phan); k trung lap giua cac phan -> loi
        seen_k, skipped_heavy = {}, 0
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                pre = zip_prefix(zf)
                for info in zf.infolist():
                    n = info.filename
                    if not n.startswith(pre) or n.endswith("/"):
                        continue
                    rel = n[len(pre):]
                    if HEAVY in rel:
                        skipped_heavy += 1
                        continue
                    if len(zips) > 1 and rel in AGG:
                        continue      # dung lai sau khi gop
                    m = K_DIR.search(rel)
                    if m:
                        k = int(m.group(2))
                        if seen_k.setdefault(k, z) != z:
                            raise SystemExit(f"LOI: k_{k} cua {stem} co o ca {seen_k[k]} va {z}")
                    out = os.path.join(run_dir, *rel.split("/"))
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with zf.open(info) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        if len(zips) > 1:
            rebuild_aggregates(run_dir)
            with open(os.path.join(run_dir, "merged_from.json"), "w") as f:
                json.dump({os.path.basename(z): sorted(k for k, zz in seen_k.items() if zz == z) for z in zips}, f, indent=1)

        # 2) Kiem tra so gene + chep fold file sang selected_features_phase1/
        ks = sorted(int(d[2:]) for d in os.listdir(run_dir) if re.fullmatch(r"k_\d+", d))
        for k in ks:
            for r in range(1, N_FOLDS + 1):
                src = os.path.join(run_dir, f"k_{k}", "selected_genes", f"fold_{r}.csv")
                if not os.path.isfile(src):
                    errors.append(f"{stem} k_{k}: thieu fold_{r}.csv")
                    continue
                df = pd.read_csv(src)
                if len(df) != k or df.gene.nunique() != k:
                    errors.append(f"{stem} k_{k} fold_{r}: {len(df)} dong / {df.gene.nunique()} gene (can {k})")
                dst = os.path.join(SEL, group, stem, f"k_{k}", f"fold_{r}.csv")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

        # 3) Chuyen zip goc vao thu muc nhom
        for z in zips:
            if os.path.dirname(z) != gdir:
                shutil.move(z, os.path.join(gdir, os.path.basename(z)))
        report.append({"nhom": group, "fs_run": stem, "so_zip": len(zips), "k": ",".join(map(str, ks)),
                       "fold_files": len(ks) * N_FOLDS, "bo_qua_population_state": skipped_heavy})

    if report:
        print(pd.DataFrame(report).sort_values(["nhom", "fs_run"]).to_string(index=False))
    if prov is not None:
        prov["params"] = {"runs": report, "errors": errors}
        save_provenance(os.path.join(RES, "fs"), prov, filename="organize_provenance.json")
    for e in errors:
        print("LOI:", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
