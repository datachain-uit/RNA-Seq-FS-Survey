"""Kiểm soát chất lượng mẫu cho 4 cohort GEO đích (QĐ21, Review 29/10/2026; Cổng 1′).

Các lệnh con (QĐ21 + QĐ22), theo thứ tự dùng:

  fetch-tcga-sex  TCGA: lấy giới tính (demographic.gender) từ GDC API theo patient_id, ghi
              <tcga>/sex_gdc.csv + sex_gdc_source.json (nguồn cho cột sex của TCGA). Cần mạng tới api.gdc.cancer.gov.
  rebuild --sex-only  chỉ thêm cột ``sex`` vào metadata.csv (không bỏ mẫu nào) để chạy check_sex_consistency.py.
  apply-sex-qc  đọc qc/sex_check_<bộ>.csv (scripts/check_sex_consistency.py) và thêm mẫu action = quarantine
              (opposite, inconsistent) vào excluded_samples_<bộ>.csv kèm lý do; không sửa nhãn/giới tính.

  rebuild     Dựng lại <cohort>/{expression.h5, metadata.csv} từ chính bản v3 hiện có:
              * bỏ các mẫu liệt kê trong <cohort>/excluded_samples_<cohort>.csv (cột sample_id, reason, decision);
              * thêm cột ``sex`` (male / female / rỗng) lấy từ trường "sex:"/"gender:" của series matrix
                (đã có sẵn trong các cột Sample_characteristics_* của metadata.csv; GSE10245 không công bố giới tính);
              * KHÔNG đổi giá trị nào khác: mọi mẫu còn lại giữ nguyên thứ tự, nhãn, biểu hiện (kiểm tra bằng assert);
              * sao lưu bản cũ vào <cohort>/_backup_<YYYYmmdd-HHMMSS>/ và ghi <cohort>/rebuild_log.json
                (SHA-256 cũ/mới, mẫu đã bỏ, commit).
  sex-check   Giới tính theo biểu hiện (XIST so với RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY; giống check_duplicate_pair.py)
              cho MỌI mẫu, so với cột ``sex``: status ∈ {match, mismatch, unclear (|sex_score| < --unclear), no_metadata}.
              Ghi <out>/sex_check_<cohort>.csv. Mẫu mismatch: KHÔNG tự cách ly — báo GVHD (QĐ21 mục 7a).
  rz-screen   Sàng mẫu trùng bằng r_z (Pearson sau khi z-score từng gen trong cohort) trong từng cohort và giữa các
              cohort (trên gen chung của cả 4 cohort), kèm r thô trong cohort. Ghi <out>/rz_screen_pairs.csv (mọi cặp
              r_z >= --threshold hoặc r >= --raw-threshold) và <out>/rz_screen_summary.csv (theo cặp cohort).

    python scripts/geo_sample_qc.py fetch-tcga-sex
    python scripts/geo_sample_qc.py rebuild --sex-only --cohorts gse10245 gse30219 gse37745 gse42127 tcga_luad_lusc_primary_tumor
    python scripts/check_sex_consistency.py
    python scripts/geo_sample_qc.py apply-sex-qc
    python scripts/geo_sample_qc.py rebuild --cohorts gse10245 gse30219 gse37745 gse42127 tcga_luad_lusc_primary_tumor
    python scripts/geo_sample_qc.py sex-check --out D:/KLTN/Nop-GVHD-2026-10-29/04_Du-lieu/_kiem_tra
    python scripts/geo_sample_qc.py rz-screen --out D:/KLTN/Nop-GVHD-2026-10-29/04_Du-lieu/_kiem_tra

Dữ liệu: $KLTN_DATA_ROOT (hoặc --data-root). ``rebuild`` dừng nếu script chưa commit + push (truy vết, Y13) trừ khi
đặt KLTN_ALLOW_UNPUSHED=1 (chỉ để thử). Chạy ``rebuild`` lại lần hai không đổi gì (idempotent): mẫu đã bỏ thì bỏ qua,
cột sex đã có thì giữ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import resolve_dataset_dir  # noqa: E402

GEO = tuple(C.FEWSHOT_COHORTS)
FEMALE = ("XIST",)
MALE = ("RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY")
SEX_FIELD = re.compile(r"^\s*(sex|gender)\s*:\s*(.*)$", re.I)
SEX_MAP = {"m": "male", "male": "male", "f": "female", "female": "female"}


# --------------------------------------------------------------------------------------------- tiện ích chung
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, OSError):
            return None
    head = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    tracked = run("ls-files", "--error-unmatch", str(Path(__file__).resolve().relative_to(REPO))) is not None
    pushed = None
    if head:
        remote = run("branch", "-r", "--contains", head)
        pushed = bool(remote)
    return {"commit_hash": head, "dirty": bool(dirty), "dirty_files": (dirty or "").splitlines()[:20],
            "script_tracked": tracked, "pushed": pushed}


def require_traceable() -> dict:
    state = git_state()
    problems = []
    if not state["script_tracked"]:
        problems.append("scripts/geo_sample_qc.py chưa được commit")
    if state["dirty"]:
        problems.append("kho có thay đổi chưa commit (git status):\n    " + "\n    ".join(state["dirty_files"]))
    if not state["pushed"]:
        problems.append("HEAD chưa push")
    state["traceable"] = not problems
    if problems and os.environ.get("KLTN_ALLOW_UNPUSHED", "").strip().lower() not in {"1", "true", "yes"}:
        raise SystemExit("[traceable] DỪNG: " + "; ".join(problems) + " (thử: KLTN_ALLOW_UNPUSHED=1; kết quả khi đó không được dùng)")
    state["traceable_errors"] = problems
    return state


def read_metadata_text(path: Path) -> pd.DataFrame:
    """metadata.csv đọc nguyên văn (mọi ô là chuỗi) để ghi lại không đổi định dạng các cột cũ."""
    meta = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "sample_id" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "sample_id"})
    return meta


def read_expression(path: Path) -> tuple[pd.DataFrame, str]:
    with pd.HDFStore(path, "r") as store:
        keys = store.keys()
    key = "/expression" if "/expression" in keys else keys[0]
    return pd.read_hdf(path, key=key), key.lstrip("/")


def h5_filters(path: Path) -> tuple[int, str]:
    import tables
    with tables.open_file(str(path), "r") as f:
        for node in f.walk_nodes("/", "Array"):
            flt = node.filters
            return int(flt.complevel), str(flt.complib or "zlib")
    return 4, "zlib"


def parse_sex(meta: pd.DataFrame) -> tuple[pd.Series, str | None]:
    """Giới tính từ cột Sample_characteristics_* có dạng 'sex: m' / 'gender: female'. Trả về (Series, tên cột nguồn)."""
    for col in meta.columns:
        if not col.startswith("Sample_characteristics"):
            continue
        m = meta[col].astype(str).str.extract(SEX_FIELD)
        if m[0].notna().mean() >= 0.9:
            values = m[1].str.strip().str.lower().map(SEX_MAP)
            if values.notna().mean() >= 0.9:
                return values.fillna(""), col
    return pd.Series([""] * len(meta), index=meta.index), None


TCGA = "tcga_luad_lusc_primary_tumor"
GDC_API = "https://api.gdc.cancer.gov"


def sex_for(d: Path, meta: pd.DataFrame) -> tuple[pd.Series, str | None]:
    """Cột sex: GEO từ Sample_characteristics (theo khoá sex:/gender:); TCGA từ sex_gdc.csv (theo patient_id)."""
    gdc = d / "sex_gdc.csv"
    if gdc.exists() and "patient_id" in meta.columns:
        table = pd.read_csv(gdc, dtype=str, keep_default_na=False).set_index("patient_id")["sex"]
        return meta["patient_id"].map(table).fillna(""), "sex_gdc.csv (GDC demographic: gender, nếu trống thì sex_at_birth; ngày truy vấn trong sex_gdc_source.json)"
    return parse_sex(meta)


# --------------------------------------------------------------------------------------------- rebuild
def rebuild(data_root: Path, cohort: str, prov: dict, dry_run: bool = False, sex_only: bool = False) -> dict:
    d = resolve_dataset_dir(data_root, cohort)
    meta_path, expr_path = d / "metadata.csv", d / "expression.h5"
    meta = read_metadata_text(meta_path)
    X, key = read_expression(expr_path)
    X.index = X.index.astype(str)
    if list(X.index) != meta["sample_id"].tolist():
        raise SystemExit(f"{cohort}: thứ tự sample_id của expression.h5 và metadata.csv khác nhau — dừng, kiểm tra tay")

    excl_path = d / f"excluded_samples_{cohort}.csv"
    excluded = pd.read_csv(excl_path, dtype=str) if excl_path.exists() else pd.DataFrame(columns=["sample_id"])
    drop = [] if sex_only else [s for s in excluded["sample_id"].astype(str) if s in set(meta["sample_id"])]
    missing = [s for s in excluded["sample_id"].astype(str) if s not in set(meta["sample_id"])]
    if cohort == TCGA and not (d / "sex_gdc.csv").exists() and "sex" not in meta.columns:
        raise SystemExit(f"{cohort}: chưa có sex_gdc.csv — chạy `geo_sample_qc.py fetch-tcga-sex` trước")

    sex, sex_source = sex_for(d, meta)
    has_sex = "sex" in meta.columns
    new_meta = meta.copy()
    new_meta["sex"] = sex.values                     # cột dẫn xuất: luôn tính lại từ nguồn (series matrix / sex_gdc.csv)
    sex_changed = (not has_sex) or not meta["sex"].equals(new_meta["sex"])
    keep = ~new_meta["sample_id"].isin(drop)
    new_meta = new_meta.loc[keep].reset_index(drop=True)
    new_X = X.loc[new_meta["sample_id"].tolist()]

    # --- bất biến: mẫu còn lại không đổi gì ngoài cột sex mới
    base_cols = [c for c in meta.columns if c != "sex"]
    old_kept = meta.loc[keep.values].reset_index(drop=True)
    assert old_kept[base_cols].equals(new_meta[base_cols]), "metadata của mẫu còn lại bị đổi"
    assert np.array_equal(X.loc[keep.values].to_numpy(), new_X.to_numpy(), equal_nan=True), "biểu hiện bị đổi"
    assert list(new_X.columns) == list(X.columns)
    assert new_X.shape[0] == X.shape[0] - len(drop)

    labels = new_meta["label"].astype(int)
    n_per_class = [int((labels == 0).sum()), int((labels == 1).sum())]
    changed = bool(drop) or sex_changed
    log = {"cohort": cohort, "data_dir": str(d), "excluded_file": str(excl_path) if excl_path.exists() else None,
           "mode": "sex-only" if sex_only else "full", "excluded_now": drop, "excluded_already_absent": missing, "n_before": int(X.shape[0]), "n_after": int(new_X.shape[0]),
           "n_per_class_after": n_per_class, "n_per_class_expected": list(C.DATASET_EXPECTED_N.get(cohort, ())),
           "sex_column": ("unchanged" if not sex_changed else ("updated" if has_sex else "added")) + " from " + (sex_source or "nothing (not in source metadata)"),
           "sex_counts": new_meta["sex"].replace("", "unknown").value_counts().to_dict(),
           "sha256_before": {"expression.h5": sha256(expr_path), "metadata.csv": sha256(meta_path)},
           "changed": changed, "script": "scripts/geo_sample_qc.py rebuild", **prov,
           "run_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    if not changed or dry_run:
        log["sha256_after"] = log["sha256_before"]
        return log

    stamp_ = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup, k = d / f"_backup_{stamp_}", 1
    while backup.exists():                           # hai lần chạy trong cùng một giây
        backup, k = d / f"_backup_{stamp_}_{k}", k + 1
    backup.mkdir()
    for p in (expr_path, meta_path):
        shutil.copy2(p, backup / p.name)
    tmp_h5, tmp_csv = d / "expression.h5.tmp", d / "metadata.csv.tmp"
    new_meta.to_csv(tmp_csv, index=False, encoding="utf-8")
    chk_meta = read_metadata_text(tmp_csv)          # đọc lại từ đĩa trước khi thay
    assert chk_meta.equals(new_meta)
    if drop:                                         # expression.h5 chỉ ghi lại khi có mẫu bị bỏ (giữ nguyên SHA nếu không)
        complevel, complib = h5_filters(expr_path)
        new_X.to_hdf(tmp_h5, key=key, mode="w", complevel=complevel, complib=complib)
        chk_X, _ = read_expression(tmp_h5)
        assert np.array_equal(chk_X.to_numpy(), new_X.to_numpy(), equal_nan=True)
        assert list(chk_X.index.astype(str)) == chk_meta["sample_id"].tolist()
        os.replace(tmp_h5, expr_path)
    os.replace(tmp_csv, meta_path)
    log["backup_dir"] = str(backup)
    log["sha256_after"] = {"expression.h5": sha256(expr_path), "metadata.csv": sha256(meta_path)}
    history_path = d / "rebuild_log.json"                     # lịch sử: mỗi lần ghi thêm một mục
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    history = history if isinstance(history, list) else [history]
    history.append(log)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return log


# --------------------------------------------------------------------------------------------- sex-check
def load_geo(data_root: Path, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = resolve_dataset_dir(data_root, cohort)
    X, _ = read_expression(d / "expression.h5")
    X.index = X.index.astype(str)
    meta = read_metadata_text(d / "metadata.csv").set_index("sample_id")
    meta = meta.loc[X.index]
    return X, meta


def sex_check(data_root: Path, cohort: str, unclear: float) -> pd.DataFrame:
    X, meta = load_geo(data_root, cohort)
    male = [g for g in MALE if g in X.columns]
    if "XIST" not in X.columns or not male:
        raise SystemExit(f"{cohort}: thiếu XIST hoặc gen Y trong expression.h5")
    t = X[["XIST", *male]].astype(float).copy()
    t.insert(0, "label", meta["label"].astype(int))
    t["sex_score"] = t[male].mean(axis=1) - t["XIST"]
    t["sex_inferred"] = np.where(t["sex_score"] > 0, "male", "female")
    meta_sex = meta["sex"] if "sex" in meta.columns else parse_sex(meta.reset_index())[0].set_axis(meta.index)
    t["sex_metadata"] = meta_sex.replace("", np.nan)
    status = np.where(t["sex_metadata"].isna(), "no_metadata",
                      np.where(t["sex_score"].abs() < unclear, "unclear",
                               np.where(t["sex_metadata"] == t["sex_inferred"], "match", "mismatch")))
    t["status"] = status
    t.index.name = "sample_id"
    return t.round(4)


# --------------------------------------------------------------------------------------------- rz-screen
def zscore_genes(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    ok = sd > 0
    Z = np.zeros_like(X)
    Z[:, ok] = (X[:, ok] - mu[ok]) / sd[ok]
    return Z


def row_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def rz_screen(data_root: Path, cohorts, threshold: float, raw_threshold: float):
    data = {c: load_geo(data_root, c) for c in cohorts}
    shared = sorted(set.intersection(*(set(X.columns) for X, _ in data.values())))
    pairs, summary = [], []

    def info(c, s):
        meta = data[c][1]
        return {"label": int(meta.loc[s, "label"]), "sex_metadata": meta.loc[s, "sex"] if "sex" in meta.columns else ""}

    for i, a in enumerate(cohorts):
        for b in cohorts[i:]:
            within = a == b
            if within:
                Xa = data[a][0].to_numpy(float)
                Rz = row_corr(zscore_genes(Xa), zscore_genes(Xa))
                Rr = row_corr(Xa, Xa)
                genes_used = Xa.shape[1]
                iu = np.triu_indices(len(Xa), 1)
                rz_vals, r_vals = Rz[iu], Rr[iu]
                idx = list(zip(*iu))
            else:
                Za = zscore_genes(data[a][0][shared].to_numpy(float))
                Zb = zscore_genes(data[b][0][shared].to_numpy(float))
                Rz = row_corr(Za, Zb)
                genes_used = len(shared)
                ii, jj = np.indices(Rz.shape)
                rz_vals, r_vals = Rz.ravel(), np.full(Rz.size, np.nan)
                idx = list(zip(ii.ravel(), jj.ravel()))
            sa, sb = data[a][0].index, data[b][0].index
            hit = (rz_vals >= threshold) | (np.nan_to_num(r_vals, nan=-1) >= raw_threshold)
            for k in np.flatnonzero(hit):
                p, q = idx[k]
                ia, ib = info(a, sa[p]), info(b, sb[q])
                pairs.append({"cohort_a": a, "sample_a": sa[p], "cohort_b": b, "sample_b": sb[q], "r_z": round(float(rz_vals[k]), 4),
                              "r_raw": None if np.isnan(r_vals[k]) else round(float(r_vals[k]), 4), "genes_used": genes_used,
                              "label_a": ia["label"], "label_b": ib["label"], "sex_metadata_a": ia["sex_metadata"],
                              "sex_metadata_b": ib["sex_metadata"], "same_label": ia["label"] == ib["label"]})
            summary.append({"cohort_a": a, "cohort_b": b, "scope": "within" if within else "between", "genes_used": genes_used,
                            "n_pairs": int(rz_vals.size), "rz_max": round(float(rz_vals.max()), 4),
                            "rz_q999": round(float(np.quantile(rz_vals, 0.999)), 4), "rz_median": round(float(np.median(rz_vals)), 4),
                            "r_raw_max": None if within is False else round(float(np.nanmax(r_vals)), 4),
                            f"n_rz_ge_{threshold}": int((rz_vals >= threshold).sum()),
                            f"n_r_ge_{raw_threshold}": None if not within else int((r_vals >= raw_threshold).sum())})
    cols = ["cohort_a", "sample_a", "cohort_b", "sample_b", "r_z", "r_raw", "genes_used", "label_a", "label_b",
            "sex_metadata_a", "sex_metadata_b", "same_label"]
    return pd.DataFrame(pairs, columns=cols).sort_values("r_z", ascending=False), pd.DataFrame(summary)


# --------------------------------------------------------------------------------------------- fetch-tcga-sex
def fetch_tcga_sex(data_root: Path, prov: dict) -> pd.DataFrame:
    import requests
    d = resolve_dataset_dir(data_root, TCGA)
    meta = read_metadata_text(d / "metadata.csv")
    patients = sorted(set(meta["patient_id"]))
    payload = {"filters": {"op": "in", "content": {"field": "submitter_id", "value": patients}},
               "fields": "submitter_id,case_id,project.project_id,demographic.gender,demographic.sex_at_birth",
               "format": "JSON", "size": len(patients) + 100}
    r = requests.post(f"{GDC_API}/cases", json=payload, timeout=180)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    def demo(h):
        v = h.get("demographic") or {}
        return v[0] if isinstance(v, list) and v else (v if isinstance(v, dict) else {})
    rows = [{"patient_id": h["submitter_id"], "case_id": h.get("case_id"), "project": (h.get("project") or {}).get("project_id"),
             "gender_raw": demo(h).get("gender"), "sex_at_birth_raw": demo(h).get("sex_at_birth")} for h in hits]
    table = pd.DataFrame(rows, columns=["patient_id", "case_id", "project", "gender_raw", "sex_at_birth_raw"]).drop_duplicates("patient_id")
    norm = lambda col: table[col].fillna("").astype(str).str.strip().str.lower().map(SEX_MAP).fillna("")
    g, b = norm("gender_raw"), norm("sex_at_birth_raw")
    table["sex"] = g.where(g != "", b)
    table["sex_conflict"] = (g != "") & (b != "") & (g != b)
    if (table["sex"] == "").all():
        print("Ví dụ bản ghi GDC:", json.dumps(hits[:2], ensure_ascii=False)[:800])
        raise SystemExit("GDC không trả về giới tính cho bệnh nhân nào — KHÔNG ghi sex_gdc.csv; báo lại để sửa trường truy vấn")
    missing = sorted(set(patients) - set(table.patient_id))
    table = pd.concat([table, pd.DataFrame({"patient_id": missing, "sex": ""})], ignore_index=True).sort_values("patient_id")
    table.to_csv(d / "sex_gdc.csv", index=False, encoding="utf-8")
    source = {"api": f"POST {GDC_API}/cases", "fields": payload["fields"], "n_patients": len(patients), "n_found": len(hits),
              "n_missing": len(missing), "missing": missing, "sex_counts": table["sex"].replace("", "unknown").value_counts().to_dict(),
              "sha256": sha256(d / "sex_gdc.csv"), **prov, "run_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    (d / "sex_gdc_source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{TCGA}] GDC: {len(hits)}/{len(patients)} bệnh nhân; sex = {source['sex_counts']}; "
          f"gender/sex_at_birth mâu thuẫn: {int(table['sex_conflict'].sum())}; thiếu: {missing or '—'}")
    return table


# --------------------------------------------------------------------------------------------- apply-sex-qc
REASON = {
    "opposite": "QĐ22 sex check: opposite — XIST và gen Y đều trái metadata (meta_sex = {meta_sex}, biểu hiện = {category}); "
                "nhãn mô học cùng dòng metadata nên không tin được",
    "inconsistent": "QĐ22 sex check: inconsistent — XIST và gen Y cùng cao (meta_sex = {meta_sex}); không xác minh được danh tính mẫu",
}


def apply_sex_qc(data_root: Path, qc_dir: Path, cohorts) -> list[dict]:
    out = []
    for c in cohorts:
        path = qc_dir / f"sex_check_{c}.csv"
        if not path.exists():
            print(f"[{c}] không có {path} — bỏ qua")
            continue
        t = pd.read_csv(path, dtype={"sample_id": str})
        q = t[t["action"] == "quarantine"]
        d = resolve_dataset_dir(data_root, c)
        excl_path = d / f"excluded_samples_{c}.csv"
        excl = pd.read_csv(excl_path, dtype=str, keep_default_na=False) if excl_path.exists() else pd.DataFrame(columns=["sample_id", "reason", "decision"])
        # các dòng QĐ22 cũ được tính lại từ sex_check hiện tại (chạy lại sau khi sửa metadata không để lại lý do sai)
        excl = excl[~excl.get("decision", pd.Series("", index=excl.index)).astype(str).str.startswith("QĐ22")]
        have = set(excl["sample_id"])
        new = [{"sample_id": r.sample_id, "reason": REASON[r.verdict].format(meta_sex=r.meta_sex if isinstance(r.meta_sex, str) else "không có",
                                                                             category=r.category),
                "decision": "QĐ22 2026-10-29", "verdict": r.verdict, "category": r.category} for r in q.itertuples() if r.sample_id not in have]
        merged = pd.concat([excl, pd.DataFrame(new)], ignore_index=True).fillna("")
        if len(merged) or excl_path.exists():
            merged.to_csv(excl_path, index=False, encoding="utf-8")
        flagged = t[t["verdict"] == "inconclusive"]["sample_id"].tolist()
        print(f"[{c}] cách ly mới: {[n['sample_id'] for n in new] or '—'}; tổng trong {excl_path.name}: {len(merged)}; "
              f"inconclusive (giữ, ghi cờ data card): {len(flagged)}")
        out.append({"cohort": c, "added": [n["sample_id"] for n in new], "n_excluded_total": len(merged), "inconclusive": flagged})
    return out


# --------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["rebuild", "sex-check", "rz-screen", "fetch-tcga-sex", "apply-sex-qc"])
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--cohorts", nargs="+", default=list(GEO))
    ap.add_argument("--out", type=Path, default=None, help="thư mục ghi kết quả kiểm tra (sex-check, rz-screen)")
    ap.add_argument("--unclear", type=float, default=2.0, help="|sex_score| < ngưỡng này: 'unclear' (mặc định 2)")
    ap.add_argument("--threshold", type=float, default=0.90, help="ngưỡng r_z (mặc định 0,90)")
    ap.add_argument("--raw-threshold", type=float, default=0.99, help="ngưỡng r thô trong cohort (mặc định 0,99)")
    ap.add_argument("--dry-run", action="store_true", help="rebuild: chỉ in kế hoạch, không ghi")
    ap.add_argument("--sex-only", action="store_true", help="rebuild: chỉ thêm cột sex, không bỏ mẫu (trước check_sex_consistency.py)")
    ap.add_argument("--qc-dir", type=Path, default=None, help="apply-sex-qc: thư mục sex_check_<bộ>.csv (mặc định $KLTN_DATA_ROOT/qc)")
    args = ap.parse_args(argv)
    root = args.data_root or (Path(os.environ["KLTN_DATA_ROOT"]) if os.environ.get("KLTN_DATA_ROOT") else None)
    if root is None:
        ap.error("đặt KLTN_DATA_ROOT hoặc --data-root")

    if args.command == "fetch-tcga-sex":
        fetch_tcga_sex(root, require_traceable())
        return
    if args.command == "apply-sex-qc":
        cohorts = args.cohorts if args.cohorts != list(GEO) else list(C.DATASETS)
        apply_sex_qc(root, args.qc_dir or (root / "qc"), cohorts)
        return
    if args.command == "rebuild":
        prov = git_state() if args.dry_run else require_traceable()
        for c in args.cohorts:
            log = rebuild(root, c, prov, dry_run=args.dry_run, sex_only=args.sex_only)
            exp = log["n_per_class_expected"]
            flag = "" if not exp or exp == log["n_per_class_after"] else f"   <-- khác DATASET_EXPECTED_N {tuple(exp)}"
            print(f"[{c}] {log['n_before']} -> {log['n_after']} mẫu, LUAD/LUSC = {tuple(log['n_per_class_after'])}{flag}; "
                  f"bỏ {log['excluded_now'] or '—'}; sex: {log['sex_column']} {log['sex_counts']}; "
                  f"{'ĐÃ GHI' if log['changed'] and not args.dry_run else 'không đổi' if not log['changed'] else 'dry-run'}")
            for f in ("expression.h5", "metadata.csv"):
                print(f"    {f}: {log['sha256_before'][f]} -> {log['sha256_after'][f]}")
        return

    out = args.out or (root / "_kiem_tra")
    out.mkdir(parents=True, exist_ok=True)
    stamp = {**git_state(), "run_at": datetime.now().astimezone().isoformat(timespec="seconds"), "data_root": str(root)}
    if args.command == "sex-check":
        rows = []
        for c in args.cohorts:
            t = sex_check(root, c, args.unclear)
            t.to_csv(out / f"sex_check_{c}.csv", encoding="utf-8")
            counts = t["status"].value_counts().to_dict()
            rows.append({"cohort": c, "n": len(t), **counts})
            print(f"[{c}] n = {len(t)}: {counts}")
            bad = t[t.status == "mismatch"]
            if len(bad):
                print("    MISMATCH (báo GVHD, không sửa nhãn):\n" + bad[["label", "XIST", "sex_score", "sex_inferred", "sex_metadata"]].to_string())
        summary = pd.DataFrame(rows).fillna(0)
        summary.to_csv(out / "sex_check_summary.csv", index=False)
        (out / "sex_check_config.json").write_text(json.dumps({**stamp, "unclear_abs_sex_score": args.unclear,
                                                               "female_genes": FEMALE, "male_genes": MALE,
                                                               "sex_score": "mean(male genes) - XIST"}, indent=2) + "\n", encoding="utf-8")
    else:
        pairs, summary = rz_screen(root, list(args.cohorts), args.threshold, args.raw_threshold)
        pairs.to_csv(out / "rz_screen_pairs.csv", index=False)
        summary.to_csv(out / "rz_screen_summary.csv", index=False)
        (out / "rz_screen_config.json").write_text(json.dumps({**stamp, "rz_threshold": args.threshold,
                                                               "raw_threshold": args.raw_threshold, "cohorts": args.cohorts,
                                                               "r_z": "Pearson between samples after z-scoring each gene within its cohort; between cohorts on genes shared by all cohorts"},
                                                              indent=2) + "\n", encoding="utf-8")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(summary.to_string(index=False))
            print(f"\ncặp vượt ngưỡng (r_z >= {args.threshold} hoặc r >= {args.raw_threshold}): {len(pairs)}")
            if len(pairs):
                print(pairs.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
