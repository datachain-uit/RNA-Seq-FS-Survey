"""Va nhan giai doan (va primary_diagnosis) trong metadata cua bo v2 tcga_luad_lusc_primary_tumor.

Loi goc: notebooks/phase_0/00_data/lung_datasets_v2_patient_folds.ipynb chi doc diagnoses[0]
cua moi case, trong khi GDC tra ve nhieu ban ghi diagnoses (u chi diem, Prior primary,
Synchronous primary, metastasis ...) theo thu tu khong co dinh -> 121 LUAD / 85 LUSC mat giai doan.

Script nay:
  A1. POST GDC /cases, lay MOI ban ghi diagnoses, luu JSON tho, chon giai doan theo quy tac co dinh.
  A2. Them cot vao metadata.csv (stage_raw cu -> stage_raw_v2_original, primary_diagnosis cu ->
      primary_diagnosis_v2_original).
  A3. Kiem tra: khop v1, so ca theo stage_group, ca con thieu, checksum file bieu hien / fold.
  A4. Cap nhat DATA_CARD.md cua tcga_luad_lusc_primary_tumor.
  A5. Ghi commit hash vao config.json + DATA_CARD.md cua ca 3 bo v2.
  A6. Chep metadata.csv cua ca 3 bo vao experiment-data-v2/h5/<bo>/.

Khong sua expression.h5, fold_assignment.csv, folds/ hay h5/. Chay lai nhieu lan cho ket qua nhu nhau.

    python scripts/patch_v2_tcga_stage.py                 # commit lay tu git qua scripts/provenance.py
    KLTN_ALLOW_UNTRACKED=1 python scripts/patch_v2_tcga_stage.py  # chay tam khi chua co git (commit_hash = null)
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

try:  # import khi chạy test (scripts.<x>) hoặc chạy trực tiếp (python scripts/<x>.py)
    from scripts.provenance import DATA_ROOT, require
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import DATA_ROOT, require

REPO = DATA_ROOT  # thu muc du lieu (KLTN_DATA_ROOT); code nam trong kho Git
API = "https://api.gdc.cancer.gov"
TARGET = "tcga_luad_lusc_primary_tumor"
DATASETS = ["gse68465_stage_tumor_only", TARGET, "tcga_normal_vs_tumor_sanity"]

# Ten truong da doi chieu voi GDC Data Dictionary (node `diagnosis`) va GET /cases/_mapping
DIAG_FIELDS = ["diagnosis_id", "submitter_id", "ajcc_pathologic_stage", "ajcc_staging_system_edition",
               "ajcc_pathologic_t", "ajcc_pathologic_n", "ajcc_pathologic_m", "primary_diagnosis",
               "diagnosis_is_primary_disease", "classification_of_tumor"]
CASE_FIELDS = ["case_id", "submitter_id", "project.project_id"] + [f"diagnoses.{f}" for f in DIAG_FIELDS]

STAGE_GROUPS = ["I", "II", "III", "IV"]
STAGE_RE = re.compile(r"^Stage (IV|III|II|I)[ABC]?$")   # Stage IA/IB -> I, IIIC -> III, IVB -> IV
NEW_COLS = ["stage_raw", "stage_group", "ajcc_edition", "path_t", "path_n", "path_m", "stage_conflict",
            "stage_rule", "stage_diagnosis_submitter_id", "primary_diagnosis", "diagnosis_mismatch"]

# So khoi u co stage_group mong doi (GDC Data Release 46.0), dem theo quy tac cua v1 (ban ghi dau tien)
EXPECTED = {"TCGA-LUAD": {"I": 275, "II": 123, "III": 85, "IV": 26},
            "TCGA-LUSC": {"I": 243, "II": 161, "III": 86, "IV": 7}}
EXPECTED_MISSING = {"TCGA-LUAD": 8, "TCGA-LUSC": 4}

RULE = ("ung vien = ban ghi diagnoses co ajcc_pathologic_stage khac rong/'Not Reported'; khong co -> thieu; "
        "cung mot gia tri -> lay; khac nhau -> stage_conflict=True, lay ban ghi diagnosis_is_primary_disease=True "
        "(u chi diem) neu co dung mot")
CARD_START = "<!-- patch_v2_tcga_stage:start -->"
CARD_END = "<!-- patch_v2_tcga_stage:end -->"


# ---------------------------------------------------------------- tien ich
def die(msg):
    sys.exit(f"\nLOI: {msg}")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def protected_files(v2):
    # File khong duoc thay doi: bieu hien, fold, moi file trong h5/ (tru metadata.csv do A6 chep vao)
    files = []
    for name in DATASETS:
        d = v2 / "datasets" / name
        files += [d / "expression.h5", d / "fold_assignment.csv"]
        files += sorted(p for p in (d / "folds").rglob("*") if p.is_file())
    files += sorted(p for p in (v2 / "h5").rglob("*") if p.is_file() and p.name != "metadata.csv")
    return files


def checksums(v2):
    return {str(p.relative_to(v2)): sha256(p) for p in protected_files(v2)}


def md_table(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def resolve_commit(arg):
    """Tra ve (commit hien thi, nguon, provenance). Dung neu khong truy vet duoc (scripts/provenance.py)."""
    if arg:
        die("--commit nhap tay da bo (khong kiem chung duoc). Chay trong git clone cua kho; neu buoc phai chay tam, dat KLTN_ALLOW_UNTRACKED=1.")
    prov = require(__file__)
    if prov["commit_hash"] is None:
        return "untracked", "khong phai git clone (KLTN_ALLOW_UNTRACKED=1), scripts/patch_v2_tcga_stage.py", prov
    return prov["commit_hash"], "git rev-parse HEAD (provenance), scripts/patch_v2_tcga_stage.py", prov


# ---------------------------------------------------------------- A1: GDC
def gdc(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, **kw)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError as ex:
        die(f"Khong ket noi duoc GDC API ({API}). Kiem tra mang roi chay lai. Chi tiet: {ex}")
    except requests.exceptions.RequestException as ex:
        die(f"Goi GDC API {path} that bai: {ex}")


def check_fields():
    mapping = set(gdc("GET", "/cases/_mapping")["fields"])
    missing = [f for f in CASE_FIELDS if f not in mapping]
    if missing:
        die(f"Truong khong co trong GDC /cases/_mapping: {missing}")


def fetch_cases(patient_ids):
    payload = {"filters": {"op": "in", "content": {"field": "submitter_id", "value": sorted(patient_ids)}},
               "fields": ",".join(CASE_FIELDS), "format": "JSON", "size": len(patient_ids) + 100}
    res = gdc("POST", "/cases", json=payload)
    pg = res["data"]["pagination"]
    if pg["total"] > pg["count"]:
        die(f"GDC tra ve {pg['count']}/{pg['total']} case -- tang size")
    return payload, res


def has_stage(d):
    s = d.get("ajcc_pathologic_stage")
    return s is not None and str(s).strip() not in ("", "Not Reported")


def select_diagnosis(diagnoses):
    """Quy tac co dinh (giong cell 13 cua lung_datasets_v2_patient_folds.ipynb).

    1. Ung vien = ban ghi co ajcc_pathologic_stage khac rong va khac 'Not Reported'.
    2. Khong co ung vien -> khong co giai doan (rule 'none').
    3. Moi ung vien cung mot gia tri -> lay gia tri do; neu nhieu ban ghi thi uu tien ban ghi
       u chi diem (diagnosis_is_primary_disease = True) de lay T/N/M/edition (rule 'single').
    4. Ung vien co gia tri khac nhau -> stage_conflict = True; lay ban ghi u chi diem neu co DUNG
       mot ung vien nhu vay (rule 'conflict_index_tumor'), nguoc lai bo trong (rule 'conflict_unresolved').
    Tra ve (ban ghi duoc chon hoac None, stage_conflict, rule).
    """
    cands = [d for d in diagnoses if has_stage(d)]
    if not cands:
        return None, False, "none"
    index = [d for d in cands if d.get("diagnosis_is_primary_disease") is True]
    if len({d["ajcc_pathologic_stage"] for d in cands}) == 1:
        return (index or cands)[0], False, "single"
    if len(index) == 1:
        return index[0], True, "conflict_index_tumor"
    return None, True, "conflict_unresolved"


def index_tumor_diagnosis(diagnoses):
    index = [d for d in diagnoses if d.get("diagnosis_is_primary_disease") is True]
    return index[0] if len(index) == 1 else None


def stage_group(stage):
    if stage is None or pd.isna(stage):
        return None
    m = STAGE_RE.match(str(stage).strip())
    return m.group(1) if m else None


def diagnosis_mismatch(project, diag):
    # LUAD ma chan doan la ung thu te bao vay, hoac LUSC ma chan doan khong phai te bao vay
    if diag is None or pd.isna(diag):
        return False
    squamous = "squamous" in diag.lower() and "adenosquamous" not in diag.lower()
    return squamous if project == "TCGA-LUAD" else not squamous


def build_stage_table(hits):
    rows = []
    for h in hits:
        diags = h.get("diagnoses") or []
        chosen, conflict, rule = select_diagnosis(diags)
        idx = index_tumor_diagnosis(diags)
        c = chosen or {}
        pdx = (idx or {}).get("primary_diagnosis")
        rows.append({
            "patient_id": h["submitter_id"],
            "gdc_project": h["project"]["project_id"],
            "stage_raw": c.get("ajcc_pathologic_stage"),
            "stage_group": stage_group(c.get("ajcc_pathologic_stage")),
            "ajcc_edition": c.get("ajcc_staging_system_edition"),
            "path_t": c.get("ajcc_pathologic_t"),
            "path_n": c.get("ajcc_pathologic_n"),
            "path_m": c.get("ajcc_pathologic_m"),
            "stage_conflict": conflict,
            "stage_rule": rule,
            "stage_diagnosis_submitter_id": c.get("submitter_id"),
            "primary_diagnosis": pdx,
            "diagnosis_mismatch": diagnosis_mismatch(h["project"]["project_id"], pdx),
        })
    return pd.DataFrame(rows).set_index("patient_id")


# ---------------------------------------------------------------- A3: kiem tra
def load_v1(v1_root):
    v1 = pd.concat([pd.read_csv(v1_root / f"tcga_{p}" / "fold_assignment.csv") for p in ("luad", "lusc")])
    v1 = v1[v1["stage"].isin(STAGE_GROUPS)][["patient_id", "stage"]].drop_duplicates()
    assert v1["patient_id"].is_unique, "v1: mot benh nhan co nhieu giai doan"
    return v1.set_index("patient_id")["stage"]


def check_v1(meta, v1):
    both = meta[meta["stage_group"].notna()].join(v1.rename("stage_v1"), on="patient_id", how="inner")
    both = both[both["stage_v1"].notna()]
    plain, confl = both[~both["stage_conflict"]], both[both["stage_conflict"]]
    diff = plain[plain["stage_group"] != plain["stage_v1"]]
    print(f"\n[A3] Doi chieu v1: {len(both)} benh nhan co giai doan o ca hai ban "
          f"({len(plain)} khong conflict, {len(confl)} stage_conflict)")
    if len(diff):
        print(diff[["patient_id", "project", "stage_raw", "stage_group", "stage_v1"]].to_string())
        die(f"{len(diff)} ca khong conflict lech giai doan voi v1")
    print(f"  ca khong conflict: stage_group trung v1 100% ({len(plain)}/{len(plain)})")
    cdiff = confl[confl["stage_group"] != confl["stage_v1"]]
    print(f"  ca stage_conflict: {len(confl) - len(cdiff)}/{len(confl)} trung v1; lech (v1 lay ban ghi dau tien, "
          f"ban va lay u chi diem):")
    print(cdiff[["patient_id", "project", "stage_raw", "stage_group", "stage_v1"]].to_string(index=False))
    return {"n_both": len(both), "n_plain": len(plain), "n_conflict": len(confl), "conflict_diff": cdiff}


def group_counts(meta):
    return {p: meta[meta["project"] == p]["stage_group"].value_counts().reindex(STAGE_GROUPS, fill_value=0)
            .astype(int).to_dict() for p in EXPECTED}


def fmt_counts(counts):
    return ", ".join(f"{p} {sum(v.values())} ({'/'.join(map(str, v.values()))})" for p, v in counts.items())


def check_counts(meta, conflict_diff):
    # EXPECTED dem theo v1 (ban ghi dau tien). Thay stage_group cua cac ca conflict lech v1 bang gia tri v1
    # phai ra dung EXPECTED -> moi chenh lech chi do quy tac chon u chi diem.
    tab = pd.crosstab(meta["project"], meta["stage_group"].fillna("missing"), margins=True, margins_name="TONG")
    print("\n[A3] project x stage_group:\n" + tab.to_string())
    got = group_counts(meta)
    as_v1 = meta.copy()
    as_v1.loc[as_v1["patient_id"].isin(conflict_diff["patient_id"]), "stage_group"] = \
        as_v1["patient_id"].map(conflict_diff.set_index("patient_id")["stage_v1"])
    got_v1 = group_counts(as_v1)
    miss = meta[meta["stage_group"].isna()]["project"].value_counts().reindex(list(EXPECTED), fill_value=0).to_dict()
    if got_v1 != EXPECTED or miss != EXPECTED_MISSING:
        die(f"So ca theo stage_group khac mong doi.\n  thu duoc (quy tac v1 cho ca conflict): {got_v1} | "
            f"thieu {miss}\n  mong doi: {EXPECTED} | thieu {EXPECTED_MISSING}")
    print(f"  ban va (u chi diem):          {fmt_counts(got)}")
    print(f"  neu ca conflict lay theo v1:  {fmt_counts(got_v1)} = mong doi; thieu {miss}")
    return got_v1


def print_missing(meta, hits):
    raw = {h["submitter_id"]: h for h in hits}
    miss = meta[meta["stage_group"].isna()]
    print(f"\n[A3] {len(miss)} ca van thieu giai doan:", miss.groupby("project").size().to_dict())
    for pid, proj, rule in miss[["patient_id", "project", "stage_rule"]].itertuples(index=False):
        print(f"\n--- {pid} ({proj}, rule={rule})")
        print(json.dumps(raw[pid].get("diagnoses", []), ensure_ascii=False, indent=1))
    return miss


# ---------------------------------------------------------------- A2: metadata
def patch_metadata(md, stage):
    md = md.copy()
    # Chay lai: cot goc da doi ten tu lan truoc -> bo cac cot moi roi tinh lai
    for old in ("stage_raw", "primary_diagnosis"):
        orig = f"{old}_v2_original"
        if orig in md.columns:
            md = md.drop(columns=[c for c in NEW_COLS if c in md.columns])
        else:
            md = md.rename(columns={old: orig})
    assert set(md["patient_id"]) == set(stage.index), "patient_id cua metadata va GDC khac nhau"
    bad = stage.loc[md["patient_id"], "gdc_project"].values != md["project"].values
    assert not bad.any(), "project cua GDC khac metadata"
    new = stage.loc[md["patient_id"], NEW_COLS].reset_index(drop=True)
    new.index = md.index
    md = pd.concat([md, new], axis=1)
    md["stage_conflict"] = md["stage_conflict"].astype(bool)
    md["diagnosis_mismatch"] = md["diagnosis_mismatch"].astype(bool)
    return md


# ---------------------------------------------------------------- A4/A5: data card + config
def stage_section(meta, ctx):
    ed = meta.assign(stage_group=meta["stage_group"].fillna("thiếu"),
                     ajcc_edition=meta["ajcc_edition"].fillna("không ghi"))
    tab = pd.crosstab([ed["project"], ed["stage_group"]], ed["ajcc_edition"], margins=True, margins_name="tổng")
    tab = tab.reset_index()
    grp = pd.crosstab(meta["project"], meta["stage_group"].fillna("thiếu"), margins=True, margins_name="tổng")
    grp = grp.reset_index()
    miss = meta[meta["stage_group"].isna()]
    confl = meta[meta["stage_conflict"]]
    pdx = meta.groupby(["project", "primary_diagnosis"]).size().rename("n").reset_index()
    mism = meta[meta["diagnosis_mismatch"]]
    old_mism = sum(diagnosis_mismatch(p, d) for p, d in zip(meta["project"], meta["primary_diagnosis_v2_original"]))
    old_nr = int((meta["primary_diagnosis_v2_original"] == "Not Reported").sum())
    cd = ctx["v1"]["conflict_diff"]
    cdiff_txt = ", ".join(f"`{r.patient_id}` ({r.stage_group} vs v1 {r.stage_v1})" for r in cd.itertuples())
    lines = [
        "## Nhãn giai đoạn (cho bài toán phụ)",
        "",
        CARD_START,
        f"Metadata được vá bằng `scripts/patch_v2_tcga_stage.py` tại commit `{ctx['commit']}`; "
        "ma trận biểu hiện và fold không đổi (sha256 của `expression.h5`, `fold_assignment.csv`, "
        "`folds/` và `experiment-data-v2/h5/` giống hệt trước/sau khi chạy script).",
        "",
        "**Lỗi được vá.** Bản v2 gốc chỉ đọc `diagnoses[0]` của mỗi case. GDC trả về nhiều bản ghi "
        "`diagnoses` cho một case (u chỉ điểm, *Prior primary*, *Synchronous primary*, *metastasis*...) theo "
        "thứ tự không cố định, nên 121 LUAD / 85 LUSC bị mất giai đoạn và `primary_diagnosis` có khi lấy từ "
        "khối u khác. Cột cũ được giữ nguyên dưới tên `stage_raw_v2_original` và `primary_diagnosis_v2_original` "
        "(không dùng cho phân tích).",
        "",
        f"- Nguồn: GDC API `POST {API}/cases`, lọc `submitter_id` ∈ {len(meta)} `patient_id` của bộ này; lấy "
        "mọi bản ghi `diagnoses` với các trường `ajcc_pathologic_stage`, `ajcc_staging_system_edition`, "
        "`ajcc_pathologic_t/n/m`, `primary_diagnosis`, `diagnosis_is_primary_disease`, `classification_of_tumor` "
        "(tên trường đối chiếu GDC Data Dictionary, node `diagnosis`, và `GET /cases/_mapping`).",
        f"- Ngày truy vấn (UTC): {ctx['queried_at']}; GDC {ctx['release']}.",
        "- JSON thô: `gdc_diagnoses_raw.json` (cùng thư mục).",
        "- Quy tắc chọn giai đoạn cho mỗi bệnh nhân:",
        "  1. Ứng viên = bản ghi có `ajcc_pathologic_stage` khác rỗng và khác `Not Reported`.",
        "  2. Không có ứng viên → không có giai đoạn.",
        "  3. Mọi ứng viên cùng giá trị → lấy giá trị đó (`T/N/M`, `edition` lấy từ bản ghi u chỉ điểm nếu có).",
        "  4. Ứng viên có giá trị khác nhau → `stage_conflict = True`; lấy bản ghi u chỉ điểm "
        "(`diagnosis_is_primary_disease = True`, `classification_of_tumor = primary`) nếu có đúng một. "
        "Các bản ghi còn lại là khối u khác của cùng bệnh nhân (*Prior/Synchronous primary*).",
        "- `stage_group`: `Stage IA/IB/I` → I, `IIA/IIB/II` → II, `IIIA/IIIB/IIIC/III` → III, `IV/IVA/IVB` → IV.",
        "- Cột mới trong `metadata.csv`: `stage_raw`, `stage_group`, `ajcc_edition`, `path_t`, `path_n`, `path_m`, "
        "`stage_conflict`, `stage_rule` (`single` / `conflict_index_tumor` / `none`), "
        "`stage_diagnosis_submitter_id`, `primary_diagnosis` (bản ghi u chỉ điểm), `diagnosis_mismatch`.",
        "",
        "Số khối u theo giai đoạn:",
        "",
        md_table(grp),
        "",
        "Theo phiên bản AJCC (`ajcc_edition`):",
        "",
        md_table(tab),
        "",
        f"- Thiếu giai đoạn: {len(miss)} ca ("
        + ", ".join(f"{p} {n}" for p, n in miss.groupby("project").size().items())
        + "): " + ", ".join(f"`{x}`" for x in miss["patient_id"]) + ". Không bản ghi `diagnoses` nào có giai đoạn.",
        f"- `stage_conflict = True`: {len(confl)} ca ("
        + ", ".join(f"{p} {n}" for p, n in confl.groupby("project").size().items())
        + f"); {int(confl['stage_group'].notna().sum())} ca đã lấy giai đoạn của u "
        "chỉ điểm. Nên chạy lại phân tích giai đoạn khi bỏ các ca này (phân tích độ nhạy).",
        f"- Đối chiếu v1 (`experiment-data/h5/tcga_{{luad,lusc}}/fold_assignment.csv`): {ctx['v1']['n_both']} bệnh "
        f"nhân có giai đoạn ở cả hai bản; {ctx['v1']['n_plain']} ca không conflict trùng `stage_group` 100%. "
        f"Trong {ctx['v1']['n_conflict']} ca conflict, {len(cd)} ca lệch v1 vì v1 lấy bản ghi đầu tiên còn bản vá "
        f"lấy u chỉ điểm: {cdiff_txt}. Nếu 7 ca này lấy theo v1, số ca theo giai đoạn trùng đúng v1 "
        f"({fmt_counts(ctx['counts_v1'])}).",
        "",
        "**`primary_diagnosis` (bản ghi u chỉ điểm) theo project:**",
        "",
        md_table(pdx),
        "",
        f"- Không khớp project (`diagnosis_mismatch = True`: LUAD mà chẩn đoán tế bào vảy, hoặc LUSC mà chẩn đoán "
        f"không phải tế bào vảy): {len(mism)} ca"
        + (": " + ", ".join(f"`{r.patient_id}` ({r.project}, {r.primary_diagnosis})" for r in mism.itertuples())
           if len(mism) else "")
        + ". Dùng cho phân tích độ nhạy (bỏ các ca này rồi đánh giá lại LUAD vs LUSC).",
        f"- Với cột cũ `primary_diagnosis_v2_original` (`diagnoses[0]`) sẽ có {old_mism} ca \"không khớp\" và "
        f"{old_nr} ca `Not Reported` — phần lớn là chẩn đoán của khối u khác (vd \"Squamous cell carcinoma, NOS\" "
        "trong TCGA-LUAD), không phải lỗi nhãn.",
        CARD_END,
    ]
    return "\n".join(lines)


def update_card(path, commit, source, section=None):
    text = path.read_text(encoding="utf-8")
    text, n = re.subn(r"^- Commit: .*? — notebook", f"- Commit: `{commit}` ({source}) — notebook", text,
                      count=1, flags=re.M)
    assert n == 1, f"{path}: khong tim thay dong '- Commit:'"
    if section is not None:
        header = "## Nhãn giai đoạn (cho bài toán phụ)"
        if CARD_START in text:
            pat = re.compile(re.escape(header) + r"\n\n" + re.escape(CARD_START) + ".*?" + re.escape(CARD_END), re.S)
            text, n = pat.subn(lambda _: section, text)
            assert n == 1, f"{path}: muc giai doan cu bi hong"
        else:
            assert "\n## Tệp\n" in text, f"{path}: khong tim thay muc '## Tệp'"
            text = text.replace("\n## Tệp\n", "\n" + section + "\n\n## Tệp\n", 1)
    path.write_text(text, encoding="utf-8")


def update_config(path, commit, source, extra=None, prov=None):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["commit_hash"], cfg["commit_source"] = commit, source
    if prov is not None:
        cfg["commit_hash"] = prov["commit_hash"]          # null khi chay khong truy vet
        cfg["provenance"] = prov
    if extra:
        cfg.update(extra)
        cfg["files"] = sorted(set(cfg.get("files", [])) | {"gdc_diagnoses_raw.json"})
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", help="DA BO: commit lay tu git qua provenance()")
    ap.add_argument("--v2", type=Path, default=REPO / "experiment-data-v2")
    ap.add_argument("--v1", type=Path, default=REPO / "experiment-data" / "h5")
    args = ap.parse_args()

    commit, commit_source, prov = resolve_commit(args.commit)
    v2 = args.v2.resolve()
    ds = v2 / "datasets" / TARGET
    print(f"Commit: {commit} ({commit_source})\nv2: {v2}\nv1: {args.v1.resolve()}")

    print("\nTinh sha256 file bieu hien / fold / h5 ...")
    before = checksums(v2)
    print(f"  {len(before)} file")

    # A1
    md = pd.read_csv(ds / "metadata.csv")
    assert md["patient_id"].is_unique
    status = gdc("GET", "/status")
    check_fields()
    queried_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload, res = fetch_cases(md["patient_id"].tolist())
    hits = sorted(res["data"]["hits"], key=lambda h: h["submitter_id"])
    print(f"\n[A1] GDC {status.get('data_release')}: {len(hits)} case, "
          f"{sum(len(h.get('diagnoses') or []) for h in hits)} ban ghi diagnoses")
    stage = build_stage_table(hits)
    unknown = stage[stage["stage_raw"].notna() & stage["stage_group"].isna()]
    if len(unknown):
        die(f"Gia tri ajcc_pathologic_stage khong anh xa duoc ve I-IV:\n{unknown[['stage_raw']]}")
    if (stage["stage_rule"] == "conflict_unresolved").any():
        die("Co ca conflict khong co dung 1 ban ghi u chi diem:\n"
            + stage[stage["stage_rule"] == "conflict_unresolved"].to_string())
    print("  stage_rule:", stage["stage_rule"].value_counts().to_dict())

    # A2 (trong bo nho) + A3 (truoc khi ghi)
    new_md = patch_metadata(md, stage)
    v1_info = check_v1(new_md, load_v1(args.v1))
    counts_v1 = check_counts(new_md, v1_info["conflict_diff"])
    print_missing(new_md, hits)
    print(f"\n[A3] stage_conflict: {int(new_md['stage_conflict'].sum())} ca | diagnosis_mismatch: "
          f"{int(new_md['diagnosis_mismatch'].sum())} ca")
    print(new_md[new_md["diagnosis_mismatch"]][["patient_id", "project", "primary_diagnosis"]].to_string(index=False))

    # Ghi
    raw = {"queried_at_utc": queried_at, "endpoint": f"POST {API}/cases", "gdc_status": status,
           "request": payload, "response": res}
    (ds / "gdc_diagnoses_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    new_md.to_csv(ds / "metadata.csv", index=False)

    # A4 + A5
    ctx = {"counts_v1": counts_v1, "commit": commit, "queried_at": queried_at, "release": status.get("data_release", "khong ro"), "v1": v1_info}
    patch_info = {"metadata_patch": {
        "script": "scripts/patch_v2_tcga_stage.py", "commit": commit, "queried_at_utc": queried_at,
        "gdc_data_release": ctx["release"], "endpoint": f"POST {API}/cases",
        "rule": RULE,
        "stage_group_counts": {p: new_md[new_md["project"] == p]["stage_group"].value_counts()
                               .reindex(STAGE_GROUPS, fill_value=0).astype(int).to_dict() for p in EXPECTED},
        "n_missing_stage": int(new_md["stage_group"].isna().sum()),
        "n_stage_conflict": int(new_md["stage_conflict"].sum()),
        "n_diagnosis_mismatch": int(new_md["diagnosis_mismatch"].sum()),
        "unchanged": "expression.h5, fold_assignment.csv, folds/, experiment-data-v2/h5/ (sha256)",
    }}
    for name in DATASETS:
        d = v2 / "datasets" / name
        update_config(d / "config.json", commit, commit_source, patch_info if name == TARGET else None, prov)
        update_card(d / "DATA_CARD.md", commit, commit_source, stage_section(new_md, ctx) if name == TARGET else None)

    # A6
    for name in DATASETS:
        shutil.copy2(v2 / "datasets" / name / "metadata.csv", v2 / "h5" / name / "metadata.csv")
    print(f"\n[A6] Da chep metadata.csv vao {v2 / 'h5'}/<bo>/ cho {len(DATASETS)} bo")

    # A3: checksum sau
    after = checksums(v2)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if changed:
        die(f"File bieu hien / fold bi thay doi: {changed}")
    print(f"\n[A3] sha256 khong doi truoc/sau: {len(after)} file (expression.h5, fold_assignment.csv, folds/, h5/)")
    print("\nXONG.")


if __name__ == "__main__":
    main()
