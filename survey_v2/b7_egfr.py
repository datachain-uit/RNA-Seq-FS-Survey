"""B7 — bài toán khó hơn: đột biến EGFR ở LUAD (QĐ177, QĐ185, QĐ186, QĐ188; Review 12/11 và 13/11).

Hàm thuần (không tải mạng) để dựng nhãn và biểu hiện của hai bộ `tcga_luad_egfr`, `gse31210_egfr`:

* TCGA-LUAD: nhãn từ MAF mở ("Masked Somatic Mutation") của GDC. Đột biến = ít nhất một biến thể hoạt hoá vùng kinase
  (mất đoạn khung đọc exon 19, L858R, L861Q, G719X, S768I, chèn khung đọc exon 20); hoang dại = không có biến thể không đồng
  nghĩa nào của EGFR; mẫu chỉ có biến thể EGFR khác bị loại và ghi lại; mẫu không có MAF bị loại. TP53 (dự phòng): có / không có
  biến thể không đồng nghĩa. Biến thể khử trùng theo (sample_id, Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2)
  trước khi đếm, vì một mẫu có thể nằm trong nhiều tệp MAF (nhiều aliquot).
* GSE31210: mỗi ô `!Sample_characteristics_ch1` đọc theo tiền tố "khoá: giá trị" của CHÍNH ô đó (Y98: các mẫu không có cùng số
  trường nên đọc theo thứ tự dòng làm lệch giá trị). Dương = "EGFR mutation +"; âm = "EGFR/KRAS/ALK -", "KRAS mutation +",
  "ALK-fusion +" (QĐ185); "normal lung" bị loại.
* Biểu hiện GEO: `parse_series_matrix`, `load_platform_annotation_file`, `collapse_probes` chép đúng hàm của
  notebooks/phase_0/00_data/geo_luad_lusc_target_cohorts_5fold.ipynb (chỉ tách phần tải mạng ra ngoài). Thang giá trị quyết định
  trước khi mở dữ liệu (A.2): phân vị 99 % > B7_LINEAR_SCALE_P99 → thang tuyến tính, sàn 1 rồi log2 trước khi gom probe.
"""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

NONSYN = ("Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
          "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site")
ACTIVATING_CLASSES = ("ex19del", "L858R", "L861Q", "G719X", "S768I", "ex20ins")
VARIANT_KEY = ("sample_id", "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2")
MAF_COLUMNS = ("Hugo_Symbol", "Variant_Classification", "Variant_Type", "HGVSp_Short", "Exon_Number", "Protein_position",
               "Tumor_Sample_Barcode", "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2", "FILTER")
EXON19_PROTEIN = (729, 761)       # dùng khi MAF thiếu Exon_Number
EXON20_PROTEIN = (762, 823)


# ------------------------------------------------------------------------------------------------- TCGA-LUAD
def _exon(row) -> int | None:
    e = str(row.get("Exon_Number") or "")
    return int(e.split("/")[0]) if re.match(r"^\d+/", e) else None


def _protein_start(row) -> int | None:
    m = re.match(r"^(\d+)", str(row.get("Protein_position") or ""))
    return int(m.group(1)) if m else None


def activating_class(row) -> str | None:
    """Lớp hoạt hoá của một dòng EGFR không đồng nghĩa, hoặc None (giống hệt quy tắc của notebook bước 0)."""
    hg, vc = str(row.get("HGVSp_Short") or ""), row["Variant_Classification"]
    ex, pos = _exon(row), _protein_start(row)
    if hg == "p.L858R":
        return "L858R"
    if hg == "p.L861Q":
        return "L861Q"
    if re.fullmatch(r"p\.G719[A-Z]", hg):
        return "G719X"
    if hg == "p.S768I":
        return "S768I"
    if vc == "In_Frame_Del" and (ex == 19 or (ex is None and pos is not None and EXON19_PROTEIN[0] <= pos <= EXON19_PROTEIN[1])):
        return "ex19del"
    if vc == "In_Frame_Ins" and (ex == 20 or (ex is None and pos is not None and EXON20_PROTEIN[0] <= pos <= EXON20_PROTEIN[1])):
        return "ex20ins"
    return None


def read_maf(handle, name: str = "x.maf.gz") -> tuple[pd.DataFrame, set]:
    """(các dòng EGFR, TP53 với MAF_COLUMNS có trong tệp, tập sample_id 16 ký tự có trong tệp)."""
    df = pd.read_csv(handle, sep="\t", comment="#", dtype=str, low_memory=False,
                     compression="gzip" if str(name).endswith(".gz") else None)
    samples = set(df["Tumor_Sample_Barcode"].str[:16]) if len(df) else set()
    keep = df[df["Hugo_Symbol"].isin(["EGFR", "TP53"])]
    return keep[[c for c in MAF_COLUMNS if c in keep.columns]].copy(), samples


def dedup_variants(var: pd.DataFrame) -> pd.DataFrame:
    """Khử trùng biến thể theo VARIANT_KEY (giữ dòng đầu; cột maf_file_id gộp mọi tệp chứa biến thể)."""
    var = var.copy()
    var["sample_id"] = var["Tumor_Sample_Barcode"].str[:16]
    key = list(VARIANT_KEY) + ["Hugo_Symbol"]
    if "maf_file_id" in var.columns:
        files = var.groupby(key, dropna=False)["maf_file_id"].agg(lambda s: ";".join(sorted(set(s.dropna()))))
        out = var.drop_duplicates(key).drop(columns="maf_file_id").merge(files.rename("maf_file_id").reset_index(), on=key, how="left")
    else:
        out = var.drop_duplicates(key)
    return out.reset_index(drop=True)


def label_tcga_luad(var: pd.DataFrame, luad: pd.DataFrame, maf_samples: set, maf_files: dict | None = None) -> pd.DataFrame:
    """Nhãn từng mẫu LUAD của cohort. `var` đã khử trùng; `luad` có cột sample_id, patient_id; `maf_files`: sample_id → mã tệp.

    Trả về cột sample_id, patient_id, label (1 đột biến / 0 hoang dại / NaN loại), label_status, egfr_variants,
    activating_classes, tp53_label, maf_file_ids.
    """
    var = var[var.Variant_Classification.isin(NONSYN)].copy()
    egfr = var[var.Hugo_Symbol == "EGFR"].copy()
    egfr["activating"] = egfr.apply(activating_class, axis=1) if len(egfr) else pd.Series(dtype=object)
    tp53 = set(var.loc[var.Hugo_Symbol == "TP53", "sample_id"])
    by_sample = {s: g for s, g in egfr.groupby("sample_id")}
    rows = []
    for sid, pid in zip(luad["sample_id"], luad["patient_id"]):
        g = by_sample.get(sid)
        acts = sorted(set(g["activating"].dropna())) if g is not None else []
        if sid not in maf_samples:
            status = "no_maf"
        elif g is None:
            status = "wildtype"
        else:
            status = "mutant" if acts else "excluded_other_egfr"
        rows.append({"sample_id": sid, "patient_id": pid,
                     "label": {"mutant": 1, "wildtype": 0}.get(status, np.nan), "label_status": status,
                     "egfr_variants": ";".join(g["HGVSp_Short"].fillna("?")) if g is not None else "",
                     "activating_classes": ";".join(acts),
                     "tp53_label": "no_maf" if sid not in maf_samples else ("mutant" if sid in tp53 else "wildtype"),
                     "maf_file_ids": (maf_files or {}).get(sid, "")})
    return pd.DataFrame(rows)


def check_counts(observed: dict, expected: dict, what: str) -> None:
    """Dừng nếu số mẫu từng nhóm khác số chốt của bước 0 (Review 13/11 A.1)."""
    obs = {k: int(observed.get(k, 0)) for k in expected}
    extra = {k: v for k, v in observed.items() if k not in expected and v}
    if obs != dict(expected) or extra:
        raise ValueError(f"{what}: số mẫu {dict(observed)} khác số chốt {dict(expected)} — dừng và báo GVHD")


# ------------------------------------------------------------------------------------------------- GEO / GSE31210
def geo_stub(accession: str) -> str:
    """Thư mục FTP của GEO (chép notebook GEO): GSE31210 → GSE31nnn, GPL570 → GPLnnn, GPL6884 → GPL6nnn."""
    prefix, number = accession[:3], accession[3:]
    return prefix + (number[:-3] if len(number) > 3 else "") + "nnn"


def geo_series_matrix_url(gse: str) -> str:
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_stub(gse)}/{gse}/matrix/{gse}_series_matrix.txt.gz"


def geo_annot_url(gpl: str) -> str:
    return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{geo_stub(gpl)}/{gpl}/annot/{gpl}.annot.gz"


def characteristics_by_key(cells_per_sample: dict[str, list[str]]) -> pd.DataFrame:
    """Y98: mỗi ô "khoá: giá trị" đọc theo tiền tố của chính ô đó. Đầu vào: GSM → danh sách ô characteristics.

    Ô rỗng bỏ qua; ô không có ":" ghi vào khoá "_unkeyed"; một mẫu có cùng khoá hai lần → lỗi.
    """
    rows = {}
    for gsm, cells in cells_per_sample.items():
        rec = {}
        for cell in cells:
            cell = (cell or "").strip().strip('"').strip()
            if not cell:
                continue
            if ":" in cell:
                k, v = cell.split(":", 1)
                k, v = k.strip(), v.strip()
            else:
                k, v = "_unkeyed", cell
            if k in rec:
                raise ValueError(f"{gsm}: khoá '{k}' xuất hiện hai lần trong characteristics")
            rec[k] = v
        rows[gsm] = rec
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "sample_id"
    return out


def label_gse31210(chars: pd.DataFrame) -> pd.DataFrame:
    """QĐ185: 1 = "EGFR mutation +"; 0 = âm cả ba, KRAS+, ALK+; mô lành loại. Giá trị lạ → lỗi."""
    tissue = chars["tissue"].fillna("")
    alt = chars["gene alteration status"].fillna("") if "gene alteration status" in chars else pd.Series("", index=chars.index)
    normal = tissue.str.contains("normal", case=False)
    label = pd.Series(np.nan, index=chars.index)
    label[~normal & alt.isin(C.B7_GSE31210_POSITIVE)] = 1
    label[~normal & alt.isin(C.B7_GSE31210_NEGATIVE)] = 0
    unknown = ~normal & label.isna()
    if unknown.any():
        raise ValueError(f"GSE31210: mẫu u có giá trị 'gene alteration status' ngoài quy tắc QĐ185: "
                         f"{alt[unknown].value_counts().to_dict()}")
    status = np.where(normal, "normal_excluded", np.where(label == 1, "positive", "negative"))
    return pd.DataFrame({"label": label, "label_status": status, "gene_alteration_status": alt, "tissue": tissue},
                        index=chars.index)


def parse_series_matrix(path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Như notebook GEO (expression probe × mẫu, metadata Sample_*), thêm GSM → danh sách ô characteristics_ch1 (Y98)."""
    sample_rows, values, in_table = {}, [], False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                values.append(line)
            elif line.startswith("!Sample_"):
                key, *row = line.split("\t")
                sample_rows.setdefault(key[1:], []).append([x.strip().strip('"') for x in row])
    flat = {}
    for key, rows in sample_rows.items():
        for occurrence, row in enumerate(rows, start=1):
            flat[key if len(rows) == 1 else f"{key}_{occurrence}"] = row
    metadata = pd.DataFrame(flat).set_index("Sample_geo_accession")
    metadata.index.name = "sample_id"
    gsm = list(metadata.index)
    char_lines = sample_rows.get("Sample_characteristics_ch1", [])
    cells = {g: [line[i] for line in char_lines] for i, g in enumerate(gsm)}
    expression = pd.read_csv(io.StringIO("\n".join(values)), sep="\t", index_col=0, low_memory=False)
    expression = expression.loc[:, expression.columns.intersection(metadata.index)]
    return expression.apply(pd.to_numeric, errors="coerce"), metadata, cells


def load_platform_annotation_file(path) -> pd.Series:
    """Như load_platform_annotation của notebook GEO, đọc từ tệp .annot.gz đã tải."""
    rows, in_table = [], False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                rows.append(line)
    annotation = pd.read_csv(io.StringIO("".join(rows)), sep="\t", dtype=str)
    id_column = next((c for c in ("ID", "ID_REF", "Probe Set ID") if c in annotation), None)
    gene_column = next((c for c in ("Gene Symbol", "Gene symbol", "GENE_SYMBOL", "Symbol", "gene_assignment") if c in annotation), None)
    if id_column is None or gene_column is None:
        raise ValueError(f"{path}: không nhận diện được cột probe/gene; cột có: {list(annotation.columns)}")
    symbols = annotation.set_index(id_column)[gene_column].fillna("").str.split(" /// ").str[0].str.strip()
    out = symbols[(symbols != "") & ~symbols.str.contains("///", regex=False)]
    out.attrs["gene_column"] = gene_column
    return out


def collapse_probes(expression: pd.DataFrame, symbols: pd.Series) -> pd.DataFrame:
    """Chép đúng notebook GEO: bỏ probe thiếu giá trị, giữ probe phương sai lớn nhất mỗi gen; trả về mẫu × gen (float32)."""
    symbols = symbols.reindex(expression.index).dropna()
    kept = expression.loc[symbols.index].dropna(axis=0, how="any")
    symbols = symbols.loc[kept.index]
    order = kept.var(axis=1).sort_values(ascending=False).index
    kept, symbols = kept.loc[order], symbols.loc[order]
    kept = kept.loc[~symbols.duplicated()]
    kept.index = pd.Index(symbols.loc[kept.index], name="gene")
    return kept.sort_index().T.astype(np.float32)


def decide_scale(expression: pd.DataFrame, threshold: float = C.B7_LINEAR_SCALE_P99) -> tuple[pd.DataFrame, dict]:
    """A.2 (chốt trước khi mở dữ liệu): p99 của toàn ma trận > ngưỡng → tuyến tính: sàn 1 rồi log2; ngược lại giữ nguyên."""
    p99 = float(np.nanpercentile(expression.to_numpy(dtype=float), 99))
    linear = p99 > threshold
    out = np.log2(expression.clip(lower=1.0)) if linear else expression
    return out, {"p99": p99, "threshold": threshold, "scale": "linear → log2(max(x, 1))" if linear else "log2 (kept as is)"}


def sex_from_characteristics(chars: pd.DataFrame) -> pd.Series:
    """Cột sex (male / female / rỗng) từ trường gender/sex theo khoá (dùng cho QC giới tính như bốn cohort GEO)."""
    col = next((c for c in chars.columns if c.strip().lower() in ("gender", "sex")), None)
    if col is None:
        return pd.Series("", index=chars.index, name="sex")
    v = chars[col].fillna("").str.strip().str.lower()
    return v.map(lambda s: "male" if s in ("male", "m") else ("female" if s in ("female", "f") else "")).rename("sex")


def write_dataset(out_dir: Path, X: pd.DataFrame, meta: pd.DataFrame) -> None:
    """expression.h5 (key "expression") + metadata.csv (index sample_id), cùng định dạng các bộ hiện có."""
    out_dir.mkdir(parents=True, exist_ok=True)
    assert list(X.index) == list(meta.index), "expression và metadata khác thứ tự mẫu"
    X.to_hdf(out_dir / "expression.h5", key="expression", mode="w", complevel=4, complib="zlib")
    meta.to_csv(out_dir / "metadata.csv", index_label="sample_id")
