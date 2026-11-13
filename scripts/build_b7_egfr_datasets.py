"""B7 bước 1 (QĐ188, Review 13/11 Phụ lục A): dựng hai bộ `tcga_luad_egfr`, `gse31210_egfr` và data card.

Lệnh con, theo thứ tự (chạy trên Kaggle có Internet; dữ liệu TCGA lấy từ Kaggle Input experiment-data-v3):

  tcga      TCGA-LUAD: tải (hoặc đọc từ --cache) mọi tệp MAF mở "Masked Somatic Mutation" của GDC, khử trùng biến thể theo
            (sample_id, Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2), gắn nhãn EGFR theo quy tắc chốt 12/11
            (QĐ186) và TP53 (dự phòng); kiểm số 58/435/12/10 và 243/262/10, khác thì dừng. Biểu hiện = đúng các mẫu LUAD đã gắn
            nhãn của cohort TCGA hiện có (log2(TPM + 1), cùng gen, cùng QC); không tải lại biểu hiện.
  gse31210  GSE31210: series matrix + GPL570.annot.gz (tải hoặc --cache), trường mô tả đọc theo khoá (Y98), nhãn QĐ185 (127/99,
            loại 20 mô lành), quyết định thang giá trị theo quy tắc A.2 (p99 > 100 → sàn 1, log2) trước khi gom probe bằng đúng
            hàm của notebook GEO; ghi |G| = gen ∩ TCGA.
  card      Data card (JSON + Markdown) cho mỗi bộ, sau QC và bản chia: nguồn, định nghĩa nhãn, số mẫu từng nhóm, mẫu loại kèm
            lý do, phân bố FILTER (chỉ ghi lại), thang giá trị, |G|, SHA-256 tệp dữ liệu và bản chia.

Sau `gse31210` và trước `card` (cùng thủ tục QC như bốn cohort GEO):
    python scripts/check_sex_consistency.py --datasets gse31210_egfr
    python scripts/geo_sample_qc.py apply-sex-qc --cohorts gse31210_egfr
    python scripts/geo_sample_qc.py rz-screen --cohorts gse31210_egfr gse10245 gse30219 gse37745 --out $KLTN_DATA_ROOT/qc
    python scripts/geo_sample_qc.py rebuild --cohorts gse31210_egfr
    python scripts/build_e1_repeated_splits.py --datasets tcga_luad_egfr gse31210_egfr

Đầu ra trong thư mục của từng bộ: expression.h5, metadata.csv, excluded_samples_<bộ>.csv (QC; nhãn loại ghi riêng),
b7_labels_all.csv (mọi mẫu kể cả bị loại, kèm lý do), b7_build_info.json; TCGA thêm b7_variants_dedup.csv,
b7_filter_distribution.csv, b7_maf_files.csv; card: data_card_<bộ>.{json,md}.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import b7_egfr as B  # noqa: E402
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, read_gene_list, resolve_dataset_dir, sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, start_run  # noqa: E402

TCGA_B7, GEO_B7 = C.B7_DATASETS
GDC = "https://api.gdc.cancer.gov"
GSE31210_MATRIX = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31210/matrix/GSE31210_series_matrix.txt.gz"
GPL570_ANNOT = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570nnn/GPL570/annot/GPL570.annot.gz"
CLASS_NAMES = {0: "EGFR_wt", 1: "EGFR_mut"}


# ------------------------------------------------------------------------------------------------- mạng (chỉ trên Kaggle)
def _http(method, url, retries=5, **kw):
    import requests
    for i in range(retries):
        try:
            r = requests.request(method, url, timeout=600, **kw)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001 — mạng chập chờn: thử lại rồi mới dừng
            print(f"thử lại {i + 1}/{retries}: {exc}", flush=True)
            time.sleep(10 * (i + 1))
    raise SystemExit(f"không tải được {url}")


def download(url: str, path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        print("tải:", url, flush=True)
        path.write_bytes(_http("GET", url).content)
    return path


def fetch_gdc_mafs(cache: Path) -> tuple[dict, pd.DataFrame]:
    """Danh sách và nội dung MAF mở của TCGA-LUAD (cache: tệp gdc_files.csv + maf/<file_id>.maf.gz)."""
    status = _http("GET", f"{GDC}/status").json()
    files_csv = cache / "gdc_files.csv"
    if not files_csv.exists():
        filters = {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LUAD"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Masked Somatic Mutation"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}}]}
        fields = ["file_id", "file_name", "md5sum", "data_format", "experimental_strategy", "analysis.workflow_type",
                  "cases.submitter_id", "cases.samples.submitter_id", "cases.samples.sample_type"]
        hits = _http("GET", f"{GDC}/files", params={"filters": json.dumps(filters), "fields": ",".join(fields),
                                                     "format": "JSON", "size": "5000"}).json()["data"]["hits"]
        rows = [{"file_id": h["file_id"], "file_name": h["file_name"], "md5sum": h.get("md5sum"), "strategy": h.get("experimental_strategy"),
                 "workflow": (h.get("analysis") or {}).get("workflow_type"), "case": c.get("submitter_id"),
                 "sample": s.get("submitter_id"), "sample_type": s.get("sample_type")}
                for h in hits for c in h.get("cases", []) for s in c.get("samples", [])]
        cache.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(files_csv, index=False)
    files = pd.read_csv(files_csv, dtype=str)
    maf_dir = cache / "maf"
    maf_dir.mkdir(parents=True, exist_ok=True)
    todo = sorted(set(files.file_id) - {p.name.split(".")[0] for p in maf_dir.glob("*.maf.gz")})
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        r = _http("POST", f"{GDC}/data", json={"ids": batch})
        if len(batch) == 1:
            (maf_dir / f"{batch[0]}.maf.gz").write_bytes(r.content)
        else:
            with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
                for m in tf.getmembers():
                    if m.isfile() and ".maf" in m.name:
                        (maf_dir / f"{m.name.split('/')[0]}.maf.gz").write_bytes(tf.extractfile(m).read())
        print(f"MAF {min(i + 100, len(todo))}/{len(todo)}", flush=True)
    return status, files


# ------------------------------------------------------------------------------------------------- tcga
def build_tcga(args, prov) -> dict:
    status, files = fetch_gdc_mafs(args.cache / "gdc") if args.status_json is None else (json.loads(args.status_json.read_text()), pd.read_csv(args.cache / "gdc" / "gdc_files.csv", dtype=str))
    release = str(status.get("data_release", ""))
    if not release.startswith(C.B7_GDC_RELEASE_EXPECTED) and not args.allow_other_release:
        raise SystemExit(f"GDC {release!r} khác {C.B7_GDC_RELEASE_EXPECTED} của biểu hiện và bước 0 — dừng (--allow-other-release để bỏ qua)")
    parts, maf_samples = [], set()
    for fid in sorted(files.file_id.unique()):
        df, samples = B.read_maf(args.cache / "gdc" / "maf" / f"{fid}.maf.gz", f"{fid}.maf.gz")
        df["maf_file_id"] = fid
        parts.append(df)
        maf_samples |= samples
    maf_samples |= {s[:16] for s in files["sample"].dropna()}          # MAF rỗng vẫn tính là "có MAF" (như bước 0)
    var = B.dedup_variants(pd.concat(parts, ignore_index=True))
    tumour = files[files.sample_type.fillna("").str.contains("Tumor")]
    maf_files = tumour.groupby(tumour["sample"].str[:16])["file_id"].agg(lambda s: ";".join(sorted(set(s)))).to_dict()

    src = load_dataset(DATA_ROOT, C.SOURCE_DATASET, gene_space=None)
    meta_src = pd.read_csv(src.data_dir / "metadata.csv", dtype=str).set_index("sample_id").loc[list(src.sample_ids)]
    luad = meta_src[meta_src["class"] == "LUAD"].reset_index()[["sample_id", "patient_id"]]
    labels = B.label_tcga_luad(var, luad, maf_samples, maf_files)
    B.check_counts(labels.label_status.value_counts().to_dict(), C.B7_TCGA_EGFR_COUNTS, "TCGA-LUAD EGFR")
    B.check_counts(labels.tp53_label.value_counts().to_dict(), C.B7_TCGA_TP53_COUNTS, "TCGA-LUAD TP53")
    if labels.patient_id.duplicated().any():
        raise SystemExit("TCGA-LUAD: một bệnh nhân có nhiều mẫu — dừng")

    keep = labels[labels.label.notna()].copy()
    keep["label"] = keep["label"].astype(int)
    idx = {s: i for i, s in enumerate(src.sample_ids)}
    X = pd.DataFrame(src.X[[idx[s] for s in keep.sample_id]], index=keep.sample_id.to_numpy(), columns=src.genes).astype(np.float32)
    X.index.name = "sample_id"
    meta = keep.set_index("sample_id")
    meta["class"] = meta["label"].map(CLASS_NAMES)
    meta["sex"] = meta_src.loc[meta.index, "sex"] if "sex" in meta_src.columns else ""
    meta = meta[["patient_id", "class", "label", "sex", "label_status", "activating_classes", "egfr_variants", "tp53_label", "maf_file_ids"]]
    out = resolve_dataset_dir(DATA_ROOT, TCGA_B7)
    B.write_dataset(out, X, meta)
    labels.to_csv(out / "b7_labels_all.csv", index=False)
    pd.DataFrame(columns=["sample_id", "reason", "decision"]).to_csv(out / f"excluded_samples_{TCGA_B7}.csv", index=False)
    egfr_tp53 = var[var.sample_id.isin(set(luad.sample_id))]
    egfr_tp53.to_csv(out / "b7_variants_dedup.csv", index=False)
    filt = (egfr_tp53.assign(FILTER=egfr_tp53.get("FILTER", pd.Series("NA", index=egfr_tp53.index)).fillna("NA"))
            .groupby(["Hugo_Symbol", "FILTER"]).size().rename("n_variants").reset_index())
    filt.to_csv(out / "b7_filter_distribution.csv", index=False)
    files.to_csv(out / "b7_maf_files.csv", index=False)
    info = {"dataset": TCGA_B7, "source": f"GDC {release}; TCGA-LUAD Masked Somatic Mutation (open), {files.file_id.nunique()} MAF files; "
                                         f"expression = {C.SOURCE_DATASET} (log2(TPM + 1), same genes and QC)",
            "gdc_status": status, "workflows": sorted(files.workflow.dropna().unique().tolist()),
            "label_rule": "1 = >= 1 activating EGFR kinase-domain variant (ex19 in-frame del, L858R, L861Q, G719X, S768I, ex20 in-frame ins); "
                          "0 = no non-synonymous EGFR variant; other EGFR variants and samples without an open MAF excluded (QĐ186)",
            "variant_dedup_key": list(B.VARIANT_KEY), "n_variant_rows_raw": int(sum(len(p) for p in parts)), "n_variant_rows_dedup": int(len(var)),
            "egfr_counts": labels.label_status.value_counts().to_dict(), "tp53_counts": labels.tp53_label.value_counts().to_dict(),
            "activating_class_counts": {c: int(labels.activating_classes.str.split(";").apply(lambda s: c in s).sum()) for c in B.ACTIVATING_CLASSES},
            "excluded": labels.loc[labels.label.isna(), ["sample_id", "label_status", "egfr_variants"]].to_dict("records"),
            "n_samples": int(len(meta)), "n_per_class": [int((meta.label == 0).sum()), int((meta.label == 1).sum())],
            "n_genes": int(X.shape[1]), "scale": "log2(TPM + 1) (as source)", "provenance": prov, "versions": all_versions()}
    (out / "b7_build_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"[{TCGA_B7}] {info['n_samples']} mẫu (hoang dại/đột biến = {tuple(info['n_per_class'])}); |G| = {info['n_genes']}; {out}", flush=True)
    return info


# ------------------------------------------------------------------------------------------------- gse31210
def build_gse31210(args, prov) -> dict:
    raw = args.cache / "geo"
    matrix = download(GSE31210_MATRIX, raw / "GSE31210_series_matrix.txt.gz")
    annot = download(GPL570_ANNOT, raw / "GPL570.annot.gz")
    expression, meta_raw, cells = B.parse_series_matrix(matrix)
    if not meta_raw["Sample_platform_id"].eq("GPL570").all():
        raise SystemExit(f"GSE31210: nền tảng {meta_raw['Sample_platform_id'].unique().tolist()} không phải GPL570")
    chars = B.characteristics_by_key(cells)
    lab = B.label_gse31210(chars)
    B.check_counts(lab.label_status.value_counts().to_dict(), C.B7_GSE31210_COUNTS, "GSE31210")
    tumour = lab.index[lab.label.notna()]
    scaled, scale_info = B.decide_scale(expression.loc[:, tumour])      # quy tắc A.2 trên đúng các mẫu u được dùng
    symbols = B.load_platform_annotation_file(annot)
    X = B.collapse_probes(scaled, symbols).loc[tumour]
    tcga_genes = set(read_gene_list(DATA_ROOT, C.SOURCE_DATASET))
    meta = meta_raw.loc[tumour].copy()
    meta.insert(0, "patient_id", meta.index)           # một mẫu u mỗi bệnh nhân (mỗi GSM là một khối u; xem data card)
    meta.insert(1, "class", lab.loc[tumour, "label"].astype(int).map(CLASS_NAMES))
    meta.insert(2, "label", lab.loc[tumour, "label"].astype(int))
    meta.insert(3, "gene_alteration_status", lab.loc[tumour, "gene_alteration_status"])
    meta.insert(4, "sex", B.sex_from_characteristics(chars).loc[tumour])
    for col in chars.columns:
        meta[f"char: {col}"] = chars.loc[tumour, col]
    out = resolve_dataset_dir(DATA_ROOT, GEO_B7)
    B.write_dataset(out, X, meta)
    all_lab = lab.join(chars.add_prefix("char: ")).reset_index()
    all_lab.to_csv(out / "b7_labels_all.csv", index=False)
    pd.DataFrame(columns=["sample_id", "reason", "decision"]).to_csv(out / f"excluded_samples_{GEO_B7}.csv", index=False)
    processing = meta_raw.filter(like="Sample_data_processing").iloc[0].tolist() if meta_raw.filter(like="Sample_data_processing").shape[1] else []
    info = {"dataset": GEO_B7, "source": "NCBI GEO GSE31210 series matrix (GPL570); probe → gene by GPL570.annot.gz "
                                         f"(column '{symbols.attrs.get('gene_column')}'), functions of geo_luad_lusc_target_cohorts_5fold.ipynb",
            "files_sha256": {"series_matrix": sha256(matrix), "gpl570_annot": sha256(annot)},
            "label_rule": "1 = 'EGFR mutation +'; 0 = 'EGFR/KRAS/ALK -', 'KRAS mutation +', 'ALK-fusion +' (QĐ185); 'normal lung' excluded; "
                          "characteristics read by the key prefix of each cell (Y98)",
            "label_counts": lab.label_status.value_counts().to_dict(),
            "gene_alteration_counts_tumour": lab.loc[tumour, "gene_alteration_status"].value_counts().to_dict(),
            "excluded_normal": lab.index[lab.label_status == "normal_excluded"].tolist(),
            "sample_data_processing": processing, "scale_decision": scale_info,
            "n_samples": int(len(meta)), "n_per_class": [int((meta.label == 0).sum()), int((meta.label == 1).sum())],
            "n_genes_platform": int(X.shape[1]), "n_genes_in_tcga": int(len(set(X.columns) & tcga_genes)),
            "one_sample_per_patient": "each GSM of the tumour set is one primary tumour (Okayama et al. 2012); patient_id = sample_id",
            "provenance": prov, "versions": all_versions()}
    (out / "b7_build_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"[{GEO_B7}] {info['n_samples']} mẫu u (âm/dương = {tuple(info['n_per_class'])}); |G| = {info['n_genes_in_tcga']} "
          f"(nền tảng {info['n_genes_platform']}); thang: {scale_info['scale']} (p99 = {scale_info['p99']:.1f}); {out}", flush=True)
    return info


# ------------------------------------------------------------------------------------------------- card
def build_card(name: str, prov) -> dict:
    d = resolve_dataset_dir(DATA_ROOT, name)
    info = json.loads((d / "b7_build_info.json").read_text(encoding="utf-8"))
    ds = load_dataset(DATA_ROOT, name)
    excl_path = d / f"excluded_samples_{name}.csv"
    qc_excl = pd.read_csv(excl_path, dtype=str).to_dict("records") if excl_path.exists() else []
    split = d / "repeated_splits.csv"
    filt = d / "b7_filter_distribution.csv"
    card = {"dataset": name, "role": C.DATASETS.get(name), "platform": C.DATASET_PLATFORM.get(name), "source": info["source"],
            "label_mapping": {"EGFR wild-type / negative": 0, "EGFR mutant / positive": 1}, "label_rule": info["label_rule"],
            "n_after_qc": int(ds.n), "n_per_class_after_qc": [int((ds.y == 0).sum()), int((ds.y == 1).sum())],
            "positive_rate": round(float(ds.y.mean()), 4), "n_patients": int(len(set(ds.groups))),
            "label_exclusions": info.get("excluded") or info.get("excluded_normal"), "qc_exclusions": qc_excl,
            "filter_distribution": pd.read_csv(filt).to_dict("records") if filt.exists() else None,
            "scale": info.get("scale_decision") or info.get("scale"), "gene_space_G": int(ds.p),
            "build_info": {k: v for k, v in info.items() if k not in ("provenance", "versions", "excluded", "excluded_normal")},
            "sha256": {f: sha256(d / f) for f in ("expression.h5", "metadata.csv", "repeated_splits.csv", f"excluded_samples_{name}.csv",
                                                     "b7_labels_all.csv") if (d / f).exists()},
            "splits": str(split) if split.exists() else "MISSING — run build_e1_repeated_splits.py first",
            "card_provenance": prov}
    (d / f"data_card_{name}.json").write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md = [f"# Data card — {name}", "", f"- Role: {card['role']} (B7, QĐ177); platform: {card['platform']}", f"- Source: {card['source']}",
          f"- Label rule: {card['label_rule']}", f"- After QC: n = {card['n_after_qc']}, wild-type/negative = {card['n_per_class_after_qc'][0]}, "
          f"mutant/positive = {card['n_per_class_after_qc'][1]} (positive rate {card['positive_rate']}); patients = {card['n_patients']}",
          f"- Gene space |G| = {card['gene_space_G']}; scale: {card['scale']}",
          f"- Label exclusions: {len(card['label_exclusions'] or [])}; QC exclusions: {len(qc_excl)}", "", "## SHA-256", ""]
    md += [f"- `{f}`: `{h}`" for f, h in card["sha256"].items()]
    if card["filter_distribution"]:
        md += ["", "## FILTER of EGFR, TP53 rows (recorded only; labels unchanged)", ""]
        md += [f"- {r['Hugo_Symbol']} · {r['FILTER']}: {r['n_variants']}" for r in card["filter_distribution"]]
    (d / f"data_card_{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[{name}] data card: n = {card['n_after_qc']}, {tuple(card['n_per_class_after_qc'])}, |G| = {card['gene_space_G']}", flush=True)
    return card


def main(argv=None, script=__file__):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["tcga", "gse31210", "card"])
    ap.add_argument("--cache", type=Path, default=None, help="thư mục tải về (mặc định $KLTN_DATA_ROOT/_b7_raw)")
    ap.add_argument("--status-json", type=Path, default=None, help="tcga: dùng status GDC đã lưu thay vì gọi API (test)")
    ap.add_argument("--allow-other-release", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(C.B7_DATASETS), help="card: các bộ cần lập data card")
    args = ap.parse_args(argv)
    args.cache = args.cache or (DATA_ROOT / "_b7_raw")
    prov = start_run(script, params={k: str(v) for k, v in vars(args).items()})
    if args.command == "tcga":
        build_tcga(args, prov)
    elif args.command == "gse31210":
        build_gse31210(args, prov)
    else:
        for name in args.datasets:
            build_card(name, prov)


if __name__ == "__main__":
    main()
