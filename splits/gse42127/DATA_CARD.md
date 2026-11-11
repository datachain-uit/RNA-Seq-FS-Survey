# Data card GSE42127

## Cập nhật 02/11/2026 — code J, sàng trùng, độ phủ gen panel (G-02, G-04; Review 31/10)

### Kiểm tra chéo nhãn với series matrix gốc (code J)

- **Cách làm:** `scripts/check_label_crosscheck.py` đọc `GSE42127_series_matrix.txt.gz` (thư mục `geo_target_work/raw/GSE42127/`), lấy trường `histology` trong `!Sample_characteristics_ch1` theo khoá (không theo vị trí cột), ánh xạ bằng khớp chính xác chuỗi theo `label_mapping.json`, không regex: `histology: Adenocarcinoma` → 0; `histology: Squamous` → 1.
- **Kết quả: ĐẠT.** Series matrix 176 mẫu → LUAD/LUSC 133/43 (trùng `n_samples` trong `label_mapping.json`) → trừ 9 mẫu trong `excluded_samples_gse42127.csv` → **126/41** = `DATASET_EXPECTED_N` 126/41. Tập mẫu trùng hoàn toàn với `metadata.csv` (không mẫu nào chỉ có ở một bên); nhãn trùng 100 % (0 mẫu lệch).
- **Tệp:** `qc/label_crosscheck_gse42127.csv` (từng mẫu của series matrix: giá trị gốc, nhãn suy ra, trạng thái giu / cach_ly_QC / khong_thuoc_bai_toan, nhãn trong metadata, cột `match`); tóm tắt `qc/label_crosscheck_summary.csv`. Ngày chạy 02/11/2026.

### Sàng mẫu trùng (r_z, 29/10/2026)

- `scripts/geo_sample_qc.py` (cấu hình `qc/rz_screen_config.json`, commit `5facab3`, chạy 29/10 trên bản trước QC): r_z = Pearson giữa hai mẫu sau khi chuẩn hoá z từng gen trong cohort; ngưỡng r_z ≥ 0,90, kèm r thô ≥ 0,99.
- **Trong cohort:** 15.400 cặp trên 19.405 gen; r_z lớn nhất 0,5770, phân vị 99,9 % 0,4366; **0 cặp r_z ≥ 0,90**; r thô lớn nhất 0,9662 (0 cặp r ≥ 0,99).
- **Giữa cohort** (17.062 gen chung của 4 cohort): r_z lớn nhất 0,4579 (với GSE30219); 0 cặp r_z ≥ 0,90 với mọi cohort khác.

### patient_id (O15)

- `metadata.csv` có cột `patient_id` (= `sample_id`), nên `load_dataset` ghi `patient_id_assumed_from_sample_id = false`. Căn cứ để coi mỗi mẫu là một bệnh nhân: title ghi mã mẫu ("… sample 250-T"); bỏ hậu tố -T/-T1/-T2/-Tx thì 176/176 mã khác nhau trên toàn series; sàng r_z không thấy cặp trùng; QC giới tính đã cách ly mẫu không xác minh được danh tính. Chia fold theo nhóm `patient_id` vì vậy tương đương chia theo mẫu.

### Độ phủ gen của panel tham chiếu

Nền tảng GPL6884. Không gian E1/E1b = gen của cohort giao với gen TCGA (`load_dataset(gene_space="reference")`). Panel được đánh giá với phần gen còn lại; `config.json` của E1 ghi danh sách thiếu và cờ `complete = false` (`qc/panel_gene_coverage.csv`, `scripts/check_panel_coverage.py`).

| Panel | Có / tổng (gen gốc của nền tảng) | Có / tổng (không gian E1/E1b) | Gen thiếu |
|---|---|---|---|
| IHC-2 | 2/2 | 2/2 | — |
| IHC-4 | 4/4 | 4/4 | — |
| Girard 2016 (42 gen) | 41/42 | 41/42 | PNCK |
| Wilkerson 2013 (52 gen) | 52/52 | 52/52 | — |

Ghi chú ký hiệu (03/11/2026): Wilkerson 2013 (Table 1) ghi TUBA1; khảo sát dùng TUBA4A theo HGNC (HGNC:12407, previous symbol TUBA1; tra 03/11/2026), không đối chiếu theo mồi PCR của bài gốc.

---

## Cập nhật 29/10/2026 — QC mẫu (QĐ21, QĐ22, QĐ22b) — **số mẫu chính thức cho survey v3**

- **n trước → sau:** 176 (133/43) → **167 (126/41)** LUAD/LUSC; `DATASET_EXPECTED_N` = (126, 41) (commit `380d01f`, "QĐ21+QĐ22").
- **Dựng lại tệp cohort:** `scripts/geo_sample_qc.py rebuild` tại commit `380d01f` (2026-10-29T13:24:03+07:00); chỉ bỏ các mẫu trong `excluded_samples_gse42127.csv` và thêm cột `sex`; mọi giá trị còn lại giữ nguyên (script kiểm tra bằng assert). Bản cũ: `_backup_20261029-132403/`.
- **SHA-256 mới:** `expression.h5` `6595946619c4a39def08b957725a2883358bbc8228ff0626799444ceb91532c8`; `metadata.csv` `8b18fd7f004f6fdfacbb4fbc8b93f89def8384152f559cdaad1c388570fe3d55`.
- **Bản chia (build_v3_splits.py --force, commit `380d01f`):** `repeated_splits.csv` `6a0f194aa586b94f3836019b273d12ac2bc8abbdfe0151e612c876be073fd959` (4175 dòng); `fewshot_splits.csv` `5cfec85ebf3b88981ba081114fdd2777ff6d65dc9085dac31859ecaeb34a5f01` (n_T: 5, 10, 20, pool; 5 split × 4 tập con).
- **Nguồn GEO (Mục 7.2.3):** GSE42127 — "Expression data for non-small-cell lung cancer"; nộp 07/11/2012, cập nhật lần cuối 22/06/2020 (theo `!Series_*` của series matrix); bài gốc PMID 23357979 (series còn liệt kê 32512502); series matrix tải về `geo_target_work/raw/GSE42127/` ngày 17/10/2026 09:21 (UTC+7) (thời điểm ghi tệp trên máy).
- **Nguồn giới tính:** trường `gender:` trong `!Sample_characteristics_ch1` của series matrix, đọc theo khoá (đã kiểm tra trực tiếp series matrix: không lệch cột).
- **Quy tắc (QĐ22, chế độ hai trục):** XIST và trung bình 5 gen Y (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY), mỗi trục chia hai cụm k-means trên chính cohort. opposite (cả hai trục trái metadata) và inconsistent (XIST và gen Y cùng cao) → cách ly; inconclusive (both-low, chỉ một trục trái, hoặc trục không phân biệt) → giữ, ghi cờ.
- **Ngưỡng:** XIST 7.07; trục Y 5.80 (log2).
- **Kết quả:** match 161, inconclusive 6, opposite 7, inconsistent 2.

### Mẫu cách ly (9)

QĐ22 — QC giới tính (opposite: giới tính trái metadata, nhãn mô học cùng dòng metadata nên không tin được; inconsistent: không xác minh được danh tính mẫu):

| Mẫu | Lớp | Giới (metadata) | XIST | trục Y (TB 5 gen) | category | verdict |
|---|---|---|---|---|---|---|
| GSM1032890 | LUSC | male | 7.91 | 8.20 | both-high | inconsistent |
| GSM1032894 | LUAD | male | 9.29 | 4.21 | F-like | opposite |
| GSM1032898 | LUAD | female | 3.68 | 7.80 | M-like | opposite |
| GSM1032907 | LUAD | male | 8.07 | 6.82 | both-high | inconsistent |
| GSM1032927 | LUAD | male | 8.84 | 4.12 | F-like | opposite |
| GSM1032941 | LUSC | male | 12.76 | 5.77 | F-like | opposite |
| GSM1032954 | LUAD | male | 11.05 | 4.26 | F-like | opposite |
| GSM1032956 | LUAD | female | 5.13 | 7.89 | M-like | opposite |
| GSM1033051 | LUAD | female | 4.60 | 7.87 | M-like | opposite |

**Ghi riêng cho phần hạn chế (QĐ22b):** GSE42127 mất 9/176 mẫu (5,1 %) — 7 opposite, 2 inconsistent; danh sách và giá trị ở bảng trên.

### Mẫu giữ nhưng ghi cờ — inconclusive (6)

| Mẫu | Lớp | Giới (metadata) | XIST | trục Y (TB 5 gen) | category | verdict |
|---|---|---|---|---|---|---|
| GSM1032926 | LUSC | male | 4.98 | 5.14 | both-low | inconclusive |
| GSM1032940 | LUAD | male | 5.51 | 5.38 | both-low | inconclusive |
| GSM1032960 | LUSC | male | 4.67 | 5.55 | both-low | inconclusive |
| GSM1032987 | LUAD | male | 4.50 | 5.36 | both-low | inconclusive |
| GSM1032990 | LUAD | female | 6.78 | 4.11 | both-low | inconclusive |
| GSM1033000 | LUAD | male | 4.73 | 5.67 | both-low | inconclusive |

---

*Phần dưới là data card trước QC 29/10 (giữ để truy vết; số mẫu và SHA-256 ở dưới là của bản cũ).*

- Role: secondary-platform GPL6884 target cohort for LUAD versus LUSC; report separately from the GPL570 pooled k-star analysis.
- Source: GEO GSE42127 series matrix; platform GPL6884.
- Included samples: 176 (LUAD 133, LUSC 43); one sample per patient, with `patient_id = sample_id`.
- Expression: 19,405 gene-symbol columns. The source series matrix states MBCB background correction and quality control; 17,235 genes intersect TCGA. NKX2-1, NAPSA, TP63 and KRT5 are present.
- Label source: `Sample_characteristics_ch1_7`; the explicit mapping is in `label_mapping.json`.
- Input SHA-256: `expression.h5` `c86ae8d9d308b28b12c3731af8111c25d136e00383c2ddaac4f59755cb99f8f5`; `metadata.csv` `0149812c5474539d0f619f8836d034d3769b3243beac1471bb2c57e3ae540d91`.
- Audit 2026-10-27: metadata IDs match expression IDs and the 176 source GEO IDs; no shared sample ID with the other target cohorts.

The historical fold files are not v3 splits. Official v3 splits: see the 29/10 update above.
