# Data card GSE30219

## Cập nhật 02/11/2026 — code J, sàng trùng, độ phủ gen panel (G-02, G-04; Review 31/10)

### Kiểm tra chéo nhãn với series matrix gốc (code J)

- **Cách làm:** `scripts/check_label_crosscheck.py` đọc `GSE30219_series_matrix.txt.gz` (thư mục `geo_target_work/raw/GSE30219/`), lấy trường `histology` trong `!Sample_characteristics_ch1` theo khoá (không theo vị trí cột), ánh xạ bằng khớp chính xác chuỗi theo `label_mapping.json`, không regex: `histology: ADC` → 0; `histology: SQC` → 1.
- **Kết quả: ĐẠT.** Series matrix 307 mẫu → LUAD/LUSC 85/61 (trùng `n_samples` trong `label_mapping.json`) → trừ 5 mẫu trong `excluded_samples_gse30219.csv` → **81/60** = `DATASET_EXPECTED_N` 81/60. Tập mẫu trùng hoàn toàn với `metadata.csv` (không mẫu nào chỉ có ở một bên); nhãn trùng 100 % (0 mẫu lệch).
- **Giá trị nhãn không thuộc bài toán (bị loại do không khớp chính xác):** histology: BAS (39); histology: CARCI (24); histology: LCC (3); histology: LCNE (56); histology: NTL (14); histology: Other (4); histology: SCC (21). Ở GSE30219, `SCC` là ung thư phổi tế bào nhỏ, còn tế bào vảy ghi là `SQC`.
- **Tệp:** `qc/label_crosscheck_gse30219.csv` (từng mẫu của series matrix: giá trị gốc, nhãn suy ra, trạng thái giu / cach_ly_QC / khong_thuoc_bai_toan, nhãn trong metadata, cột `match`); tóm tắt `qc/label_crosscheck_summary.csv`. Ngày chạy 02/11/2026.

### Sàng mẫu trùng (r_z, 29/10/2026)

- `scripts/geo_sample_qc.py` (cấu hình `qc/rz_screen_config.json`, commit `5facab3`, chạy 29/10 trên bản trước QC): r_z = Pearson giữa hai mẫu sau khi chuẩn hoá z từng gen trong cohort; ngưỡng r_z ≥ 0,90, kèm r thô ≥ 0,99.
- **Trong cohort:** 10.585 cặp trên 20.848 gen; r_z lớn nhất 0,5721, phân vị 99,9 % 0,4824; **0 cặp r_z ≥ 0,90**; r thô lớn nhất 0,9818 (0 cặp r ≥ 0,99).
- **Giữa cohort** (17.062 gen chung của 4 cohort): r_z lớn nhất 0,5074 (với GSE37745); 0 cặp r_z ≥ 0,90 với mọi cohort khác.

### patient_id (O15)

- `metadata.csv` có cột `patient_id` (= `sample_id`), nên `load_dataset` ghi `patient_id_assumed_from_sample_id = false`. Căn cứ để coi mỗi mẫu là một bệnh nhân: title không trùng (141/141); sàng r_z không thấy cặp trùng; QC giới tính đã cách ly mẫu không xác minh được danh tính. Chia fold theo nhóm `patient_id` vì vậy tương đương chia theo mẫu.

### Hai bản dựng GSE30219 (Y12)

- Lịch sử có hai bản: bản trước R5 và bản R5 đã sửa (dựng bằng `scripts/rebuild_gse30219_r5.py`; thư mục từng mang hậu tố `_r5_fixed`, nay là `gse30219`). Mọi phiên survey v3 dùng bản R5 (141 mẫu sau QC). `DATASET_EXPECTED_N` = (81, 60) là số của bản này; `load_dataset` cảnh báo khi số mẫu khác (dừng khi `KLTN_STRICT_N=1`).
- Script so khớp hai bản không nộp (Review 31/10: chuyển thành ghi chú). Thay vào đó, code J đối chiếu thẳng bản R5 với series matrix gốc (307 mẫu): chỉ `histology: ADC`/`histology: SQC` được nhận, tập mẫu và nhãn trùng 100 %.

### Độ phủ gen của panel tham chiếu

Nền tảng GPL570. Không gian E1/E1b = gen của cohort giao với gen TCGA (`load_dataset(gene_space="reference")`). Panel được đánh giá với phần gen còn lại; `config.json` của E1 ghi danh sách thiếu và cờ `complete = false` (`qc/panel_gene_coverage.csv`, `scripts/check_panel_coverage.py`).

| Panel | Có / tổng (gen gốc của nền tảng) | Có / tổng (không gian E1/E1b) | Gen thiếu |
|---|---|---|---|
| IHC-2 | 2/2 | 2/2 | — |
| IHC-4 | 4/4 | 4/4 | — |
| Girard 2016 (42 gen) | 39/42 | 39/42 | ABCC6, KRT6C, KRT17 |
| Wilkerson 2013 (52 gen) | 50/52 | 50/52 | CLEC3B, TCP1 |

Ghi chú ký hiệu (03/11/2026): Wilkerson 2013 (Table 1) ghi TUBA1; khảo sát dùng TUBA4A theo HGNC (HGNC:12407, previous symbol TUBA1; tra 03/11/2026), không đối chiếu theo mồi PCR của bài gốc.

Các gen thiếu không có cột trong `expression.h5` của bản dựng GEO. Nguyên nhân nằm ở bước gộp probe → gen khi dựng, chưa truy lại theo probe. Bản dựng không được sửa để bù.

---

## Cập nhật 29/10/2026 — QC mẫu (QĐ21, QĐ22, QĐ22b) — **số mẫu chính thức cho survey v3**

- **n trước → sau:** 146 (85/61) → **141 (81/60)** LUAD/LUSC; `DATASET_EXPECTED_N` = (81, 60) (commit `380d01f`, "QĐ21+QĐ22").
- **Dựng lại tệp cohort:** `scripts/geo_sample_qc.py rebuild` tại commit `380d01f` (2026-10-29T13:24:02+07:00); chỉ bỏ các mẫu trong `excluded_samples_gse30219.csv` và thêm cột `sex`; mọi giá trị còn lại giữ nguyên (script kiểm tra bằng assert). Bản cũ: `_backup_20261029-132402/`.
- **SHA-256 mới:** `expression.h5` `6c2f7377ad802a7df3d804473884ad64fa08beadfa44ecb5232343801230e2d9`; `metadata.csv` `570923f3f20216def97dfc47ca06a2e78f30669f4b7829f59b6426d7a32501bc`.
- **Bản chia (build_v3_splits.py --force, commit `380d01f`):** `repeated_splits.csv` `de92722bd27a853fef77abe4fda22bcd64886ef2724c298ac386b0bfd9b979bc` (3525 dòng); `fewshot_splits.csv` `88667258ebb2d5150a45971e121bbd8d36b2a722f4986a9d2d48dc3ae48aeb7b` (n_T: 5, 10, 20, 40, pool; 5 split × 4 tập con).
- **Nguồn GEO (Mục 7.2.3):** GSE30219 — ""Off-context" gene expression in lung cancer identifies a group of metastatic-prone tumors"; nộp 26/06/2011, cập nhật lần cuối 25/03/2019 (theo `!Series_*` của series matrix); bài gốc PMID 23698379; series matrix tải về `geo_target_work/raw/GSE30219/` ngày 17/10/2026 02:31 (UTC+7) (thời điểm ghi tệp trên máy).
- **Nguồn giới tính:** trường `gender:` trong `!Sample_characteristics_ch1` của series matrix, đọc theo khoá (đã kiểm tra trực tiếp series matrix: không lệch cột).
- **Quy tắc (QĐ22, chế độ hai trục):** XIST và trung bình 5 gen Y (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY), mỗi trục chia hai cụm k-means trên chính cohort. opposite (cả hai trục trái metadata) và inconsistent (XIST và gen Y cùng cao) → cách ly; inconclusive (both-low, chỉ một trục trái, hoặc trục không phân biệt) → giữ, ghi cờ.
- **Ngưỡng:** XIST 7.72; trục Y 5.92 (log2).
- **Kết quả:** match 132, inconclusive 9, opposite 1, inconsistent 4.

### Mẫu cách ly (5)

QĐ22 — QC giới tính (opposite: giới tính trái metadata, nhãn mô học cùng dòng metadata nên không tin được; inconsistent: không xác minh được danh tính mẫu):

| Mẫu | Lớp | Giới (metadata) | XIST | trục Y (TB 5 gen) | category | verdict |
|---|---|---|---|---|---|---|
| GSM748065 | LUSC | male | 9.01 | 7.03 | both-high | inconsistent |
| GSM748078 | LUAD | male | 13.05 | 3.52 | F-like | opposite |
| GSM748240 | LUAD | male | 12.34 | 6.38 | both-high | inconsistent |
| GSM748278 | LUAD | male | 8.77 | 6.95 | both-high | inconsistent |
| GSM748279 | LUAD | male | 8.91 | 8.22 | both-high | inconsistent |

### Mẫu giữ nhưng ghi cờ — inconclusive (9)

| Mẫu | Lớp | Giới (metadata) | XIST | trục Y (TB 5 gen) | category | verdict |
|---|---|---|---|---|---|---|
| GSM748059 | LUSC | male | 3.85 | 5.52 | both-low | inconclusive |
| GSM748061 | LUSC | male | 4.32 | 5.29 | both-low | inconclusive |
| GSM748081 | LUAD | male | 4.72 | 5.48 | both-low | inconclusive |
| GSM748113 | LUAD | male | 4.14 | 5.18 | both-low | inconclusive |
| GSM748120 | LUSC | male | 4.59 | 5.62 | both-low | inconclusive |
| GSM748172 | LUSC | male | 4.77 | 4.89 | both-low | inconclusive |
| GSM748177 | LUSC | male | 3.87 | 5.84 | both-low | inconclusive |
| GSM748285 | LUSC | male | 4.25 | 5.87 | both-low | inconclusive |
| GSM1465998 | LUSC | male | 3.87 | 5.85 | both-low | inconclusive |

Ghi chú: `excluded_samples.csv` (cũ, R5) liệt kê các mẫu không phải ADC/SQC; `excluded_samples_gse30219.csv` (mới) liệt kê mẫu cách ly theo QC.

---

*Phần dưới là data card trước QC 29/10 (giữ để truy vết; số mẫu và SHA-256 ở dưới là của bản cũ).*

- Role: primary GPL570 target cohort for LUAD versus LUSC.
- Source: GEO GSE30219 series matrix; platform GPL570; R5 corrected reconstruction.
- Included samples: 146 (LUAD 85, LUSC 61). Only exact `ADC` and `SQC` records are included; other histologies are excluded in `excluded_samples.csv`.
- Expression: 20,848 gene-symbol columns. The source series matrix states RMA normalization; 16,479 genes intersect TCGA. NKX2-1, NAPSA, TP63 and KRT5 are present.
- Label source: `histology_raw`; the explicit mapping is in `label_mapping.json`.
- Input SHA-256: `expression.h5` `622162531d63db0f977cb73217fefe5ac6fc802ca31ce55777015791334d3b92`; `metadata.csv` `c36345825cbdbd60673c9a96d2e1cafea2af21c869cf1a9c39652a2e09c3eea0`.
- Audit 2026-10-27: metadata IDs match expression IDs and are all present in the 307-sample source GEO matrix; no shared sample ID with the other target cohorts.

The historical fold files are not v3 splits. Official v3 splits: see the 29/10 update above.
