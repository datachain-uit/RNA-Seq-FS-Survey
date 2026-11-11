# Data card GSE10245

## Cập nhật 02/11/2026 — code J, sàng trùng, độ phủ gen panel (G-02, G-04; Review 31/10)

### Kiểm tra chéo nhãn với series matrix gốc (code J)

- **Cách làm:** `scripts/check_label_crosscheck.py` đọc `GSE10245_series_matrix.txt.gz` (thư mục `geo_target_work/raw/GSE10245/`), lấy trường `disease state` trong `!Sample_characteristics_ch1` theo khoá (không theo vị trí cột), ánh xạ bằng khớp chính xác chuỗi theo `label_mapping.json`, không regex: `disease state: adenocarcinoma` → 0; `disease state: squamous cell carcinoma` → 1.
- **Kết quả: ĐẠT.** Series matrix 58 mẫu → LUAD/LUSC 40/18 (trùng `n_samples` trong `label_mapping.json`) → trừ 3 mẫu trong `excluded_samples_gse10245.csv` → **37/18** = `DATASET_EXPECTED_N` 37/18. Tập mẫu trùng hoàn toàn với `metadata.csv` (không mẫu nào chỉ có ở một bên); nhãn trùng 100 % (0 mẫu lệch).
- **Tệp:** `qc/label_crosscheck_gse10245.csv` (từng mẫu của series matrix: giá trị gốc, nhãn suy ra, trạng thái giu / cach_ly_QC / khong_thuoc_bai_toan, nhãn trong metadata, cột `match`); tóm tắt `qc/label_crosscheck_summary.csv`. Ngày chạy 02/11/2026.

### Sàng mẫu trùng (r_z, 29/10/2026)

- `scripts/geo_sample_qc.py` (cấu hình `qc/rz_screen_config.json`, commit `5facab3`, chạy 29/10 trên bản trước QC): r_z = Pearson giữa hai mẫu sau khi chuẩn hoá z từng gen trong cohort; ngưỡng r_z ≥ 0,90, kèm r thô ≥ 0,99.
- **Trong cohort:** 1.653 cặp trên 20.848 gen; r_z lớn nhất 0,4795, phân vị 99,9 % 0,4522; **0 cặp r_z ≥ 0,90**; r thô lớn nhất 0,9608 (0 cặp r ≥ 0,99).
- **Giữa cohort** (17.062 gen chung của 4 cohort): r_z lớn nhất 0,4828 (với GSE30219); 0 cặp r_z ≥ 0,90 với mọi cohort khác.

### patient_id (O15)

- `metadata.csv` có cột `patient_id` (= `sample_id`), nên `load_dataset` ghi `patient_id_assumed_from_sample_id = false`. Căn cứ để coi mỗi mẫu là một bệnh nhân: title không trùng (55/55); sàng r_z không thấy cặp trùng; QC giới tính đã cách ly mẫu không xác minh được danh tính. Chia fold theo nhóm `patient_id` vì vậy tương đương chia theo mẫu.

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

- **n trước → sau:** 58 (40/18) → **55 (37/18)** LUAD/LUSC; `DATASET_EXPECTED_N` = (37, 18) (commit `380d01f`, "QĐ21+QĐ22").
- **Dựng lại tệp cohort:** `scripts/geo_sample_qc.py rebuild` tại commit `380d01f` (2026-10-29T13:24:02+07:00); chỉ bỏ các mẫu trong `excluded_samples_gse10245.csv` và thêm cột `sex`; mọi giá trị còn lại giữ nguyên (script kiểm tra bằng assert). Bản cũ: `_backup_20261029-132402/`.
- **SHA-256 mới:** `expression.h5` `7cb29e07d6c1489dbd90ad81ebfa9675548bac729289f70cf04766f2119226ea`; `metadata.csv` `571f9d0ce2fc78275f41d09aa57a71f7a49ac3c6f1814c5b6f5b0002c12c2e26`.
- **Bản chia (build_v3_splits.py --force, commit `380d01f`):** `repeated_splits.csv` `823637702262a5fea795a52e9d2972f7857d9b75156d0667b1f7cac769496d5e` (1375 dòng); `fewshot_splits.csv` `b31efa81e03b21454e23d455d58e05100234fd1fd09029e1d655cff22d5b30d5` (n_T: 5, 10, pool; 5 split × 4 tập con).
- **Nguồn GEO (Mục 7.2.3):** GSE10245 — "Gene expression differences between adenocarcinoma and squamous cell carcinoma in human NSCLC"; nộp 23/01/2008, cập nhật lần cuối 25/03/2019 (theo `!Series_*` của series matrix); bài gốc PMID 18486272; series matrix tải về `geo_target_work/raw/GSE10245/` ngày 17/10/2026 09:19 (UTC+7) (thời điểm ghi tệp trên máy).
- **Nguồn giới tính:** trường `—` trong `!Sample_characteristics_ch1` của series matrix, đọc theo khoá — **GSE10245 không công bố giới tính**, cột `sex` để trống; chỉ phân loại theo biểu hiện (both-high vẫn cách ly).
- **Quy tắc (QĐ22, chế độ hai trục):** XIST và trung bình 5 gen Y (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY), mỗi trục chia hai cụm k-means trên chính cohort. opposite (cả hai trục trái metadata) và inconsistent (XIST và gen Y cùng cao) → cách ly; inconclusive (both-low, chỉ một trục trái, hoặc trục không phân biệt) → giữ, ghi cờ.
- **Ngưỡng:** XIST 7.02; trục Y 5.70 (log2).
- **Kết quả:** no-metadata 55, inconsistent 3.

### Mẫu cách ly (3)

QĐ22 — QC giới tính (opposite: giới tính trái metadata, nhãn mô học cùng dòng metadata nên không tin được; inconsistent: không xác minh được danh tính mẫu):

| Mẫu | Lớp | Giới (metadata) | XIST | trục Y (TB 5 gen) | category | verdict |
|---|---|---|---|---|---|---|
| GSM258572 | LUAD | không có | 12.94 | 7.88 | both-high | inconsistent |
| GSM258574 | LUAD | không có | 10.34 | 8.21 | both-high | inconsistent |
| GSM258593 | LUAD | không có | 12.97 | 6.97 | both-high | inconsistent |

### Mẫu giữ nhưng ghi cờ — inconclusive (0)

Không có.

---

*Phần dưới là data card trước QC 29/10 (giữ để truy vết; số mẫu và SHA-256 ở dưới là của bản cũ).*

- Role: primary GPL570 target cohort for LUAD versus LUSC.
- Source: GEO GSE10245 series matrix; platform GPL570.
- Included samples: 58 (LUAD 40, LUSC 18); one sample per patient, with `patient_id = sample_id`.
- Expression: 20,848 gene-symbol columns. The source series matrix states gcRMA normalization; 16,479 genes intersect TCGA. NKX2-1, NAPSA, TP63 and KRT5 are present.
- Label source: `Sample_characteristics_ch1`; the explicit mapping is in `label_mapping.json`.
- Input SHA-256: `expression.h5` `0ec3ed892ba76a4bf212a0ea0ae9b8b0a93771e614922ce6a54a12181392ce15`; `metadata.csv` `16f7dfedf06e2cbab396fd6ffb3d81858cd788beda27f56b190101ae5f134afe`.
- Audit 2026-10-27: metadata IDs match expression IDs and the 58 source GEO IDs; no shared sample ID with the other target cohorts.

The historical fold files are not v3 splits. Official v3 splits: see the 29/10 update above.
