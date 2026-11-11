# Data card GSE37745

## Cập nhật 02/11/2026 — code J, sàng trùng, độ phủ gen panel (G-02, G-04; Review 31/10)

### Kiểm tra chéo nhãn với series matrix gốc (code J)

- **Cách làm:** `scripts/check_label_crosscheck.py` đọc `GSE37745_series_matrix.txt.gz` (thư mục `geo_target_work/raw/GSE37745/`), lấy trường `histology` trong `!Sample_characteristics_ch1` theo khoá (không theo vị trí cột), ánh xạ bằng khớp chính xác chuỗi theo `label_mapping.json`, không regex: `histology: adeno` → 0; `histology: squamous` → 1.
- **Kết quả: ĐẠT.** Series matrix 196 mẫu → LUAD/LUSC 106/66 (trùng `n_samples` trong `label_mapping.json`) → trừ 3 mẫu trong `excluded_samples_gse37745.csv` → **104/65** = `DATASET_EXPECTED_N` 104/65. Tập mẫu trùng hoàn toàn với `metadata.csv` (không mẫu nào chỉ có ở một bên); nhãn trùng 100 % (0 mẫu lệch).
- **Giá trị nhãn không thuộc bài toán (bị loại do không khớp chính xác):** histology: large (24).
- **Tệp:** `qc/label_crosscheck_gse37745.csv` (từng mẫu của series matrix: giá trị gốc, nhãn suy ra, trạng thái giu / cach_ly_QC / khong_thuoc_bai_toan, nhãn trong metadata, cột `match`); tóm tắt `qc/label_crosscheck_summary.csv`. Ngày chạy 02/11/2026.

### Sàng mẫu trùng (r_z, 29/10/2026)

- `scripts/geo_sample_qc.py` (cấu hình `qc/rz_screen_config.json`, commit `5facab3`, chạy 29/10 trên bản trước QC): r_z = Pearson giữa hai mẫu sau khi chuẩn hoá z từng gen trong cohort; ngưỡng r_z ≥ 0,90, kèm r thô ≥ 0,99.
- **Trong cohort:** 14.706 cặp trên 20.848 gen; r_z lớn nhất 0,7175, phân vị 99,9 % 0,4875; **0 cặp r_z ≥ 0,90**; r thô lớn nhất 0,9925 (2 cặp r ≥ 0,99).
- **Giữa cohort** (17.062 gen chung của 4 cohort): r_z lớn nhất 0,5074 (với GSE30219); 0 cặp r_z ≥ 0,90 với mọi cohort khác.
- Hai cặp r ≥ 0,99 đã xử lý ở mục 29/10: GSM1019246/GSM1019247 cách ly (QĐ21); GSM1019319/GSM1019325 giữ (hai bệnh nhân khác nhau, cùng giới, cùng nhãn).

### patient_id (O15)

- `metadata.csv` có cột `patient_id` (= `sample_id`), nên `load_dataset` ghi `patient_id_assumed_from_sample_id = false`. Căn cứ để coi mỗi mẫu là một bệnh nhân: title ghi số bệnh nhân ("Patient N, …"), 169/169 số khác nhau; sàng r_z không thấy cặp trùng; QC giới tính đã cách ly mẫu không xác minh được danh tính. Chia fold theo nhóm `patient_id` vì vậy tương đương chia theo mẫu.

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

- **n trước → sau:** 172 (106/66) → **169 (104/65)** LUAD/LUSC; `DATASET_EXPECTED_N` = (104, 65) (commit `380d01f`, "QĐ21+QĐ22").
- **Dựng lại tệp cohort:** `scripts/geo_sample_qc.py rebuild` tại commit `380d01f` (2026-10-29T13:24:03+07:00); chỉ bỏ các mẫu trong `excluded_samples_gse37745.csv` và thêm cột `sex`; mọi giá trị còn lại giữ nguyên (script kiểm tra bằng assert). Bản cũ: `_backup_20261029-132403/`.
- **SHA-256 mới:** `expression.h5` `f5dcaf11d7f36defc37ad67cf47da30b2f85823e3360fef2f8f4d0ab96a34ec7`; `metadata.csv` `568b557e43c09daee94f8959f0144db642064502d73cb6493e8f5ff77c031328`.
- **Bản chia (build_v3_splits.py --force, commit `380d01f`):** `repeated_splits.csv` `18dc9577a10b86f9dec236cefad6beac48fb43db4c9d1383fed547b80712fd20` (4225 dòng); `fewshot_splits.csv` `ede2fd00f8dbff7ace0cac6b48761aafbab1601878e5f1ab2967b3889ae95a74` (n_T: 5, 10, 20, 40, pool; 5 split × 4 tập con).
- **Nguồn GEO (Mục 7.2.3):** GSE37745 — "Biomarker discovery in non-small cell lung cancer: integrating gene expression profiling, meta-analysis and tissue microarray validation"; nộp 03/05/2012, cập nhật lần cuối 26/05/2022 (theo `!Series_*` của series matrix); bài gốc PMID 23032747 (series còn liệt kê 29112949, 26608184, 33576873, 35574381); series matrix tải về `geo_target_work/raw/GSE37745/` ngày 17/10/2026 02:23 (UTC+7) (thời điểm ghi tệp trên máy).
- **Nguồn giới tính:** trường `gender:` trong `!Sample_characteristics_ch1` của series matrix, đọc theo khoá (đã kiểm tra trực tiếp series matrix: không lệch cột).
- **Quy tắc (QĐ22, chế độ hai trục):** XIST và trung bình 5 gen Y (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY), mỗi trục chia hai cụm k-means trên chính cohort. opposite (cả hai trục trái metadata) và inconsistent (XIST và gen Y cùng cao) → cách ly; inconclusive (both-low, chỉ một trục trái, hoặc trục không phân biệt) → giữ, ghi cờ.
- **Ngưỡng:** XIST 8.51; trục Y 6.30 (log2).
- **Kết quả:** match 169, inconclusive 1, opposite 1, inconsistent 1.

### Mẫu cách ly (3)

QĐ21 — cặp gần trùng GSM1019246/GSM1019247: r = 0,9925 (cặp kế tiếp ≤ 0,959); cả hai nữ theo biểu hiện, GSM1019247 ghi nam; hai nhãn đối nhau (LUSC/LUAD) → cách ly cả hai.

QĐ22 — QC giới tính (opposite: giới tính trái metadata, nhãn mô học cùng dòng metadata nên không tin được; inconsistent: không xác minh được danh tính mẫu):

| Mẫu | Lớp | Giới (metadata) | XIST | trục Y (TB 5 gen) | category | verdict |
|---|---|---|---|---|---|---|
| GSM1019197 | LUAD | male | 11.25 | 8.47 | both-high | inconsistent |

### Mẫu giữ nhưng ghi cờ — inconclusive (1)

| Mẫu | Lớp | Giới (metadata) | XIST | trục Y (TB 5 gen) | category | verdict |
|---|---|---|---|---|---|---|
| GSM1019177 | LUSC | male | 4.81 | 5.84 | both-low | inconclusive |

### Cặp tương quan cao được giữ

GSM1019319 / GSM1019325: r = 0,9905 (r_z = 0,7175) trên 20.848 gen; metadata nhất quán (Patient 294 và Patient 329, cùng nam, cùng LUSC; gen giới tính khớp nam) → giữ cả hai theo quy tắc (hai bệnh nhân khác nhau, cùng giới, cùng nhãn).

---

*Phần dưới là data card trước QC 29/10 (giữ để truy vết; số mẫu và SHA-256 ở dưới là của bản cũ).*

- Role: primary GPL570 target cohort for LUAD versus LUSC.
- Source: GEO GSE37745 series matrix; platform GPL570.
- Included samples: 172 (LUAD 106, LUSC 66); samples with histologies other than explicit adeno or squamous are excluded.
- Expression: 20,848 gene-symbol columns. The source series matrix states RMA preprocessing; 16,479 genes intersect TCGA. NKX2-1, NAPSA, TP63 and KRT5 are present.
- Label source: `Sample_characteristics_ch1_3`; the explicit mapping is in `label_mapping.json`.
- Input SHA-256: `expression.h5` `783d49f8599f82321f27cfb9081c2b2b71a064d7f481f5bab1fdb4b1b1de768b`; `metadata.csv` `af9ef378801234b02d9bb7b7003ad8fcaade99d8fe58be57f5a9d86d2ab3e742`.
- Audit 2026-10-27: metadata IDs match expression IDs and are all present in the 196-sample source GEO matrix; no shared sample ID with the other target cohorts.
- Duplicate screen: GSM1019246 and GSM1019247 have expression correlation 0.992273 on genes shared across the four targets. The original series matrix identifies them as Patient 30 female squamous and Patient 36 male adeno, respectively; both are retained.

The historical fold files are not v3 splits. Official v3 splits: see the 29/10 update above.
