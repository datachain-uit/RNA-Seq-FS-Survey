# Data card: `tcga_luad_lusc_primary_tumor`

## Cập nhật 02/11/2026 — độ phủ gen panel (G-02; Review 31/10)

- Code J không áp dụng: nhãn TCGA theo dự án gốc (LUAD → 0, LUSC → 1; mục Nhãn).
- TCGA là bộ tham chiếu của không gian gen (`GENE_SPACE_REFERENCE`): E1/E1b dùng nguyên 19.938 gen.

| Panel | Có / tổng (gen gốc của nền tảng) | Có / tổng (không gian E1/E1b) | Gen thiếu |
|---|---|---|---|
| IHC-2 | 2/2 | 2/2 | — |
| IHC-4 | 4/4 | 4/4 | — |
| Girard 2016 (42 gen) | 42/42 | 42/42 | — |
| Wilkerson 2013 (52 gen) | 52/52 | 52/52 | — |

Ghi chú ký hiệu (03/11/2026): Wilkerson 2013 (Table 1) ghi TUBA1; khảo sát dùng TUBA4A theo HGNC (HGNC:12407, previous symbol TUBA1; tra 03/11/2026), không đối chiếu theo mồi PCR của bài gốc.

---

## Cập nhật 29/10/2026 — QC mẫu (QĐ21, QĐ22, QĐ22b) — **số mẫu chính thức cho survey v3**

- **n trước → sau:** 1018 (517/501) → **1016 (515/501)** LUAD/LUSC; `DATASET_EXPECTED_N` = (515, 501) (commit `380d01f`, "QĐ21+QĐ22").
- **Dựng lại tệp cohort:** `scripts/geo_sample_qc.py rebuild` tại commit `380d01f` (2026-10-29T13:24:04+07:00); chỉ bỏ các mẫu trong `excluded_samples_tcga_luad_lusc_primary_tumor.csv` và thêm cột `sex`; mọi giá trị còn lại giữ nguyên (script kiểm tra bằng assert). Bản cũ: `_backup_20261029-132404/`.
- **SHA-256 mới:** `expression.h5` `41bf54bda90aae286eab2ceec7e705aeaa778cd77eb85587ab7ff8aaa75c2cff`; `metadata.csv` `ee65ea8b4075967867ce4f7fa8383122449d6ba2e10dd5e178a33b27b5ea7b26`.
- **Bản chia (build_v3_splits.py --force, commit `380d01f`):** `repeated_splits.csv` `ff909fd90415014e8dce4c1692346baa4873bee0e533932515880a5ed18cb6ce` (25400 dòng).
- **Nguồn giới tính:** GDC API `POST /cases`, trường `demographic.sex_at_birth` (trường `gender` trống ở mọi hồ sơ), truy vấn 2026-10-29T11:52:39+07:00 (commit `45b11a6`); 1018/1018 bệnh nhân, 610 nam / 408 nữ; tệp `sex_gdc.csv` (SHA-256 `72d16061e79f43081a3bc5210b075431ac6ad07fc354e0b583b141588ef1518c`).
- **Quy tắc (QĐ22b, chế độ một trục — TCGA v2 chỉ có gen mã hoá protein, không có XIST):** bỏ phiếu từng gen Y; ngưỡng mỗi gen = trung điểm (phân vị 95 của nhóm ghi nữ, phân vị 5 của nhóm ghi nam); M-like(Y) khi ≥ 4/5 gen trên ngưỡng, F-like(Y) khi ≤ 1/5; còn lại intermediate(Y) → inconclusive. opposite → cách ly; inconclusive → giữ, ghi cờ.
- **Ngưỡng 5 gen (log2 TPM+1):** RPS4Y1 2.72; DDX3Y 1.10; KDM5D 0.66; UTY 0.31; EIF1AY 0.62.
- **Kết quả:** match 1011, inconclusive 5, opposite 2.
- **Ghi chú:** bản v3.1.2 (trục trung bình 5 gen Y) cho 5 opposite giả ở nam LUSC giảm biểu hiện gen Y trong u, nên đổi sang bỏ phiếu từng gen (29/10).

### Mẫu cách ly (2)

QĐ22b — QC giới tính (opposite: giới tính trái metadata, nhãn mô học cùng dòng metadata nên không tin được; inconsistent: không xác minh được danh tính mẫu):

| Mẫu | Lớp | Giới (metadata) | RPS4Y1 | DDX3Y | KDM5D | UTY | EIF1AY | Phiếu nam | category | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| TCGA-49-4505-01A | LUAD | female | 6.67 | 3.28 | 1.57 | 0.97 | 2.52 | 5/5 | M-like(Y) | opposite |
| TCGA-49-AARN-01A | LUAD | female | 6.45 | 2.14 | 1.73 | 1.01 | 1.81 | 5/5 | M-like(Y) | opposite |

### Mẫu giữ nhưng ghi cờ — inconclusive (5)

| Mẫu | Lớp | Giới (metadata) | RPS4Y1 | DDX3Y | KDM5D | UTY | EIF1AY | Phiếu nam | category | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| TCGA-21-5782-01A | LUSC | female | 1.19 | 0.63 | 0.71 | 0.91 | 0.85 | 3/5 | intermediate(Y) | inconclusive |
| TCGA-33-4582-01A | LUSC | male | 3.99 | 1.20 | 0.30 | 0.17 | 0.67 | 3/5 | intermediate(Y) | inconclusive |
| TCGA-85-A4QR-01A | LUSC | male | 3.30 | 0.86 | 0.71 | 0.20 | 0.32 | 2/5 | intermediate(Y) | inconclusive |
| TCGA-MF-A522-01A | LUSC | male | 3.28 | 1.49 | 0.59 | 0.26 | 0.75 | 3/5 | intermediate(Y) | inconclusive |
| TCGA-NC-A5HG-01A | LUSC | male | 3.23 | 1.07 | 0.74 | 0.20 | 0.61 | 2/5 | intermediate(Y) | inconclusive |

---

*Phần dưới là data card trước QC 29/10 (giữ để truy vết; số mẫu và SHA-256 ở dưới là của bản cũ).*

**Tác vụ:** Phân loại mô học LUAD vs LUSC trên mẫu u nguyên phát

- Nguồn: GDC API, TCGA-LUAD + TCGA-LUSC, STAR - Counts (GDC data release: Data Release 46.0 - August 10, 2026)
- Số mẫu / bệnh nhân / gene: 1018 / 1018 / 19938
- Ngày tạo (UTC): 2026-10-11T13:05:58+00:00
- Commit: `no-git-repo` (nhap tay --commit, scripts/patch_v2_tcga_stage.py) — notebook `notebooks/00_data/lung_datasets_v2_patient_folds.ipynb`
- Seed gốc: 42; seed chia được chọn: 139 (tìm trong 42..141)

## Tiêu chí lọc mẫu

1. Bắt đầu từ mọi file GDC STAR - Counts của TCGA-LUAD và TCGA-LUSC.
2. Chỉ file đã tải được ma trận biểu hiện.
3. **Chỉ mẫu Primary Tumor** (`sample_type = Primary Tumor`, mã barcode `01`). Loại Solid Tissue Normal, Recurrent Tumor, Metastatic, Additional - New Primary.
4. Loại bệnh nhân có mẫu ở cả hai project (nếu có).
5. Giữ 1 mẫu u / bệnh nhân: ưu tiên mẫu không FFPE, sau đó `sample_id` nhỏ nhất (vd `01A` trước `01B`).
6. Nhãn lấy theo project: TCGA-LUAD → LUAD, TCGA-LUSC → LUSC. Không yêu cầu nhãn giai đoạn.

## Số mẫu trước / sau từng bước lọc

| step | buoc | tieu_chi | n_samples | n_patients | n_removed |
|---|---|---|---|---|---|
| 0 | 0_gdc_star_counts | Moi file STAR - Counts cua TCGA-LUAD va TCGA-LUSC | 1163 | 1019 | 0 |
| 1 | 1_co_bieu_hien | Chi file da tai duoc ma tran bieu hien | 1163 | 1019 | 0 |
| 2 | 2_chi_primary_tumor | Chi mau Primary Tumor (sample_type = 'Primary Tumor' va ma barcode 01); loai mo thuong, Recurrent, Metastatic, Additional - New Primary | 1051 | 1018 | 112 |
| 3 | 3_loai_benh_nhan_2_project | Loai benh nhan co mau o ca TCGA-LUAD va TCGA-LUSC | 1051 | 1018 | 0 |
| 4 | 4_mot_mau_moi_benh_nhan | Giu 1 mau u / benh nhan (uu tien khong FFPE, roi sample_id nho nhat) | 1018 | 1018 | 33 |

## Nhãn

| label | class | n_samples | n_patients |
|---|---|---|---|
| 0 | LUAD | 517 | 517 |
| 1 | LUSC | 501 | 501 |

## Ma trận biểu hiện

- Giá trị `log2(TPM + 1)` từ cột `tpm_unstranded` của GDC STAR - Counts.
- Chỉ gene `protein_coding`; Ensembl ID đổi sang `gene_name`, trùng tên → giữ gene có phương sai lớn nhất (tính trên các mẫu của bộ này, không dùng nhãn).
- Chưa chuẩn hoá / scale; chưa hiệu chỉnh batch.

## Chia fold theo bệnh nhân

- `StratifiedGroupKFold` với `groups = patient_id`, phân tầng theo lớp (LUAD/LUSC).
- Mỗi bệnh nhân nằm trọn trong một phần; trong mỗi fold, bệnh nhân của train/val/test đôi một rời nhau (đã kiểm tra bằng `assert`).

| fold | train | val | test |
|---|---|---|---|
| fold_1 | P3+P4+P5 | P2 | P1 |
| fold_2 | P1+P4+P5 | P3 | P2 |
| fold_3 | P1+P2+P5 | P4 | P3 |
| fold_4 | P1+P2+P3 | P5 | P4 |
| fold_5 | P2+P3+P4 | P1 | P5 |

Số mẫu (số bệnh nhân) theo lớp:

| fold | split | LUAD | LUSC | TONG |
|---|---|---|---|---|
| all | all | 517 (517) | 501 (501) | 1018 (1018) |
| part | P1 | 103 (103) | 101 (101) | 204 (204) |
| part | P2 | 104 (104) | 100 (100) | 204 (204) |
| part | P3 | 103 (103) | 100 (100) | 203 (203) |
| part | P4 | 104 (104) | 100 (100) | 204 (204) |
| part | P5 | 103 (103) | 100 (100) | 203 (203) |
| fold_1 | train | 310 (310) | 300 (300) | 610 (610) |
| fold_1 | val | 104 (104) | 100 (100) | 204 (204) |
| fold_1 | test | 103 (103) | 101 (101) | 204 (204) |
| fold_2 | train | 310 (310) | 301 (301) | 611 (611) |
| fold_2 | val | 103 (103) | 100 (100) | 203 (203) |
| fold_2 | test | 104 (104) | 100 (100) | 204 (204) |
| fold_3 | train | 310 (310) | 301 (301) | 611 (611) |
| fold_3 | val | 104 (104) | 100 (100) | 204 (204) |
| fold_3 | test | 103 (103) | 100 (100) | 203 (203) |
| fold_4 | train | 310 (310) | 301 (301) | 611 (611) |
| fold_4 | val | 103 (103) | 100 (100) | 203 (203) |
| fold_4 | test | 104 (104) | 100 (100) | 204 (204) |
| fold_5 | train | 311 (311) | 300 (300) | 611 (611) |
| fold_5 | val | 103 (103) | 101 (101) | 204 (204) |
| fold_5 | test | 103 (103) | 100 (100) | 203 (203) |

## Nhãn giai đoạn (cho bài toán phụ)

<!-- patch_v2_tcga_stage:start -->
Metadata được vá bằng `scripts/patch_v2_tcga_stage.py` tại commit `no-git-repo`; ma trận biểu hiện và fold không đổi (sha256 của `expression.h5`, `fold_assignment.csv`, `folds/` và `experiment-data-v2/h5/` giống hệt trước/sau khi chạy script).

**Lỗi được vá.** Bản v2 gốc chỉ đọc `diagnoses[0]` của mỗi case. GDC trả về nhiều bản ghi `diagnoses` cho một case (u chỉ điểm, *Prior primary*, *Synchronous primary*, *metastasis*...) theo thứ tự không cố định, nên 121 LUAD / 85 LUSC bị mất giai đoạn và `primary_diagnosis` có khi lấy từ khối u khác. Cột cũ được giữ nguyên dưới tên `stage_raw_v2_original` và `primary_diagnosis_v2_original` (không dùng cho phân tích).

- Nguồn: GDC API `POST https://api.gdc.cancer.gov/cases`, lọc `submitter_id` ∈ 1018 `patient_id` của bộ này; lấy mọi bản ghi `diagnoses` với các trường `ajcc_pathologic_stage`, `ajcc_staging_system_edition`, `ajcc_pathologic_t/n/m`, `primary_diagnosis`, `diagnosis_is_primary_disease`, `classification_of_tumor` (tên trường đối chiếu GDC Data Dictionary, node `diagnosis`, và `GET /cases/_mapping`).
- Ngày truy vấn (UTC): 2026-10-14T03:51:51+00:00; GDC Data Release 46.0 - August 10, 2026.
- JSON thô: `gdc_diagnoses_raw.json` (cùng thư mục).
- Quy tắc chọn giai đoạn cho mỗi bệnh nhân:
  1. Ứng viên = bản ghi có `ajcc_pathologic_stage` khác rỗng và khác `Not Reported`.
  2. Không có ứng viên → không có giai đoạn.
  3. Mọi ứng viên cùng giá trị → lấy giá trị đó (`T/N/M`, `edition` lấy từ bản ghi u chỉ điểm nếu có).
  4. Ứng viên có giá trị khác nhau → `stage_conflict = True`; lấy bản ghi u chỉ điểm (`diagnosis_is_primary_disease = True`, `classification_of_tumor = primary`) nếu có đúng một. Các bản ghi còn lại là khối u khác của cùng bệnh nhân (*Prior/Synchronous primary*).
- `stage_group`: `Stage IA/IB/I` → I, `IIA/IIB/II` → II, `IIIA/IIIB/IIIC/III` → III, `IV/IVA/IVB` → IV.
- Cột mới trong `metadata.csv`: `stage_raw`, `stage_group`, `ajcc_edition`, `path_t`, `path_n`, `path_m`, `stage_conflict`, `stage_rule` (`single` / `conflict_index_tumor` / `none`), `stage_diagnosis_submitter_id`, `primary_diagnosis` (bản ghi u chỉ điểm), `diagnosis_mismatch`.

Số khối u theo giai đoạn:

| project | I | II | III | IV | thiếu | tổng |
|---|---|---|---|---|---|---|
| TCGA-LUAD | 276 | 122 | 85 | 26 | 8 | 517 |
| TCGA-LUSC | 244 | 162 | 84 | 7 | 4 | 501 |
| tổng | 520 | 284 | 169 | 33 | 12 | 1018 |

Theo phiên bản AJCC (`ajcc_edition`):

| project | stage_group | 3rd | 4th | 5th | 6th | 7th | không ghi | tổng |
|---|---|---|---|---|---|---|---|---|
| TCGA-LUAD | I | 3 | 1 | 14 | 96 | 154 | 8 | 276 |
| TCGA-LUAD | II | 0 | 2 | 9 | 34 | 74 | 3 | 122 |
| TCGA-LUAD | III | 0 | 2 | 6 | 43 | 27 | 7 | 85 |
| TCGA-LUAD | IV | 0 | 0 | 1 | 13 | 10 | 2 | 26 |
| TCGA-LUAD | thiếu | 0 | 0 | 0 | 0 | 0 | 8 | 8 |
| TCGA-LUSC | I | 0 | 1 | 16 | 70 | 102 | 55 | 244 |
| TCGA-LUSC | II | 2 | 0 | 5 | 45 | 86 | 24 | 162 |
| TCGA-LUSC | III | 0 | 0 | 2 | 39 | 28 | 15 | 84 |
| TCGA-LUSC | IV | 0 | 0 | 1 | 1 | 3 | 2 | 7 |
| TCGA-LUSC | thiếu | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| tổng |  | 5 | 6 | 54 | 341 | 484 | 128 | 1018 |

- Thiếu giai đoạn: 12 ca (TCGA-LUAD 8, TCGA-LUSC 4): `TCGA-22-5473`, `TCGA-38-4626`, `TCGA-46-3769`, `TCGA-50-5045`, `TCGA-52-7812`, `TCGA-55-5899`, `TCGA-64-1678`, `TCGA-67-4679`, `TCGA-69-7765`, `TCGA-69-8254`, `TCGA-73-4677`, `TCGA-92-8064`. Không bản ghi `diagnoses` nào có giai đoạn.
- `stage_conflict = True`: 25 ca (TCGA-LUAD 13, TCGA-LUSC 12); 25 ca đã lấy giai đoạn của u chỉ điểm. Nên chạy lại phân tích giai đoạn khi bỏ các ca này (phân tích độ nhạy).
- Đối chiếu v1 (`experiment-data/h5/tcga_{luad,lusc}/fold_assignment.csv`): 1006 bệnh nhân có giai đoạn ở cả hai bản; 981 ca không conflict trùng `stage_group` 100%. Trong 25 ca conflict, 7 ca lệch v1 vì v1 lấy bản ghi đầu tiên còn bản vá lấy u chỉ điểm: `TCGA-18-3407` (I vs v1 III), `TCGA-38-7271` (I vs v1 II), `TCGA-43-8118` (I vs v1 II), `TCGA-50-5932` (II vs v1 I), `TCGA-55-A492` (I vs v1 II), `TCGA-NC-A5HE` (II vs v1 III), `TCGA-NC-A5HR` (II vs v1 I). Nếu 7 ca này lấy theo v1, số ca theo giai đoạn trùng đúng v1 (TCGA-LUAD 509 (275/123/85/26), TCGA-LUSC 497 (243/161/86/7)).

**`primary_diagnosis` (bản ghi u chỉ điểm) theo project:**

| project | primary_diagnosis | n |
|---|---|---|
| TCGA-LUAD | Acinar cell carcinoma | 24 |
| TCGA-LUAD | Adenocarcinoma with mixed subtypes | 107 |
| TCGA-LUAD | Adenocarcinoma, NOS | 312 |
| TCGA-LUAD | Bronchio-alveolar carcinoma, mucinous | 5 |
| TCGA-LUAD | Bronchiolo-alveolar adenocarcinoma, NOS | 3 |
| TCGA-LUAD | Bronchiolo-alveolar carcinoma, non-mucinous | 19 |
| TCGA-LUAD | Clear cell adenocarcinoma, NOS | 2 |
| TCGA-LUAD | Invasive micropapillary carcinoma | 3 |
| TCGA-LUAD | Mucinous adenocarcinoma | 14 |
| TCGA-LUAD | Papillary adenocarcinoma, NOS | 22 |
| TCGA-LUAD | Signet ring cell carcinoma | 1 |
| TCGA-LUAD | Solid carcinoma, NOS | 5 |
| TCGA-LUSC | Adenocarcinoma, NOS | 1 |
| TCGA-LUSC | Basaloid squamous cell carcinoma | 14 |
| TCGA-LUSC | Papillary squamous cell carcinoma | 4 |
| TCGA-LUSC | Squamous cell carcinoma, NOS | 465 |
| TCGA-LUSC | Squamous cell carcinoma, keratinizing, NOS | 13 |
| TCGA-LUSC | Squamous cell carcinoma, large cell, nonkeratinizing, NOS | 3 |
| TCGA-LUSC | Squamous cell carcinoma, small cell, nonkeratinizing | 1 |

- Không khớp project (`diagnosis_mismatch = True`: LUAD mà chẩn đoán tế bào vảy, hoặc LUSC mà chẩn đoán không phải tế bào vảy): 1 ca: `TCGA-43-8118` (TCGA-LUSC, Adenocarcinoma, NOS). Dùng cho phân tích độ nhạy (bỏ các ca này rồi đánh giá lại LUAD vs LUSC).
- Với cột cũ `primary_diagnosis_v2_original` (`diagnoses[0]`) sẽ có 56 ca "không khớp" và 84 ca `Not Reported` — phần lớn là chẩn đoán của khối u khác (vd "Squamous cell carcinoma, NOS" trong TCGA-LUAD), không phải lỗi nhãn.
<!-- patch_v2_tcga_stage:end -->

## Tệp

- `expression.h5` (key `expression`): mẫu × gene, float32, index `sample_id`.
- `metadata.csv`: metadata đã lọc của các mẫu được giữ.
- `fold_assignment.csv`: `part` và vai trò train/val/test của từng mẫu trong `fold_1..fold_5`.
- `class_stats.csv`: số bệnh nhân và số mẫu theo lớp trong từng phần / từng tập.
- `filter_log.csv`: số mẫu trước / sau từng bước lọc.
- `config.json`: seed, commit, tiêu chí lọc, ánh xạ nhãn, phiên bản thư viện.
- `folds/fold_<r>/{train,val,test}.h5`: bảng mẫu × (gene..., `label`) dựng sẵn từ `fold_assignment.csv`.

## Lưu ý và hạn chế

- Nhãn theo project TCGA, không xét lại `primary_diagnosis` (được giữ trong `metadata.csv` để tham khảo).
- Mỗi bệnh nhân chỉ có 1 mẫu nên chia theo bệnh nhân tương đương chia theo mẫu; vẫn dùng `groups` để nhất quán.
- Có thể có hiệu ứng trung tâm thu mẫu (TSS) và batch; chưa hiệu chỉnh.
