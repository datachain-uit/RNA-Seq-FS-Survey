# Data card: GSE30219

This card is the English version of the data card kept by the authors while the cohort was built. The original,
in Vietnamese, is at the same path in commit `d916062` of this repository. The record files named here (`excluded_samples_*.csv`,
`qc/*.csv`) keep the wording of that time; `docs/RECORDS_GLOSSARY.md` translates it. The rules are named there by
the numbers of the authors' decision log: QĐ21 (near-duplicate pair of GSE37745), QĐ22 (two-axis sex check of the
GEO cohorts) and QĐ22b (per-gene Y vote of TCGA).

## Update of 2 November 2026: label cross-check, duplicate screen, coverage of the panel genes

### Cross-check of the labels against the original series matrix

- Method: `scripts/check_label_crosscheck.py` reads `GSE30219_series_matrix.txt.gz` (folder
  `geo_target_work/raw/GSE30219/`), takes the field `histology` of `!Sample_characteristics_ch1` by its key (not by
  its column position) and maps it by exact string match through `label_mapping.json`, without regular expressions:
  `histology: ADC` → 0; `histology: SQC` → 1.
- Result: passed. Series matrix 307 samples → LUAD/LUSC 85/61 (equal to `n_samples` in `label_mapping.json`) → minus
  the 5 samples of `excluded_samples_gse30219.csv` → 81/60, equal to `DATASET_EXPECTED_N` 81/60. The set of samples is
  identical to `metadata.csv` (no sample on one side only) and the labels agree for 100 % of the samples (0 mismatch).
- Label values outside the task (excluded because they do not match exactly): histology: BAS (39); histology: CARCI
  (24); histology: LCC (3); histology: LCNE (56); histology: NTL (14); histology: Other (4); histology: SCC (21). In
  GSE30219, `SCC` denotes small-cell lung carcinoma, and squamous cell carcinoma is recorded as `SQC`.
- Files: `qc/label_crosscheck_gse30219.csv` (one row per sample of the series matrix: original value, derived label,
  status `giu` = kept / `cach_ly_QC` = excluded by quality control / `khong_thuoc_bai_toan` = not part of the task,
  label in the metadata, column `match`); summary in `qc/label_crosscheck_summary.csv`. Run on 2 November 2026.

### Duplicate-sample screen (r_z, 29 October 2026)

- `scripts/geo_sample_qc.py` (configuration `qc/rz_screen_config.json`, commit `5facab3`, run on 29 October on the
  cohort before quality control): r_z is the Pearson correlation of two samples after z-scoring each gene within the
  cohort; threshold r_z ≥ 0.90, together with raw r ≥ 0.99.
- Within the cohort: 10,585 pairs on 20,848 genes; largest r_z 0.5721, 99.9th percentile 0.4824; 0 pairs with
  r_z ≥ 0.90; largest raw r 0.9818 (0 pairs with r ≥ 0.99).
- Between cohorts (17,062 genes shared by the four GEO cohorts): largest r_z 0.5074 (with GSE37745); 0 pairs with
  r_z ≥ 0.90 with any other cohort.

### patient_id

- `metadata.csv` has a column `patient_id` (= `sample_id`), so `load_dataset` records
  `patient_id_assumed_from_sample_id = false`. Each sample is taken as one patient because the titles are unique
  (141/141), the r_z screen found no duplicate pair, and the sex check excluded the samples whose identity could not be
  verified. Folds grouped by `patient_id` are therefore equivalent to folds by sample.

### Two builds of GSE30219

- The history holds two builds: the build before R5 and the corrected R5 build (made by
  `scripts/rebuild_gse30219_r5.py`; its folder once had the suffix `_r5_fixed` and is now `gse30219`). Every session of
  the benchmark used the R5 build (141 samples after quality control). `DATASET_EXPECTED_N` = (81, 60) are the counts of
  this build; `load_dataset` warns when the counts differ (and stops when `KLTN_STRICT_N=1`).
- No script that compares the two builds is released. Instead, the label cross-check above compares the R5 build
  directly with the original series matrix (307 samples): only `histology: ADC` and `histology: SQC` are accepted, and
  the sets of samples and labels agree 100 %.

### Coverage of the genes of the reference panels

Platform GPL570. The gene space of E1 and E1b is the set of genes of the cohort that are also TCGA genes
(`load_dataset(gene_space="reference")`). A panel is evaluated with its genes that are present; the `config.json` of E1
lists the missing genes and sets `complete = false` (`qc/panel_gene_coverage.csv`, `scripts/check_panel_coverage.py`).

| Panel | Present / total (genes of the platform) | Present / total (gene space of E1/E1b) | Missing genes |
|---|---|---|---|
| IHC-2 | 2/2 | 2/2 | — |
| IHC-4 | 4/4 | 4/4 | — |
| Girard 2016 (42 genes) | 39/42 | 39/42 | ABCC6, KRT6C, KRT17 |
| Wilkerson 2013 (52 genes) | 50/52 | 50/52 | CLEC3B, TCP1 |

Note on gene symbols (3 November 2026): Wilkerson et al. (2013, Table 1) list TUBA1. The benchmark uses TUBA4A,
following HGNC (HGNC:12407, previous symbol TUBA1; looked up on 3 November 2026), without a check against the PCR
primers of the original article.

The missing genes have no column in the `expression.h5` of the GEO build. The cause lies in the probe-to-gene step of
the build and was not traced back to the probes. The build was not changed to recover them.

---

## Update of 29 October 2026: sample quality control (sample counts used in the paper)

- n before → after: 146 (85/61) → 141 (81/60) LUAD/LUSC; `DATASET_EXPECTED_N` = (81, 60) (commit `380d01f`).
- Rebuild of the cohort files: `scripts/geo_sample_qc.py rebuild` at commit `380d01f`
  (2026-10-29T13:24:02+07:00). The rebuild only removes the samples of `excluded_samples_gse30219.csv` and adds the
  column `sex`; every other value is unchanged (checked by assertions in the script). Previous version:
  `_backup_20261029-132402/` (not in this repository).
- New SHA-256: `expression.h5` `6c2f7377ad802a7df3d804473884ad64fa08beadfa44ecb5232343801230e2d9`; `metadata.csv`
  `570923f3f20216def97dfc47ca06a2e78f30669f4b7829f59b6426d7a32501bc`.
- Splits (`build_v3_splits.py --force`, commit `380d01f`): `repeated_splits.csv`
  `de92722bd27a853fef77abe4fda22bcd64886ef2724c298ac386b0bfd9b979bc` (3525 rows); `fewshot_splits.csv`
  `88667258ebb2d5150a45971e121bbd8d36b2a722f4986a9d2d48dc3ae48aeb7b` (n_T: 5, 10, 20, 40, pool; 5 splits × 4 subsets).
- GEO source: GSE30219, ""Off-context" gene expression in lung cancer identifies a group of metastatic-prone tumors";
  submitted 26/06/2011, last updated 25/03/2019 (fields `!Series_*` of the series matrix); original article
  PMID 23698379; series matrix downloaded to `geo_target_work/raw/GSE30219/` on 17/10/2026 02:31 (UTC+7) (file time
  on the authors' machine).
- Source of sex: the field `gender:` of `!Sample_characteristics_ch1` in the series matrix, read by its key (checked
  directly in the series matrix: no column shift).
- Rule (QĐ22, two axes): XIST and the mean of five Y genes (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY); each axis is split
  into two k-means clusters within the cohort. Opposite (both axes contradict the metadata) and inconsistent (XIST and
  the Y genes both high) → excluded; inconclusive (both low, only one axis contradicts, or an axis does not separate)
  → kept and flagged.
- Thresholds: XIST 7.72; Y axis 5.92 (log2).
- Result: match 132, inconclusive 9, opposite 1, inconsistent 4.

### Excluded samples (5)

Sex check (QĐ22). Opposite: the sex contradicts the metadata, and the histology label comes from the same metadata
row, so it cannot be trusted. Inconsistent: the identity of the sample cannot be verified.

| Sample | Class | Sex (metadata) | XIST | Y axis (mean of 5 genes) | category | verdict |
|---|---|---|---|---|---|---|
| GSM748065 | LUSC | male | 9.01 | 7.03 | both-high | inconsistent |
| GSM748078 | LUAD | male | 13.05 | 3.52 | F-like | opposite |
| GSM748240 | LUAD | male | 12.34 | 6.38 | both-high | inconsistent |
| GSM748278 | LUAD | male | 8.77 | 6.95 | both-high | inconsistent |
| GSM748279 | LUAD | male | 8.91 | 8.22 | both-high | inconsistent |

### Samples kept and flagged as inconclusive (9)

| Sample | Class | Sex (metadata) | XIST | Y axis (mean of 5 genes) | category | verdict |
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

Note: `excluded_samples.csv` (earlier, R5 build) lists the samples that are not ADC or SQC;
`excluded_samples_gse30219.csv` (new) lists the samples excluded by quality control.

---

*The card before the quality control of 29 October follows (kept for traceability; its sample counts and SHA-256
are those of the previous version).*

- Role: primary GPL570 target cohort for LUAD versus LUSC.
- Source: GEO GSE30219 series matrix; platform GPL570; R5 corrected reconstruction.
- Included samples: 146 (LUAD 85, LUSC 61). Only exact `ADC` and `SQC` records are included; other histologies are
  excluded in `excluded_samples.csv`.
- Expression: 20,848 gene-symbol columns. The source series matrix states RMA normalization; 16,479 genes intersect
  TCGA. NKX2-1, NAPSA, TP63 and KRT5 are present.
- Label source: `histology_raw`; the explicit mapping is in `label_mapping.json`.
- Input SHA-256: `expression.h5` `622162531d63db0f977cb73217fefe5ac6fc802ca31ce55777015791334d3b92`; `metadata.csv`
  `c36345825cbdbd60673c9a96d2e1cafea2af21c869cf1a9c39652a2e09c3eea0`.
- Audit 2026-10-27: metadata IDs match expression IDs and are all present in the 307-sample source GEO matrix; no
  shared sample ID with the other target cohorts.

The historical fold files are not the splits of the benchmark. The splits used in the paper are those of the update
of 29 October above.
