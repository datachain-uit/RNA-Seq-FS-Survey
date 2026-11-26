# Data card: GSE10245

This card is the English version of the data card kept by the authors while the cohort was built. The original,
in Vietnamese, is at the same path in commit `d916062` of this repository. The record files named here (`excluded_samples_*.csv`,
`qc/*.csv`) keep the wording of that time; `docs/RECORDS_GLOSSARY.md` translates it. The rules are named there by
the numbers of the authors' decision log: QĐ21 (near-duplicate pair of GSE37745), QĐ22 (two-axis sex check of the
GEO cohorts) and QĐ22b (per-gene Y vote of TCGA).

## Update of 2 November 2026: label cross-check, duplicate screen, coverage of the panel genes

### Cross-check of the labels against the original series matrix

- Method: `scripts/check_label_crosscheck.py` reads `GSE10245_series_matrix.txt.gz` (folder
  `geo_target_work/raw/GSE10245/`), takes the field `disease state` of `!Sample_characteristics_ch1` by its key (not by
  its column position) and maps it by exact string match through `label_mapping.json`, without regular expressions:
  `disease state: adenocarcinoma` → 0; `disease state: squamous cell carcinoma` → 1.
- Result: passed. Series matrix 58 samples → LUAD/LUSC 40/18 (equal to `n_samples` in `label_mapping.json`) → minus
  the 3 samples of `excluded_samples_gse10245.csv` → 37/18, equal to `DATASET_EXPECTED_N` 37/18. The set of samples is
  identical to `metadata.csv` (no sample on one side only) and the labels agree for 100 % of the samples (0 mismatch).
- Files: `qc/label_crosscheck_gse10245.csv` (one row per sample of the series matrix: original value, derived label,
  status `giu` = kept / `cach_ly_QC` = excluded by quality control / `khong_thuoc_bai_toan` = not part of the task,
  label in the metadata, column `match`); summary in `qc/label_crosscheck_summary.csv`. Run on 2 November 2026.

### Duplicate-sample screen (r_z, 29 October 2026)

- `scripts/geo_sample_qc.py` (configuration `qc/rz_screen_config.json`, commit `5facab3`, run on 29 October on the
  cohort before quality control): r_z is the Pearson correlation of two samples after z-scoring each gene within the
  cohort; threshold r_z ≥ 0.90, together with raw r ≥ 0.99.
- Within the cohort: 1,653 pairs on 20,848 genes; largest r_z 0.4795, 99.9th percentile 0.4522; 0 pairs with
  r_z ≥ 0.90; largest raw r 0.9608 (0 pairs with r ≥ 0.99).
- Between cohorts (17,062 genes shared by the four GEO cohorts): largest r_z 0.4828 (with GSE30219); 0 pairs with
  r_z ≥ 0.90 with any other cohort.

### patient_id

- `metadata.csv` has a column `patient_id` (= `sample_id`), so `load_dataset` records
  `patient_id_assumed_from_sample_id = false`. Each sample is taken as one patient because the titles are unique
  (55/55), the r_z screen found no duplicate pair, and the sex check excluded the samples whose identity could not be
  verified. Folds grouped by `patient_id` are therefore equivalent to folds by sample.

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

- n before → after: 58 (40/18) → 55 (37/18) LUAD/LUSC; `DATASET_EXPECTED_N` = (37, 18) (commit `380d01f`).
- Rebuild of the cohort files: `scripts/geo_sample_qc.py rebuild` at commit `380d01f`
  (2026-10-29T13:24:02+07:00). The rebuild only removes the samples of `excluded_samples_gse10245.csv` and adds the
  column `sex`; every other value is unchanged (checked by assertions in the script). Previous version:
  `_backup_20261029-132402/` (not in this repository).
- New SHA-256: `expression.h5` `7cb29e07d6c1489dbd90ad81ebfa9675548bac729289f70cf04766f2119226ea`; `metadata.csv`
  `571f9d0ce2fc78275f41d09aa57a71f7a49ac3c6f1814c5b6f5b0002c12c2e26`.
- Splits (`build_v3_splits.py --force`, commit `380d01f`): `repeated_splits.csv`
  `823637702262a5fea795a52e9d2972f7857d9b75156d0667b1f7cac769496d5e` (1375 rows); `fewshot_splits.csv`
  `b31efa81e03b21454e23d455d58e05100234fd1fd09029e1d655cff22d5b30d5` (n_T: 5, 10, pool; 5 splits × 4 subsets).
- GEO source: GSE10245, "Gene expression differences between adenocarcinoma and squamous cell carcinoma in human
  NSCLC"; submitted 23/01/2008, last updated 25/03/2019 (fields `!Series_*` of the series matrix); original article
  PMID 18486272; series matrix downloaded to `geo_target_work/raw/GSE10245/` on 17/10/2026 09:19 (UTC+7) (file time
  on the authors' machine).
- Source of sex: the series matrix of GSE10245 reports no sex in `!Sample_characteristics_ch1`, so the column `sex`
  is empty and the samples are classified by expression only (both-high is still excluded).
- Rule (QĐ22, two axes): XIST and the mean of five Y genes (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY); each axis is split
  into two k-means clusters within the cohort. Opposite (both axes contradict the metadata) and inconsistent (XIST and
  the Y genes both high) → excluded; inconclusive (both low, only one axis contradicts, or an axis does not separate)
  → kept and flagged.
- Thresholds: XIST 7.02; Y axis 5.70 (log2).
- Result: no-metadata 55, inconsistent 3.

### Excluded samples (3)

Sex check (QĐ22). Opposite: the sex contradicts the metadata, and the histology label comes from the same metadata
row, so it cannot be trusted. Inconsistent: the identity of the sample cannot be verified.

| Sample | Class | Sex (metadata) | XIST | Y axis (mean of 5 genes) | category | verdict |
|---|---|---|---|---|---|---|
| GSM258572 | LUAD | not reported | 12.94 | 7.88 | both-high | inconsistent |
| GSM258574 | LUAD | not reported | 10.34 | 8.21 | both-high | inconsistent |
| GSM258593 | LUAD | not reported | 12.97 | 6.97 | both-high | inconsistent |

### Samples kept and flagged as inconclusive (0)

None.

---

*The card before the quality control of 29 October follows (kept for traceability; its sample counts and SHA-256
are those of the previous version).*

- Role: primary GPL570 target cohort for LUAD versus LUSC.
- Source: GEO GSE10245 series matrix; platform GPL570.
- Included samples: 58 (LUAD 40, LUSC 18); one sample per patient, with `patient_id = sample_id`.
- Expression: 20,848 gene-symbol columns. The source series matrix states gcRMA normalization; 16,479 genes intersect
  TCGA. NKX2-1, NAPSA, TP63 and KRT5 are present.
- Label source: `Sample_characteristics_ch1`; the explicit mapping is in `label_mapping.json`.
- Input SHA-256: `expression.h5` `0ec3ed892ba76a4bf212a0ea0ae9b8b0a93771e614922ce6a54a12181392ce15`; `metadata.csv`
  `16f7dfedf06e2cbab396fd6ffb3d81858cd788beda27f56b190101ae5f134afe`.
- Audit 2026-10-27: metadata IDs match expression IDs and the 58 source GEO IDs; no shared sample ID with the other
  target cohorts.

The historical fold files are not the splits of the benchmark. The splits used in the paper are those of the update
of 29 October above.
