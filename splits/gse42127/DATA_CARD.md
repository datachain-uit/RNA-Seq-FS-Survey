# Data card: GSE42127

This card is the English version of the data card kept by the authors while the cohort was built. The original,
in Vietnamese, is at the same path in commit `d916062` of this repository. The record files named here (`excluded_samples_*.csv`,
`qc/*.csv`) keep the wording of that time; `docs/RECORDS_GLOSSARY.md` translates it. The rules are named there by
the numbers of the authors' decision log: QĐ21 (near-duplicate pair of GSE37745), QĐ22 (two-axis sex check of the
GEO cohorts) and QĐ22b (per-gene Y vote of TCGA).

## Update of 2 November 2026: label cross-check, duplicate screen, coverage of the panel genes

### Cross-check of the labels against the original series matrix

- Method: `scripts/check_label_crosscheck.py` reads `GSE42127_series_matrix.txt.gz` (folder
  `geo_target_work/raw/GSE42127/`), takes the field `histology` of `!Sample_characteristics_ch1` by its key (not by
  its column position) and maps it by exact string match through `label_mapping.json`, without regular expressions:
  `histology: Adenocarcinoma` → 0; `histology: Squamous` → 1.
- Result: passed. Series matrix 176 samples → LUAD/LUSC 133/43 (equal to `n_samples` in `label_mapping.json`) → minus
  the 9 samples of `excluded_samples_gse42127.csv` → 126/41, equal to `DATASET_EXPECTED_N` 126/41. The set of samples
  is identical to `metadata.csv` (no sample on one side only) and the labels agree for 100 % of the samples
  (0 mismatch).
- Files: `qc/label_crosscheck_gse42127.csv` (one row per sample of the series matrix: original value, derived label,
  status `giu` = kept / `cach_ly_QC` = excluded by quality control / `khong_thuoc_bai_toan` = not part of the task,
  label in the metadata, column `match`); summary in `qc/label_crosscheck_summary.csv`. Run on 2 November 2026.

### Duplicate-sample screen (r_z, 29 October 2026)

- `scripts/geo_sample_qc.py` (configuration `qc/rz_screen_config.json`, commit `5facab3`, run on 29 October on the
  cohort before quality control): r_z is the Pearson correlation of two samples after z-scoring each gene within the
  cohort; threshold r_z ≥ 0.90, together with raw r ≥ 0.99.
- Within the cohort: 15,400 pairs on 19,405 genes; largest r_z 0.5770, 99.9th percentile 0.4366; 0 pairs with
  r_z ≥ 0.90; largest raw r 0.9662 (0 pairs with r ≥ 0.99).
- Between cohorts (17,062 genes shared by the four GEO cohorts): largest r_z 0.4579 (with GSE30219); 0 pairs with
  r_z ≥ 0.90 with any other cohort.

### patient_id

- `metadata.csv` has a column `patient_id` (= `sample_id`), so `load_dataset` records
  `patient_id_assumed_from_sample_id = false`. Each sample is taken as one patient because the titles give a sample
  code ("… sample 250-T") and, after removing the suffixes -T/-T1/-T2/-Tx, the 176 codes are distinct across the whole
  series; the r_z screen found no duplicate pair, and the sex check excluded the samples whose identity could not be
  verified. Folds grouped by `patient_id` are therefore equivalent to folds by sample.

### Coverage of the genes of the reference panels

Platform GPL6884. The gene space of E1 and E1b is the set of genes of the cohort that are also TCGA genes
(`load_dataset(gene_space="reference")`). A panel is evaluated with its genes that are present; the `config.json` of E1
lists the missing genes and sets `complete = false` (`qc/panel_gene_coverage.csv`, `scripts/check_panel_coverage.py`).

| Panel | Present / total (genes of the platform) | Present / total (gene space of E1/E1b) | Missing genes |
|---|---|---|---|
| IHC-2 | 2/2 | 2/2 | — |
| IHC-4 | 4/4 | 4/4 | — |
| Girard 2016 (42 genes) | 41/42 | 41/42 | PNCK |
| Wilkerson 2013 (52 genes) | 52/52 | 52/52 | — |

Note on gene symbols (3 November 2026): Wilkerson et al. (2013, Table 1) list TUBA1. The benchmark uses TUBA4A,
following HGNC (HGNC:12407, previous symbol TUBA1; looked up on 3 November 2026), without a check against the PCR
primers of the original article.

---

## Update of 29 October 2026: sample quality control (sample counts used in the paper)

- n before → after: 176 (133/43) → 167 (126/41) LUAD/LUSC; `DATASET_EXPECTED_N` = (126, 41) (commit `380d01f`).
- Rebuild of the cohort files: `scripts/geo_sample_qc.py rebuild` at commit `380d01f`
  (2026-10-29T13:24:03+07:00). The rebuild only removes the samples of `excluded_samples_gse42127.csv` and adds the
  column `sex`; every other value is unchanged (checked by assertions in the script). Previous version:
  `_backup_20261029-132403/` (not in this repository).
- New SHA-256: `expression.h5` `6595946619c4a39def08b957725a2883358bbc8228ff0626799444ceb91532c8`; `metadata.csv`
  `8b18fd7f004f6fdfacbb4fbc8b93f89def8384152f559cdaad1c388570fe3d55`.
- Splits (`build_v3_splits.py --force`, commit `380d01f`): `repeated_splits.csv`
  `6a0f194aa586b94f3836019b273d12ac2bc8abbdfe0151e612c876be073fd959` (4175 rows); `fewshot_splits.csv`
  `5cfec85ebf3b88981ba081114fdd2777ff6d65dc9085dac31859ecaeb34a5f01` (n_T: 5, 10, 20, pool; 5 splits × 4 subsets).
- GEO source: GSE42127, "Expression data for non-small-cell lung cancer"; submitted 07/11/2012, last updated
  22/06/2020 (fields `!Series_*` of the series matrix); original article PMID 23357979 (the series also lists
  32512502); series matrix downloaded to `geo_target_work/raw/GSE42127/` on 17/10/2026 09:21 (UTC+7) (file time on
  the authors' machine).
- Source of sex: the field `gender:` of `!Sample_characteristics_ch1` in the series matrix, read by its key (checked
  directly in the series matrix: no column shift).
- Rule (QĐ22, two axes): XIST and the mean of five Y genes (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY); each axis is split
  into two k-means clusters within the cohort. Opposite (both axes contradict the metadata) and inconsistent (XIST and
  the Y genes both high) → excluded; inconclusive (both low, only one axis contradicts, or an axis does not separate)
  → kept and flagged.
- Thresholds: XIST 7.07; Y axis 5.80 (log2).
- Result: match 161, inconclusive 6, opposite 7, inconsistent 2.

### Excluded samples (9)

Sex check (QĐ22). Opposite: the sex contradicts the metadata, and the histology label comes from the same metadata
row, so it cannot be trusted. Inconsistent: the identity of the sample cannot be verified.

| Sample | Class | Sex (metadata) | XIST | Y axis (mean of 5 genes) | category | verdict |
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

Note for the limitations: GSE42127 loses 9 of its 176 samples (5.1 %), 7 opposite and 2 inconsistent; the list and
the values are in the table above.

### Samples kept and flagged as inconclusive (6)

| Sample | Class | Sex (metadata) | XIST | Y axis (mean of 5 genes) | category | verdict |
|---|---|---|---|---|---|---|
| GSM1032926 | LUSC | male | 4.98 | 5.14 | both-low | inconclusive |
| GSM1032940 | LUAD | male | 5.51 | 5.38 | both-low | inconclusive |
| GSM1032960 | LUSC | male | 4.67 | 5.55 | both-low | inconclusive |
| GSM1032987 | LUAD | male | 4.50 | 5.36 | both-low | inconclusive |
| GSM1032990 | LUAD | female | 6.78 | 4.11 | both-low | inconclusive |
| GSM1033000 | LUAD | male | 4.73 | 5.67 | both-low | inconclusive |

---

*The card before the quality control of 29 October follows (kept for traceability; its sample counts and SHA-256
are those of the previous version).*

- Role: secondary-platform GPL6884 target cohort for LUAD versus LUSC; report separately from the GPL570 pooled
  k-star analysis.
- Source: GEO GSE42127 series matrix; platform GPL6884.
- Included samples: 176 (LUAD 133, LUSC 43); one sample per patient, with `patient_id = sample_id`.
- Expression: 19,405 gene-symbol columns. The source series matrix states MBCB background correction and quality
  control; 17,235 genes intersect TCGA. NKX2-1, NAPSA, TP63 and KRT5 are present.
- Label source: `Sample_characteristics_ch1_7`; the explicit mapping is in `label_mapping.json`.
- Input SHA-256: `expression.h5` `c86ae8d9d308b28b12c3731af8111c25d136e00383c2ddaac4f59755cb99f8f5`; `metadata.csv`
  `0149812c5474539d0f619f8836d034d3769b3243beac1471bb2c57e3ae540d91`.
- Audit 2026-10-27: metadata IDs match expression IDs and the 176 source GEO IDs; no shared sample ID with the other
  target cohorts.

The historical fold files are not the splits of the benchmark. The splits used in the paper are those of the update
of 29 October above.
