# Data card: GSE37745

This card is the English version of the data card kept by the authors while the cohort was built. The original,
in Vietnamese, is at the same path in commit `d7dcf9c` of this repository. The record files named here (`excluded_samples_*.csv`,
`qc/*.csv`) keep the wording of that time; `docs/RECORDS_GLOSSARY.md` translates it. The rules are named there by
the numbers of the authors' decision log: QĐ21 (near-duplicate pair of GSE37745), QĐ22 (two-axis sex check of the
GEO cohorts) and QĐ22b (per-gene Y vote of TCGA).

## Update of 2 November 2026: label cross-check, duplicate screen, coverage of the panel genes

### Cross-check of the labels against the original series matrix

- Method: `scripts/check_label_crosscheck.py` reads `GSE37745_series_matrix.txt.gz` (folder
  `geo_target_work/raw/GSE37745/`), takes the field `histology` of `!Sample_characteristics_ch1` by its key (not by
  its column position) and maps it by exact string match through `label_mapping.json`, without regular expressions:
  `histology: adeno` → 0; `histology: squamous` → 1.
- Result: passed. Series matrix 196 samples → LUAD/LUSC 106/66 (equal to `n_samples` in `label_mapping.json`) → minus
  the 3 samples of `excluded_samples_gse37745.csv` → 104/65, equal to `DATASET_EXPECTED_N` 104/65. The set of samples
  is identical to `metadata.csv` (no sample on one side only) and the labels agree for 100 % of the samples
  (0 mismatch).
- Label values outside the task (excluded because they do not match exactly): histology: large (24).
- Files: `qc/label_crosscheck_gse37745.csv` (one row per sample of the series matrix: original value, derived label,
  status `giu` = kept / `cach_ly_QC` = excluded by quality control / `khong_thuoc_bai_toan` = not part of the task,
  label in the metadata, column `match`); summary in `qc/label_crosscheck_summary.csv`. Run on 2 November 2026.

### Duplicate-sample screen (r_z, 29 October 2026)

- `scripts/geo_sample_qc.py` (configuration `qc/rz_screen_config.json`, commit `d7cb37a`, run on 29 October on the
  cohort before quality control): r_z is the Pearson correlation of two samples after z-scoring each gene within the
  cohort; threshold r_z ≥ 0.90, together with raw r ≥ 0.99.
- Within the cohort: 14,706 pairs on 20,848 genes; largest r_z 0.7175, 99.9th percentile 0.4875; 0 pairs with
  r_z ≥ 0.90; largest raw r 0.9925 (2 pairs with r ≥ 0.99).
- Between cohorts (17,062 genes shared by the four GEO cohorts): largest r_z 0.5074 (with GSE30219); 0 pairs with
  r_z ≥ 0.90 with any other cohort.
- The two pairs with r ≥ 0.99 are handled in the update of 29 October: GSM1019246/GSM1019247 excluded (QĐ21);
  GSM1019319/GSM1019325 kept (two different patients, same sex, same label).

### patient_id

- `metadata.csv` has a column `patient_id` (= `sample_id`), so `load_dataset` records
  `patient_id_assumed_from_sample_id = false`. Each sample is taken as one patient because the titles give a patient
  number ("Patient N, …") and the 169 numbers are distinct, the r_z screen found no duplicate pair, and the sex check
  excluded the samples whose identity could not be verified. Folds grouped by `patient_id` are therefore equivalent to
  folds by sample.

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

- n before → after: 172 (106/66) → 169 (104/65) LUAD/LUSC; `DATASET_EXPECTED_N` = (104, 65) (commit `75086ee`).
- Rebuild of the cohort files: `scripts/geo_sample_qc.py rebuild` at commit `75086ee`
  (2026-10-29T13:24:03+07:00). The rebuild only removes the samples of `excluded_samples_gse37745.csv` and adds the
  column `sex`; every other value is unchanged (checked by assertions in the script). Previous version:
  `_backup_20261029-132403/` (not in this repository).
- New SHA-256: `expression.h5` `f5dcaf11d7f36defc37ad67cf47da30b2f85823e3360fef2f8f4d0ab96a34ec7`; `metadata.csv`
  `568b557e43c09daee94f8959f0144db642064502d73cb6493e8f5ff77c031328`.
- Splits (`build_v3_splits.py --force`, commit `75086ee`): `repeated_splits.csv`
  `18dc9577a10b86f9dec236cefad6beac48fb43db4c9d1383fed547b80712fd20` (4225 rows); `fewshot_splits.csv`
  `ede2fd00f8dbff7ace0cac6b48761aafbab1601878e5f1ab2967b3889ae95a74` (n_T: 5, 10, 20, 40, pool; 5 splits × 4 subsets).
- GEO source: GSE37745, "Biomarker discovery in non-small cell lung cancer: integrating gene expression profiling,
  meta-analysis and tissue microarray validation"; submitted 03/05/2012, last updated 26/05/2022 (fields `!Series_*`
  of the series matrix); original article PMID 23032747 (the series also lists 29112949, 26608184, 33576873,
  35574381); series matrix downloaded to `geo_target_work/raw/GSE37745/` on 17/10/2026 02:23 (UTC+7) (file time on
  the authors' machine).
- Source of sex: the field `gender:` of `!Sample_characteristics_ch1` in the series matrix, read by its key (checked
  directly in the series matrix: no column shift).
- Rule (QĐ22, two axes): XIST and the mean of five Y genes (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY); each axis is split
  into two k-means clusters within the cohort. Opposite (both axes contradict the metadata) and inconsistent (XIST and
  the Y genes both high) → excluded; inconclusive (both low, only one axis contradicts, or an axis does not separate)
  → kept and flagged.
- Thresholds: XIST 8.51; Y axis 6.30 (log2).
- Result: match 169, inconclusive 1, opposite 1, inconsistent 1.

### Excluded samples (3)

Near-duplicate pair (QĐ21). GSM1019246/GSM1019247: r = 0.9925 (next pair ≤ 0.959); both are female by expression,
GSM1019247 is recorded as male, and the two labels disagree (LUSC/LUAD) → both excluded.

Sex check (QĐ22). Opposite: the sex contradicts the metadata, and the histology label comes from the same metadata
row, so it cannot be trusted. Inconsistent: the identity of the sample cannot be verified.

| Sample | Class | Sex (metadata) | XIST | Y axis (mean of 5 genes) | category | verdict |
|---|---|---|---|---|---|---|
| GSM1019197 | LUAD | male | 11.25 | 8.47 | both-high | inconsistent |

### Samples kept and flagged as inconclusive (1)

| Sample | Class | Sex (metadata) | XIST | Y axis (mean of 5 genes) | category | verdict |
|---|---|---|---|---|---|---|
| GSM1019177 | LUSC | male | 4.81 | 5.84 | both-low | inconclusive |

### Highly correlated pair kept

GSM1019319 / GSM1019325: r = 0.9905 (r_z = 0.7175) on 20,848 genes; consistent metadata (Patient 294 and Patient
329, both male, both LUSC; the sex genes agree with male) → both kept under the rule (two different patients, same
sex, same label).

---

*The card before the quality control of 29 October follows (kept for traceability; its sample counts and SHA-256
are those of the previous version).*

- Role: primary GPL570 target cohort for LUAD versus LUSC.
- Source: GEO GSE37745 series matrix; platform GPL570.
- Included samples: 172 (LUAD 106, LUSC 66); samples with histologies other than explicit adeno or squamous are
  excluded.
- Expression: 20,848 gene-symbol columns. The source series matrix states RMA preprocessing; 16,479 genes intersect
  TCGA. NKX2-1, NAPSA, TP63 and KRT5 are present.
- Label source: `Sample_characteristics_ch1_3`; the explicit mapping is in `label_mapping.json`.
- Input SHA-256: `expression.h5` `783d49f8599f82321f27cfb9081c2b2b71a064d7f481f5bab1fdb4b1b1de768b`; `metadata.csv`
  `af9ef378801234b02d9bb7b7003ad8fcaade99d8fe58be57f5a9d86d2ab3e742`.
- Audit 2026-10-27: metadata IDs match expression IDs and are all present in the 196-sample source GEO matrix; no
  shared sample ID with the other target cohorts.
- Duplicate screen: GSM1019246 and GSM1019247 have expression correlation 0.992273 on genes shared across the four
  targets. The original series matrix identifies them as Patient 30 female squamous and Patient 36 male adeno,
  respectively; both are retained. (Superseded on 29 October: both are excluded, see above.)

The historical fold files are not the splits of the benchmark. The splits used in the paper are those of the update
of 29 October above.
