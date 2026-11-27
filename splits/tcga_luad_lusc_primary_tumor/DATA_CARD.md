# Data card: `tcga_luad_lusc_primary_tumor`

This card is the English version of the data card kept by the authors while the cohort was built. The original,
in Vietnamese, is at the same path in commit `d7dcf9c` of this repository. The record files named here
(`excluded_samples_*.csv`, `qc/*.csv`) keep the wording of that time; `docs/RECORDS_GLOSSARY.md` translates it. The
rules are named there by the numbers of the authors' decision log: QĐ21 (near-duplicate pair of GSE37745), QĐ22
(two-axis sex check of the GEO cohorts) and QĐ22b (per-gene Y vote of TCGA).

## Update of 2 November 2026: coverage of the panel genes

- The label cross-check of the GEO cohorts does not apply: the TCGA labels follow the original project (LUAD → 0,
  LUSC → 1; section Labels).
- TCGA is the reference of the gene space (`GENE_SPACE_REFERENCE`): E1 and E1b use all 19,938 genes.

| Panel | Present / total (genes of the platform) | Present / total (gene space of E1/E1b) | Missing genes |
|---|---|---|---|
| IHC-2 | 2/2 | 2/2 | — |
| IHC-4 | 4/4 | 4/4 | — |
| Girard 2016 (42 genes) | 42/42 | 42/42 | — |
| Wilkerson 2013 (52 genes) | 52/52 | 52/52 | — |

Note on gene symbols (3 November 2026): Wilkerson et al. (2013, Table 1) list TUBA1. The benchmark uses TUBA4A,
following HGNC (HGNC:12407, previous symbol TUBA1; looked up on 3 November 2026), without a check against the PCR
primers of the original article.

---

## Update of 29 October 2026: sample quality control (sample counts used in the paper)

- n before → after: 1018 (517/501) → 1016 (515/501) LUAD/LUSC; `DATASET_EXPECTED_N` = (515, 501) (commit `75086ee`).
- Rebuild of the cohort files: `scripts/geo_sample_qc.py rebuild` at commit `75086ee`
  (2026-10-29T13:24:04+07:00). The rebuild only removes the samples of
  `excluded_samples_tcga_luad_lusc_primary_tumor.csv` and adds the column `sex`; every other value is unchanged
  (checked by assertions in the script). Previous version: `_backup_20261029-132404/` (not in this repository).
- New SHA-256: `expression.h5` `41bf54bda90aae286eab2ceec7e705aeaa778cd77eb85587ab7ff8aaa75c2cff`; `metadata.csv`
  `ee65ea8b4075967867ce4f7fa8383122449d6ba2e10dd5e178a33b27b5ea7b26`.
- Splits (`build_v3_splits.py --force`, commit `75086ee`): `repeated_splits.csv`
  `ff909fd90415014e8dce4c1692346baa4873bee0e533932515880a5ed18cb6ce` (25400 rows).
- Source of sex: GDC API `POST /cases`, field `demographic.sex_at_birth` (the field `gender` is empty in every
  record), queried at 2026-10-29T11:52:39+07:00 (commit `21c6138`); 1018/1018 patients, 610 male / 408 female; file
  `sex_gdc.csv` (SHA-256 `72d16061e79f43081a3bc5210b075431ac6ad07fc354e0b583b141588ef1518c`).
- Rule (QĐ22b, one axis; the TCGA build holds protein-coding genes only and has no XIST): vote of each Y gene; the
  threshold of each gene is the midpoint between the 95th percentile of the samples recorded as female and the 5th
  percentile of the samples recorded as male; M-like(Y) when at least 4 of the 5 genes are above their threshold,
  F-like(Y) when at most 1 of 5 is; otherwise intermediate(Y) → inconclusive. Opposite → excluded; inconclusive → kept
  and flagged.
- Thresholds of the five genes (log2 TPM + 1): RPS4Y1 2.72; DDX3Y 1.10; KDM5D 0.66; UTY 0.31; EIF1AY 0.62.
- Result: match 1011, inconclusive 5, opposite 2.
- Note: the earlier version of the rule (mean of the five Y genes as one axis) gave 5 false opposite calls in male
  LUSC patients whose tumours have lower expression of the Y genes, so the rule was changed to a vote of each gene
  (29 October).

### Excluded samples (2)

Sex check (QĐ22b). Opposite: the sex contradicts the metadata, and the histology label comes from the same metadata
row, so it cannot be trusted. Inconsistent: the identity of the sample cannot be verified.

| Sample | Class | Sex (metadata) | RPS4Y1 | DDX3Y | KDM5D | UTY | EIF1AY | Male votes | category | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| TCGA-49-4505-01A | LUAD | female | 6.67 | 3.28 | 1.57 | 0.97 | 2.52 | 5/5 | M-like(Y) | opposite |
| TCGA-49-AARN-01A | LUAD | female | 6.45 | 2.14 | 1.73 | 1.01 | 1.81 | 5/5 | M-like(Y) | opposite |

### Samples kept and flagged as inconclusive (5)

| Sample | Class | Sex (metadata) | RPS4Y1 | DDX3Y | KDM5D | UTY | EIF1AY | Male votes | category | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| TCGA-21-5782-01A | LUSC | female | 1.19 | 0.63 | 0.71 | 0.91 | 0.85 | 3/5 | intermediate(Y) | inconclusive |
| TCGA-33-4582-01A | LUSC | male | 3.99 | 1.20 | 0.30 | 0.17 | 0.67 | 3/5 | intermediate(Y) | inconclusive |
| TCGA-85-A4QR-01A | LUSC | male | 3.30 | 0.86 | 0.71 | 0.20 | 0.32 | 2/5 | intermediate(Y) | inconclusive |
| TCGA-MF-A522-01A | LUSC | male | 3.28 | 1.49 | 0.59 | 0.26 | 0.75 | 3/5 | intermediate(Y) | inconclusive |
| TCGA-NC-A5HG-01A | LUSC | male | 3.23 | 1.07 | 0.74 | 0.20 | 0.61 | 2/5 | intermediate(Y) | inconclusive |

---

*The card before the quality control of 29 October follows (kept for traceability; its sample counts and SHA-256
are those of the previous version).*

Task: histology classification LUAD versus LUSC on primary tumour samples.

- Source: GDC API, TCGA-LUAD and TCGA-LUSC, STAR - Counts (GDC Data Release 46.0, August 10, 2026).
- Samples / patients / genes: 1018 / 1018 / 19938.
- Created (UTC): 2026-10-11T13:05:58+00:00.
- Commit: `no-git-repo` (entered by hand with `--commit`, `scripts/patch_v2_tcga_stage.py`); notebook
  `notebooks/00_data/lung_datasets_v2_patient_folds.ipynb` (now `notebooks/phase_0/00_data/`).
- Base seed: 42; selected split seed: 139 (searched in 42..141).

## Sample filters

1. Start from every GDC STAR - Counts file of TCGA-LUAD and TCGA-LUSC.
2. Keep only the files whose expression matrix was downloaded.
3. Keep only Primary Tumor samples (`sample_type = Primary Tumor`, barcode code `01`). Remove Solid Tissue Normal,
   Recurrent Tumor, Metastatic and Additional - New Primary.
4. Remove patients with samples in both projects (if any).
5. Keep one tumour sample per patient: prefer a sample that is not FFPE, then the smallest `sample_id` (for example
   `01A` before `01B`).
6. The label follows the project: TCGA-LUAD → LUAD, TCGA-LUSC → LUSC. No stage label is required.

## Sample counts before and after each filter

The column `buoc` (step) and its values are the identifiers written by the notebook.

| step | buoc | criterion | n_samples | n_patients | n_removed |
|---|---|---|---|---|---|
| 0 | 0_gdc_star_counts | Every STAR - Counts file of TCGA-LUAD and TCGA-LUSC | 1163 | 1019 | 0 |
| 1 | 1_co_bieu_hien | Only files whose expression matrix was downloaded | 1163 | 1019 | 0 |
| 2 | 2_chi_primary_tumor | Only Primary Tumor samples (sample_type = 'Primary Tumor' and barcode code 01); remove normal tissue, Recurrent, Metastatic, Additional - New Primary | 1051 | 1018 | 112 |
| 3 | 3_loai_benh_nhan_2_project | Remove patients with samples in both TCGA-LUAD and TCGA-LUSC | 1051 | 1018 | 0 |
| 4 | 4_mot_mau_moi_benh_nhan | Keep one tumour sample per patient (prefer non-FFPE, then the smallest sample_id) | 1018 | 1018 | 33 |

## Labels

| label | class | n_samples | n_patients |
|---|---|---|---|
| 0 | LUAD | 517 | 517 |
| 1 | LUSC | 501 | 501 |

## Expression matrix

- Values are `log2(TPM + 1)` of the column `tpm_unstranded` of the GDC STAR - Counts files.
- Only `protein_coding` genes; Ensembl IDs are mapped to `gene_name`; for duplicated names, the gene with the largest
  variance is kept (computed on the samples of this cohort, without the labels).
- No normalisation or scaling; no batch correction.

## Patient-level folds (historical folds, not the splits of the benchmark)

- `StratifiedGroupKFold` with `groups = patient_id`, stratified by class (LUAD/LUSC).
- Each patient lies in one part only; within each fold the patients of train, validation and test are pairwise
  disjoint (checked by `assert`).

| fold | train | val | test |
|---|---|---|---|
| fold_1 | P3+P4+P5 | P2 | P1 |
| fold_2 | P1+P4+P5 | P3 | P2 |
| fold_3 | P1+P2+P5 | P4 | P3 |
| fold_4 | P1+P2+P3 | P5 | P4 |
| fold_5 | P2+P3+P4 | P1 | P5 |

Samples (patients) per class:

| fold | split | LUAD | LUSC | TOTAL |
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

## Stage labels (columns of `metadata.csv` that this study does not use)

The metadata were patched by `scripts/patch_v2_tcga_stage.py` at commit `no-git-repo`; the expression matrix and the
folds did not change (the SHA-256 of `expression.h5`, `fold_assignment.csv`, `folds/` and `experiment-data-v2/h5/`
were identical before and after the script).

The error that was patched. The original build read only `diagnoses[0]` of each case. The GDC returns several
`diagnoses` records for one case (index tumour, *Prior primary*, *Synchronous primary*, *metastasis*, ...) in no fixed
order, so 121 LUAD and 85 LUSC cases lost their stage, and `primary_diagnosis` was sometimes taken from another
tumour. The old columns are kept as `stage_raw_v2_original` and `primary_diagnosis_v2_original` (not used for
analysis).

- Source: GDC API `POST https://api.gdc.cancer.gov/cases`, filtered on `submitter_id` in the 1018 `patient_id` of this
  cohort; every `diagnoses` record with the fields `ajcc_pathologic_stage`, `ajcc_staging_system_edition`,
  `ajcc_pathologic_t/n/m`, `primary_diagnosis`, `diagnosis_is_primary_disease`, `classification_of_tumor` (field names
  checked against the GDC Data Dictionary, node `diagnosis`, and `GET /cases/_mapping`).
- Query time (UTC): 2026-10-14T03:51:51+00:00; GDC Data Release 46.0, August 10, 2026.
- Raw JSON: `gdc_diagnoses_raw.json` (in the cohort folder; not in this repository).
- Rule for the stage of each patient:
  1. Candidates are the records whose `ajcc_pathologic_stage` is neither empty nor `Not Reported`.
  2. No candidate → no stage.
  3. All candidates have the same value → that value (`T/N/M` and `edition` from the index-tumour record if there is
     one).
  4. Candidates with different values → `stage_conflict = True`; take the index-tumour record
     (`diagnosis_is_primary_disease = True`, `classification_of_tumor = primary`) if there is exactly one. The other
     records are other tumours of the same patient (*Prior/Synchronous primary*).
- `stage_group`: `Stage IA/IB/I` → I, `IIA/IIB/II` → II, `IIIA/IIIB/IIIC/III` → III, `IV/IVA/IVB` → IV.
- New columns of `metadata.csv`: `stage_raw`, `stage_group`, `ajcc_edition`, `path_t`, `path_n`, `path_m`,
  `stage_conflict`, `stage_rule` (`single` / `conflict_index_tumor` / `none`), `stage_diagnosis_submitter_id`,
  `primary_diagnosis` (index-tumour record), `diagnosis_mismatch`.

Tumours by stage:

| project | I | II | III | IV | missing | total |
|---|---|---|---|---|---|---|
| TCGA-LUAD | 276 | 122 | 85 | 26 | 8 | 517 |
| TCGA-LUSC | 244 | 162 | 84 | 7 | 4 | 501 |
| total | 520 | 284 | 169 | 33 | 12 | 1018 |

By AJCC edition (`ajcc_edition`):

| project | stage_group | 3rd | 4th | 5th | 6th | 7th | not recorded | total |
|---|---|---|---|---|---|---|---|---|
| TCGA-LUAD | I | 3 | 1 | 14 | 96 | 154 | 8 | 276 |
| TCGA-LUAD | II | 0 | 2 | 9 | 34 | 74 | 3 | 122 |
| TCGA-LUAD | III | 0 | 2 | 6 | 43 | 27 | 7 | 85 |
| TCGA-LUAD | IV | 0 | 0 | 1 | 13 | 10 | 2 | 26 |
| TCGA-LUAD | missing | 0 | 0 | 0 | 0 | 0 | 8 | 8 |
| TCGA-LUSC | I | 0 | 1 | 16 | 70 | 102 | 55 | 244 |
| TCGA-LUSC | II | 2 | 0 | 5 | 45 | 86 | 24 | 162 |
| TCGA-LUSC | III | 0 | 0 | 2 | 39 | 28 | 15 | 84 |
| TCGA-LUSC | IV | 0 | 0 | 1 | 1 | 3 | 2 | 7 |
| TCGA-LUSC | missing | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| total |  | 5 | 6 | 54 | 341 | 484 | 128 | 1018 |

- Missing stage: 12 cases (TCGA-LUAD 8, TCGA-LUSC 4): `TCGA-22-5473`, `TCGA-38-4626`, `TCGA-46-3769`, `TCGA-50-5045`,
  `TCGA-52-7812`, `TCGA-55-5899`, `TCGA-64-1678`, `TCGA-67-4679`, `TCGA-69-7765`, `TCGA-69-8254`, `TCGA-73-4677`,
  `TCGA-92-8064`. No `diagnoses` record of these cases has a stage.
- `stage_conflict = True`: 25 cases (TCGA-LUAD 13, TCGA-LUSC 12); the stage of the index tumour was taken for all 25.
  A stage analysis should be repeated without these cases (sensitivity analysis).
- Comparison with the earlier build v1 (`experiment-data/h5/tcga_{luad,lusc}/fold_assignment.csv`): 1006 patients
  have a stage in both builds; the 981 cases without conflict have the same `stage_group` (100 %). Of the 25 conflict
  cases, 7 differ from v1 because v1 took the first record and the patch takes the index tumour: `TCGA-18-3407` (I vs
  v1 III), `TCGA-38-7271` (I vs v1 II), `TCGA-43-8118` (I vs v1 II), `TCGA-50-5932` (II vs v1 I), `TCGA-55-A492` (I vs
  v1 II), `TCGA-NC-A5HE` (II vs v1 III), `TCGA-NC-A5HR` (II vs v1 I). With these 7 cases taken as in v1, the counts by
  stage equal those of v1 (TCGA-LUAD 509 (275/123/85/26), TCGA-LUSC 497 (243/161/86/7)).

`primary_diagnosis` (index-tumour record) by project:

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

- Project mismatch (`diagnosis_mismatch = True`: LUAD with a squamous diagnosis, or LUSC with a non-squamous
  diagnosis): 1 case, `TCGA-43-8118` (TCGA-LUSC, Adenocarcinoma, NOS). For a sensitivity analysis (remove these cases
  and evaluate LUAD versus LUSC again).
- With the old column `primary_diagnosis_v2_original` (`diagnoses[0]`) there would be 56 "mismatches" and 84
  `Not Reported` cases; most of them are diagnoses of another tumour (for example "Squamous cell carcinoma, NOS" in
  TCGA-LUAD), not label errors.

## Files of the cohort folder

Only `metadata.csv`, `class_stats.csv`, `label_mapping.json`, `repeated_splits.csv` and the exclusion list are in this
repository.

- `expression.h5` (key `expression`): samples × genes, float32, index `sample_id`.
- `metadata.csv`: filtered metadata of the samples that are kept.
- `fold_assignment.csv`: `part` and the train/val/test role of each sample in `fold_1..fold_5`.
- `class_stats.csv`: numbers of patients and samples per class in each part and set.
- `filter_log.csv`: numbers of samples before and after each filter.
- `config.json`: seed, commit, filters, label mapping, library versions.
- `folds/fold_<r>/{train,val,test}.h5`: tables samples × (genes..., `label`) built from `fold_assignment.csv`.

## Notes and limitations

- The labels follow the TCGA project; `primary_diagnosis` is not used to revise them (it is kept in `metadata.csv`
  for reference).
- Each patient has one sample, so a split by patient is equivalent to a split by sample; `groups` is still used for
  consistency.
- Tissue source site (TSS) and batch effects may be present; they are not corrected.
