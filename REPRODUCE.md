# Reproducing "Efficient, Stable and Transferable Gene Selection under a Panel Budget"

This document describes the code and result files of the survey *Efficient, Stable and Transferable Gene Selection
under a Panel Budget: A Survey and Empirical Evaluation in Lung Cancer Histology Diagnosis* (submitted to Data Mining
and Knowledge Discovery). It applies to the tag `survey-v3-dmkd.1` of this repository.

The tag holds the benchmark code of tag `survey-v3-submission` (commit 06ba993) unchanged, the result files that the
paper reports, and this documentation. The other README files of the repository are internal working notes written in
Vietnamese; this file is the reference for readers of the paper.

## 1. Contents of the tag

| Path | Content |
|---|---|
| `survey_v2/` | Python package of the benchmark: data access, the thirteen selectors and five reference panels, resampling, classifier, metrics, stability, provenance |
| `scripts/` | command-line runners (`run_survey_e1.py`, `run_survey_e1_wrappers.py`, `run_survey_e2_e3_timing.py`, `run_survey_e4_e5_stability.py`, ...), split builders (`build_v3_splits.py`), quality control (`geo_sample_qc.py`), table builder of the result files (`make_survey_tables.py`) |
| `notebooks/kaggle/` | the Kaggle notebooks that ran every experiment (`kaggle_E0_signal_check.ipynb` ... `kaggle_E7_transfer.ipynb`, `kaggle_panels.ipynb`, `kaggle_ablation_no_IHC.ipynb`, `kaggle_redo_svm_rfe.ipynb`, `kaggle_00_sanity.ipynb`) |
| `tests/` | unit tests (`python -m pytest -q`) |
| `results/survey_v2/` | outputs of every experiment, one folder per experiment and cohort (Section 2) |
| `results/run_info_*.json`, `results/logs/` | session information and logs of each Kaggle job, including `pip freeze` |
| `splits/` | patient-level split files and quality-control files (Section 3) |
| `paper_scripts/` | `make_tables.py` and `make_figures.py`: they write every table and the result figures (Figs. 5-12) of the paper from `results/` (Section 4) |
| `paper_figure_sources/` | sources of the diagrams (Figs. 1-4, 13) and of the graphical abstract: draw.io files and TikZ |
| `requirements-kaggle.txt` | package versions of the environment that produced every reported number (Python 3.13.15) |

## 2. Result files (`results/survey_v2/`)

| Folder | Experiment of the paper |
|---|---|
| `E0/` | E0, signal check of the histology label on each cohort (permutation and random-k scores) |
| `sanity/` | positive control of the pipeline (tumour vs. normal tissue) |
| `E1_feature_selection/`, `E1_classification/` | E1, selection and classification under a hard budget k on the five cohorts, 5 x 5 patient-level cross-validation; `selection_per_fold.csv`, `selected_genes.csv`, `classification_per_fold.csv`, `classification_summary.csv`, random-k baseline |
| `E1_wrapper_feature_selection/` | E1 for the two wrappers (GA, BPSO) on GSE10245, GSE30219 and GSE37745 |
| `E1_feature_selection_noIHC/`, `E1_classification_noIHC/` | ablation without the four immunohistochemistry marker genes |
| `E1b/` | E1b, few-shot regime on the four GEO cohorts (`e1b_per_run.csv`, `e1b_selected_genes.csv`, `e1b_stability.csv`) |
| `E2_E3/`, `E2_E3_ref/`, `E2_E3_p1/` ... `E2_E3_p3/` | E2 and E3, measured cost and scaling (calibration and machine files included) |
| `E4_E5/` | E4 and E5, stability on TCGA and GSE37745 (Nogueira index with confidence intervals, subsamples), redundancy, seed variability, overlap |
| `E7/` | E7, naive transfer strategies from TCGA to the four GEO cohorts (`e7_per_run.csv`, `e7_selected_genes.csv`) |
| `E7_k2_evidence/` | gene-level analysis of the zero-shot rule at k = 2 (Section 7.4 of the paper) |
| `tables/` | summary tables written by `scripts/make_survey_tables.py` (CSV and LaTeX) and `provenance_manifest.json` |
| `figures/` | figures written by `scripts/make_survey_tables.py` |

The files store, for every fold and run, the five metrics of the paper (AUROC, balanced accuracy, MCC, log-loss, Brier
score), the decision threshold, the selected genes and the selection time. Per-sample predictions are not stored.

Every folder holds a `config.json` with the commit of the code that produced it, the SHA-256 of the input data and of
the split file, the package versions and the machine. The result files were produced at commits 9b2d6a5, 6ea47e8,
d870008, e15fb7c and 06ba993 (Appendix B of the paper); the code at these commits is in the history of this tag, and
no code file changed after 06ba993.

## 3. Splits and quality control (`splits/`)

| Path | Content |
|---|---|
| `<cohort>/repeated_splits.csv` | 5 repeats x 5 patient-grouped outer folds (seeds 42-46) of the five cohorts |
| `<cohort>/fewshot_splits.csv` | few-shot design of the four GEO cohorts: 5 test/pool splits x 4 nested subsets per target size n_T |
| `<cohort>/label_mapping.json`, `metadata.csv`, `class_stats.csv`, `excluded_samples_<cohort>.csv` | labels (LUAD = 0, LUSC = 1), sample metadata, class counts, samples removed by quality control |
| `<cohort>/DATA_CARD.md` | data card of the cohort (in Vietnamese): source files, SHA-256, quality-control decisions |
| `split_manifests/` | generator, commit, seeds and SHA-256 of every split file |
| `qc/` | label cross-checks of the GEO cohorts, duplicate-sample screen, sex consistency checks, coverage of the panel genes |

The SHA-256 of each split file in `split_manifests/` is the value recorded in the `config.json` of the runs that used it.

## 4. Regenerate the tables and result figures (no experiment is rerun)

Tested on 11/11/2026 with Python 3.13 on Linux (numpy, pandas and matplotlib only):

```
python paper_scripts/make_tables.py --results results/survey_v2/tables --e0-configs results/survey_v2/E0 --qc splits/qc --out out/Tables
python paper_scripts/make_figures.py --results results/survey_v2/tables --e3 results/survey_v2/E2_E3 --out out/Images
cd out/Tables && sha256sum -c ../../paper_scripts/EXPECTED_SHA256.txt
```

`make_tables.py` writes the 22 table files of the paper and `numbers_log.json` (2,194 numbers, each with its source
file and column); the checksums in `paper_scripts/EXPECTED_SHA256.txt` are those of the table files of the paper.
On Windows, line endings may differ; compare with `diff --strip-trailing-cr` instead of the checksums.
`make_figures.py` writes Figures 5-12. In the journal version, six tables use a wider float environment; their content
is unchanged.

## 5. Rerun the experiments

1. Environment: `pip install -r requirements-kaggle.txt` (the environment of every reported number). `abess` (selector
   abess) and `skrebate` (selector ReliefF) are part of it; without `abess`, two tests are skipped.
2. Data: the expression data are public and are not redistributed here. TCGA-LUAD and TCGA-LUSC come from the Genomic
   Data Commons (https://portal.gdc.cancer.gov/), GSE10245, GSE30219, GSE37745 and GSE42127 from the Gene Expression
   Omnibus (https://www.ncbi.nlm.nih.gov/geo/). The data cards in `splits/<cohort>/DATA_CARD.md` give the SHA-256 of the
   cohort files used for the paper.

   *Folder layout.* Every runner reads the cohorts below the folder named by the environment variable
   `KLTN_DATA_ROOT` (`survey_v2/config.py`, `DATASET_DIRS`):

   ```
   $KLTN_DATA_ROOT/
     experiment-data-perm/tcga_luad_lusc_primary_tumor/   expression.h5, metadata.csv, sex_gdc.csv
     experiment-data-geo-target/gse10245/                 expression.h5, metadata.csv
     experiment-data-geo-target/gse30219/                 (the R5 build; gse30219_r5_fixed/ is also accepted)
     experiment-data-geo-target/gse37745/
     experiment-data-geo-target/gse42127/
     splits/<cohort>/                                     written by build_v3_splits.py (step 3)
   ```

   `expression.h5` holds a samples x genes table under the key `expression` (index `sample_id`); `metadata.csv` has
   one row per sample with `sample_id`, `patient_id`, `class` and `label` (LUAD = 0, LUSC = 1).

   *Sequence used for the paper.* The cohort files were built in this order (dates and commits are in the data cards):

   1. TCGA: `notebooks/phase_0/00_data/lung_datasets_v2_patient_folds.ipynb` (11/10/2026) downloads the STAR-Counts
      files of TCGA-LUAD and TCGA-LUSC through the GDC API (GDC Data Release 46.0), keeps one primary tumour per
      patient and only protein-coding genes, and writes `log2(TPM + 1)`. Its output folder
      `tcga_luad_lusc_primary_tumor` was placed under `experiment-data-perm/`. `scripts/patch_v2_tcga_stage.py`
      (14/10/2026) then rewrote the stage and diagnosis columns of `metadata.csv` only; the expression matrix was
      unchanged. This script expects the earlier layout `experiment-data-v2/` and `experiment-data/h5/` and does not
      run against the layout above without editing its paths; the stage columns are not used by the survey.
      `notebooks/phase_0/00_data/lung_download_label_5fold.ipynb` is the earlier version of this notebook and was
      not used for the paper's cohort file.
   2. GEO: `notebooks/phase_0/00_data/geo_luad_lusc_target_cohorts_5fold.ipynb` (16-17/10/2026) downloads the series
      matrices into `$KLTN_GEO_WORKDIR/raw/` and writes one folder per series into `$KLTN_GEO_OUTDIR` (use
      `$KLTN_DATA_ROOT/experiment-data-geo-target`). Only samples with an explicit LUAD or LUSC label are kept. The
      notebook also builds GSE43580, which is not used.
   3. GSE30219: `python scripts/rebuild_gse30219_r5.py --source $KLTN_DATA_ROOT/experiment-data-geo-target/gse30219
      --output $KLTN_DATA_ROOT/experiment-data-geo-target/gse30219_r5_fixed` keeps only the exact histology values
      `ADC` and `SQC` (`SCC` is small-cell carcinoma in this series). The output (146 samples) replaced the
      notebook's `gse30219/`.
   4. Sample quality control (29/10/2026, commits 45b11a6, 5facab3 and 380d01f), in this order:

      ```
      python scripts/geo_sample_qc.py fetch-tcga-sex
      python scripts/geo_sample_qc.py rebuild --sex-only --cohorts gse10245 gse30219 gse37745 gse42127 tcga_luad_lusc_primary_tumor
      python scripts/check_sex_consistency.py
      python scripts/geo_sample_qc.py apply-sex-qc
      python scripts/geo_sample_qc.py rebuild --cohorts gse10245 gse30219 gse37745 gse42127 tcga_luad_lusc_primary_tumor
      ```

      `fetch-tcga-sex` queries the GDC API for the sex of each TCGA patient (`sex_gdc.csv`). The final `rebuild` removes
      the samples listed in `<cohort>/excluded_samples_<cohort>.csv` and adds the column `sex`; no other value changes.
      The exclusion lists used for the paper are in `splits/<cohort>/excluded_samples_<cohort>.csv`; one pair of
      GSE37745 (GSM1019246, GSM1019247) was added by a decision recorded in its data card, not by `apply-sex-qc`.
      Copy these lists into the cohort folders to apply the same exclusions. `rebuild` refuses to run from code that is
      not committed and pushed (set `KLTN_ALLOW_UNPUSHED=1` for a local check).

   *Check.* Compare the SHA-256 of the files you built with the values in the section of 29/10/2026 of each
   `splits/<cohort>/DATA_CARD.md` ("SHA-256 mới"), for example `sha256sum expression.h5 metadata.csv` in each cohort
   folder; the same values are recorded in the `config.json` of every result folder. A rebuild may not match byte
   for byte: GDC and GEO may have revised the source files since October 2026, and the bytes of `expression.h5`
   depend on the versions of pandas and PyTables. In that case, compare sample identifiers, labels and values instead.
3. Splits: `python scripts/build_v3_splits.py` rebuilds the split files; their SHA-256 must equal `split_manifests/`.
4. Experiments: run the notebooks of `notebooks/kaggle/` (or the runners of `scripts/`) at the commit recorded in the
   `config.json` of each result folder. Every run refuses to start from a working tree with uncommitted changes and
   writes its own `config.json`.
5. Tables: `python scripts/make_survey_tables.py` (options in `--help`) writes the summary tables of
   `results/survey_v2/tables/`; then Section 4.

## 6. Licence and citation

The source code is licensed under the MIT licence (`LICENSE`). The result files, splits and figure sources are licensed
under CC BY 4.0 (`LICENSE-DATA.md`), with the third-party notices of `paper_figure_sources/THIRD_PARTY_NOTICES.md`.
Please cite the paper when you reuse the benchmark; the citation will be added here after publication.
