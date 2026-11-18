# Reproducing "Efficient, Stable and Transferable Gene Selection under a Panel Budget"

This document describes the code and result files of the survey *Efficient, Stable and Transferable Gene Selection
under a Panel Budget: A Survey and Empirical Evaluation in Lung Cancer Histology Diagnosis* (submitted to Data Mining
and Knowledge Discovery). It applies to the tag `survey-v3-dmkd.3` of this repository (branch `dmkd-revision`).

The tag holds the code and result files of the main benchmark (E0-E7; also tagged `survey-v3-dmkd.1`, unchanged), the
code and result files of the sensitivity analyses and the second task (SA1-SA4, E8; Section 6), and this documentation. The other README files
of the repository are internal working notes written in Vietnamese; this file is the reference for readers of the paper.

## 1. Contents of the tag

| Path | Content |
|---|---|
| `survey_v2/` | Python package of the benchmark: data access, the thirteen selectors and five reference panels, resampling, classifier, metrics, stability, provenance |
| `scripts/` | command-line runners (`run_survey_e1.py`, `run_survey_e1_wrappers.py`, `run_survey_e2_e3_timing.py`, `run_survey_e4_e5_stability.py`, ...), split builders (`build_v3_splits.py`), quality control (`geo_sample_qc.py`), table builder of the result files (`make_survey_tables.py`) |
| `notebooks/kaggle/` | the Kaggle notebooks that ran every experiment (`kaggle_E0_signal_check.ipynb` ... `kaggle_E7_transfer.ipynb`, `kaggle_panels.ipynb`, `kaggle_ablation_no_IHC.ipynb`, `kaggle_redo_svm_rfe.ipynb`, `kaggle_00_sanity.ipynb`; Section 6: `kaggle_B1_fewshot20.ipynb`, `kaggle_B2_B5.ipynb`, `kaggle_B7_step1.ipynb`, `kaggle_B7_E1_E4.ipynb`) |
| `notebooks/kaggle_r/kaggle_B6_e7_r.ipynb` | Section 6: ptLasso and Trans-GLM in E7 (Python kernel that installs R 4.5.3, ptLasso 1.0.2, glmtrans 2.1.0) |
| `tests/` | unit tests (`python -m pytest -q`) |
| `results/survey_v2/` | outputs of every experiment, one folder per experiment and cohort (Section 2) |
| `results/run_info_*.json`, `results/logs/` | session information and logs of each Kaggle job, including `pip freeze` |
| `splits/` | patient-level split files and quality-control files (Section 3) |
| `paper_scripts/` | `make_tables.py`: the tables of the main benchmark from `results/` (Section 4); `make_figures_dmkd.py` with `check_figure_lettering.py`: the result figures of the paper (Figs. 5-12, Section 6.4); `make_supplementary_tables.py`: the tables of Section 6 and of Online Resource 1 (Section 6.3); `check_figure4.py`: check of Fig. 4; `make_figures.py`, `make_figures_dmkd2.py`: earlier layouts of the same figures, kept for the record |
| `paper_figure_sources/` | sources of the diagrams (Figs. 1-4, 13): draw.io files and TikZ (`Figure_4_complexity.tex`, built with `pdflatex`) |
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
| `E7_k2_evidence/` | gene-level analysis of the zero-shot rule at k = 2 (Table 20 of the paper) |
| `tables/` | summary tables written by `scripts/make_survey_tables.py` (CSV and LaTeX) and `provenance_manifest.json` |
| `figures/` | figures written by `scripts/make_survey_tables.py` |
| `B1_…` … `B7_…`, `E0_b7/`, `E1_feature_selection_b7/`, `E1_classification_b7/`, `E4_E5_b7/` | sensitivity analyses SA1-SA4 and second task E8 (Section 6) |
| `B6_reference_rq2_r_7f097da/` | reference files of the B6 equality gate (copied from the method paper's run, Section 6.1) |
| `DMKD2_EXCLUDED_DATA.tsv` | the two expression matrices built for B7 that are not redistributed (path, size, SHA-256) |

The files store, for every fold and run, the five metrics of the paper (AUROC, balanced accuracy, MCC, log-loss, Brier
score), the decision threshold, the selected genes and the selection time. Per-sample predictions are not stored.

Every folder holds a `config.json` with the commit of the code that produced it, the SHA-256 of the input data and of
the split file, the package versions and the machine. The result files of the main benchmark were produced at
commits 9b2d6a5, 6ea47e8, d870008, e15fb7c and 06ba993 (Appendix A of the paper); those of Section 6 at commits
c1828d2, c50647a and f22597c, and `B1_tables/`, `B6_tables/` were regenerated at 3fe6d99 (Section 6.1). The code at all these commits is in the history of this tag.

## 3. Splits and quality control (`splits/`)

| Path | Content |
|---|---|
| `<cohort>/repeated_splits.csv` | 5 repeats x 5 patient-grouped outer folds (seeds 42-46) of the five cohorts |
| `<cohort>/fewshot_splits.csv` | few-shot design of the four GEO cohorts: 5 test/pool splits x 4 nested subsets per target size n_T |
| `<cohort>/fewshot_splits_20.csv` | Section 6 (B1): the same design with 20 splits (seeds 42-61); splits 0-4 are identical to `fewshot_splits.csv` |
| `tcga_luad_egfr/`, `gse31210_egfr/` | Section 6 (E8, folder prefix B7): metadata with EGFR labels, 5 x 5 splits, exclusions and data cards of the two EGFR data sets |
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
(cd out/Tables && sha256sum -c --ignore-missing ../../paper_scripts/EXPECTED_SHA256.txt)
```

`make_tables.py` writes the 22 table files of the paper and `numbers_log.json` (2,194 numbers, each with its source
file and column); the checksums in `paper_scripts/EXPECTED_SHA256.txt` are those of the table files of the paper.
On Windows, line endings may differ; compare with `diff --strip-trailing-cr` instead of the checksums.
`make_figures.py` writes an earlier layout of Figures 5-12; the figures of the paper are written by `make_figures_dmkd.py`
(Section 6.4) from the same tables. In the journal version, six tables use a wider float environment; their content
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

## 6. Sensitivity analyses and the second task (SA1-SA4, E8)

The specification and the reading rules of these experiments were fixed before they were run. Every run used the
protocol of the main benchmark (same splits and seeds, label-free filter to 5,000 genes inside the training part,
LR-L2 with C in {0.01, 0.1, 1, 10} chosen by inner cross-validation, balanced class weights, the same five metrics),
except where a row below says otherwise. The new data sets were appended at the end of `DATASETS` in
`survey_v2/config.py`, so the seed identifiers of the five cohorts of the main benchmark are unchanged (tested).
Settings that the specification did not fix are listed under `choices_to_confirm` in the `config.json` of each table
folder.

The folders keep their working names; the paper uses the names of the right-hand column: B1 (20 few-shot splits of E1b
and E7), B6 (ptLasso and Trans-GLM in E7), B4 = SA1 (reference model B0, Table 15), B5 = SA2 (signatures cut to ten
genes, Tables S17-S18), B2 = SA3 (k-TSP voting rule, Tables S19-S20), B3 = SA4 (zero-shot variants, Table 20 and
Table S21), B7 = E8 (EGFR mutation, Table 22 and Tables S22-S25). Table and figure numbers follow the manuscript of
18/11/2026.

### 6.1 Folders, commits and equality gates

| Experiment | Folders in `results/survey_v2/` | Code (runner) | Kaggle job, commit | Equality gate (result) |
|---|---|---|---|---|
| B1, 20 few-shot splits for E1b and E7 | `B1_splits/`, `B1_E7_20/`, `B1_E1b_20/`, `B1_tables/` | `build_fewshot_splits.py --seeds 42 … 61`, `run_survey_e7_transfer_naive.py`, `run_survey_e1b_learning_curve.py --splits-file fewshot_splits_20.csv` | `b1`, c50647a (two sessions; the second resumed the first) | splits 0-4 equal `fewshot_splits.csv`; rows of splits 0-4 equal the published E1b and E7 (maximum absolute AUROC difference 0) |
| B2, k-TSP with its voting rule | `B2_ktsp_vote/`, `B2_tables/` | `run_b2_ktsp_vote.py` | `b2_b3_b5`, c50647a | — (stored k-TSP pairs of E1, E1b; pairs selected on TCGA for E7) |
| B3, two zero-shot variants | `B3_zero_shot/`, `B3_tables/` | `run_b3_zero_shot.py` | `b2_b3_b5`, c50647a | the recomputed E7 source-only rule equals the published one (difference 0); source panels equal |
| B4, variants of the reference B0 | `B4_b0_variants/`, `B4_tables/` | `run_b4_b0_variants.py` | `b4`, c50647a | the original B0 rerun equals the published B0 (difference 0, 25 folds x 5 cohorts) |
| B5, published signatures cut to 10 genes inside the fold | `B5_signatures10/`, `B5_tables/` | `run_b5_signatures10.py` | `b2_b3_b5`, c50647a | — |
| B6, ptLasso and Trans-GLM in E7 | `B6_r_inputs/`, `B6_r_inputs_20/`, `B6_E7_gate5/`, `B6_gate/`, `B6_E7_principled/`, `B6_tables/` | `export_e7_r_inputs.py`, `r/e7_r_baselines.R`, `run_e7_r_eval.py` (copied from the method paper's code, commit 7f097da) | notebook `kaggle_B6_e7_r.ipynb`, c50647a | on seeds 42-46, panels and AUROC of every run equal the method paper's run (`B6_reference_rq2_r_7f097da/`; difference 0, 2,200 panels) |
| B7, EGFR mutation in lung adenocarcinoma | `B7_data/`, `B7_gate/`, `E0_b7/`, `E1_feature_selection_b7/`, `E1_classification_b7/`, `E4_E5_b7/`, `B7_tables/` | `run_b7_step1.py`, `build_b7_egfr_datasets.py`, `b7_gate.py`, then the E1 and E4 runners | `b7_step1` (c1828d2); `b7_e1_phaseA`, `b7_e4`, `b7_e1_phaseB` (c50647a) | gates G1 and G2 passed on both data sets (`B7_gate/b7_gate_table.csv`); every later session checks the SHA-256 of the 20 data files against `B7_data/config.json` |

`B6_tables/` and `B7_tables/` were written on a workstation from the result folders above (commits c50647a and
f22597c; f22597c only changes the layout of `b7_kstar.csv`, which now lists every selector and both margins). The
two `expression.h5` files of B7 are not redistributed; their SHA-256 are in `DMKD2_EXCLUDED_DATA.tsv` and the files
are rebuilt by `scripts/run_b7_step1.py` from the public GDC mutation files and GEO series GSE31210.
The FILTER column of the TCGA-LUAD mutation files was not stored at step 0; the labels rest on the open Masked Somatic
Mutation files as released by the GDC (`B7_data/experiment-data-b7/tcga_luad_egfr/data_card_tcga_luad_egfr_notes.md`).

O49 (commit 3fe6d99): in `B1_tables/` and `B6_tables/` the Holm correction of the split-level labels now runs within
each strategy over all its cells on the four cohorts, as the specification states (it previously ran within each
strategy and cohort). The two folders were regenerated from that commit; no experiment was rerun. The confounder
checks of B7 (tissue source site, sex, sex-chromosome genes and EGFR in the panels) are written by
`scripts/b7_confounder_check.py` to `B7_tables/b7_confounder_check.txt`:
`python scripts/b7_confounder_check.py results/survey_v2 > results/survey_v2/B7_tables/b7_confounder_check.txt`.

### 6.2 Regenerate the tables of Section 6 (no experiment is rerun)

From a clean clone of the tag, with the repository as data root:

```
KLTN_DATA_ROOT=. python scripts/make_revision_tables.py b1 b2 b3 b4 b5 b6 b7
python paper_scripts/check_revision_tables.py
```

`make_revision_tables.py` rewrites the folders `results/survey_v2/B*_tables/` from the result folders of Section 6.1
and the published results; `check_revision_tables.py` compares every rewritten CSV file with the committed one
(values, absolute tolerance 1e-9) and prints one `OK` line per file. Only the `config.json` files change (commit,
date).

### 6.3 Supplementary tables and the tables of SA1-SA4 and E8

After Section 4 (the script compares the five files `tab_app_e1_<cohort>.tex` of `make_tables.py` cell by cell and
rewrites them with the marks of the tests, so it works on a copy of them):

```
mkdir -p out/Supplement && cp out/Tables/tab_app_e1_*.tex out/Supplement/
python paper_scripts/make_supplementary_tables.py --official results/survey_v2/tables --extra results/survey_v2 \
       --k2 results/survey_v2/E7_k2_evidence --out out/Supplement
(cd out/Supplement && grep numbers_supplement ../../paper_scripts/EXPECTED_SHA256.txt | sha256sum -c)
```

The script reads result tables only. It writes 25 table files: Table 15 (SA1, `tab_sens_reference.tex`), Table 20
(two-gene rule and SA4 at k = 2, `tab_zeroshot_k2.tex`), Table 22 (E8 at k = 10, `tab_e8.tex`) and Tables S2-S6,
S9-S25 of Online Resource 1, and `numbers_supplement.json`, the derived numbers quoted in the text. It stops if a cell
of Tables S2-S6, without the new marks, differs from the file of `make_tables.py`, or if the k* re-analysis does not
reproduce `e1_kstar_pooled_by_k.csv` and `e1_kstar_pooled.csv`. Every table file names its input files and columns in
its first comment lines. The repository copy differs from the author's copy of 18/11/2026 (SHA-256
`0eade5915d151371b619e1ab1a857f653f95bdfeddfbf37664925989441f157e`) in one place only: the table note of Tables
S2-S6 is optional, because the files of `make_tables.py` have none (a one-line note on the cells is written instead); `numbers_supplement.json` is the same (checksum
above). On Windows, strip the trailing carriage returns before checking (`sed 's/\r$//'`).

### 6.4 Result figures of the paper (Figures 5-12)

```
python paper_scripts/make_figures_dmkd.py --results results/survey_v2/tables --e3 results/survey_v2/E2_E3        --b1 results/survey_v2/B1_tables --out out/Images
```

The script reads the official tables only and writes `Figure_5.pdf` ... `Figure_12.pdf` (E3, E4, E5, E1, pooled lower
bound of k*, E1b, E7, E6) and `figures_dmkd_report.json`. Every figure is drawn at the text width of the manuscript
(372 pt, at most 520 pt high) and inserted without scaling; the main text is 8 pt in one sans-serif family (Liberation
Sans) and no glyph of the written PDF is below 7 pt (`check_figure_lettering.py`, which needs `pdfminer.six`; the page
width is checked with `pikepdf`). The script also checks that every plotted value equals its table value and that no
two texts overlap, and it exits with code 1 if a check fails. Without Liberation Sans installed, Matplotlib falls back
to a wider font and the overlap check of Figure 11 fails. With `--b1`, Figure 11 is drawn from the split-level labels
of B1 after O49 (`B1_tables/b1_e7_split_labels.csv`): cell = 100 x mean difference of AUROC; solid frame = positive or
negative transfer, dashed frame = practically equivalent, no frame = inconclusive. `--only 11` draws one figure.
`make_figures_dmkd2.py` (174 mm layout of Figures 8, 11 and 12 at tag `survey-v3-dmkd.2`) is superseded by this script.

Figure 4 is built from `paper_figure_sources/Figure_4_complexity.tex` with `pdflatex`; `python paper_scripts/check_figure4.py
--tex Figure_4_complexity.tex --csv results/survey_v2/tables/e3_exponents_vs_table5.csv --log Figure_4_complexity.log`
checks its 11 pairs of exponents and its layout.

## 7. Licence and citation

The source code is licensed under the MIT licence (`LICENSE`). The result files, splits and figure sources are licensed
under CC BY 4.0 (`LICENSE-DATA.md`), with the third-party notices of `paper_figure_sources/THIRD_PARTY_NOTICES.md`.
Please cite the paper when you reuse the benchmark; the citation will be added here after publication.
