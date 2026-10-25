# E1: selection and classification as separate phases

`repeated_splits.csv` contains 25 patient-grouped **outer** train/test splits
(5 fixed seeds × 5 folds) for each cohort. It has no `val` role. Validation is
created inside each outer-train partition only when a classifier tunes its
hyperparameters; therefore the outer test partition remains untouched until the
single final evaluation.

For Kaggle, upload the current data dataset after adding each cohort's
`repeated_splits.csv` and the top-level `e1_repeated_splits_manifest.json`.
The feature matrix remains shared: do not upload 50 duplicate HDF5 files.

Run the two phases in distinct notebook cells:

```bash
python scripts/run_e1_feature_selection.py --datasets gse68465_stage_early_late tcga_luad_stage_early_late tcga_lusc_stage_early_late
python scripts/run_e1_classification.py --models knn gaussian_nb logistic_regression decision_tree random_forest linear_svm rbf_svm mlp xgboost lightgbm stacking
```

For a low-cost smoke test, add `--repeats 0 --k 10 --quick` to Phase B (and the
matching `--repeats 0 --k 10` to Phase A). Do not use `--quick` for the final
reported table.

GA and BPSO run separately from filters/embedded methods:

```bash
python scripts/run_e1_wrappers_all_cohorts.py
python scripts/run_e1_classification.py --selection-root /kaggle/working/kltn_data/results/survey_v2/E1_wrapper_feature_selection --methods ga bpso
```

Phase A writes panels, selection time/CPU/RSS/CodeCarbon emissions, selector diagnostics, stability
(pairwise Jaccard), and gene frequency. Phase B writes per-fold and aggregated
classification metrics, confusion matrices, tuning/train/inference timing, and
the best classifier per fixed panel. Legacy `run_survey_e1.py` remains the
combined B0/random-k reference runner; it is not a substitute for this split
workflow.
