# KLTN – Feature selection & extraction benchmark on lung cancer expression data

Notebooks for comparing feature-selection (FS) and feature-extraction (FE) methods, followed by a classifier grid search, on three lung cancer datasets: **GSE68465**, **TCGA-LUAD** and **TCGA-LUSC**.

All compute runs on Kaggle/Colab. The notebooks read the prepared h5 data from the Kaggle dataset `lfreedom2750/experiment-dataset`, with one folder `h5/<dataset>/` per dataset. Each folder holds 5 folds, and each fold has `train/val/test.h5` (a pandas table with key `"data"`, index = sample_id, gene columns + `label`). The expression values are raw log2 with no scaling.

## Structure

```
kltn/
├── notebooks/
│   ├── 00_data/                 download GEO/TCGA, map probes -> gene symbols, assign labels, split 5 folds, write h5
│   ├── 01_feature_selection/
│   │   ├── filter/              f_classif, mrmr, multisurf, mutual_info, relieff
│   │   ├── wrapper/             bpso, ga, gwo_pso, ibabc_cgo, sga
│   │   ├── embedded/            lasso, random_forest, ridge, scfsnn, svm_rfe, wsbnn
│   │   ├── extraction/          pca, kpca, chnmf, autoencoder
│   │   └── fs_quicktest_rf.ipynb
│   ├── 02_classification/       classifier grid search on the FS gene sets / FE data / all genes / BulkFormer embeddings
│   └── 04_analysis/             permutation test by stage, plots
└── scripts/
    ├── aggregate_5fold_results.py   5-fold means for FS and classification
    └── stage_split_metrics.py
```

## Pipeline

1. **Data** (`00_data`): produces `h5/<dataset>/fold_1..5/{train,val,test}.h5`.
2. **FS / FE** (`01_feature_selection`): each `fs_<method>.ipynb` / `fe_<method>.ipynb` notebook runs on a single dataset with `N_FEATURES ∈ {50, 100, 200, 500}`, and outputs `fs_<method>_<dataset>.zip` / `fe_<method>_<dataset>.zip`.
   - FS: `selected_genes/fold_k.csv` (`fold, rank, gene, score`), `metrics_per_fold.csv`, `config.json`.
   - FE: the transformed data, `features/fold_k/{train,val,test}.h5`, in the same format as the original h5.
   - Note: PCA uses at most `n_train` components and KPCA at most `n_train − 1`. With N = 500, every dataset has fewer training samples than that, so the number of dimensions varies by fold.
3. **Classification** (`02_classification`): 11 models (decision_tree, gaussian_nb, knn, lightgbm, linear_svm, logistic_regression, mlp, random_forest, rbf_svm, stacking, xgboost). Hyperparameters are tuned on val and evaluated on test, for each fold. Output: `results_<group>.csv` (one row per fold × model) and `summary_<group>.csv`.
   - `clf_gridsearch_fs_group.ipynb`: set `FS_DIR` to the folder of gene lists for one `<N>/<group>`.
   - `clf_gridsearch_fe.ipynb`: set `DATA_DIR` to the dimension-reduced data and use all of its columns.
   - `clf_gridsearch_no_fs.ipynb`: all genes (baseline).
4. **Analysis** (`04_analysis`, `scripts/`): 5-fold means, permutation test, plots.

`scripts/aggregate_5fold_results.py` still has a hard-coded path to the original machine (`CLF_ROOT`), so update it before running.
