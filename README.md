# RNA-SEQ Feature Selection & Deep Learning Benchmark
### Comprehensive Benchmarking of FS Algorithms and Classifiers on Lung Cancer RNA-Seq (GSE131907)

This repository provides a complete, reproducible pipeline for:
- Preprocessing RNA-Seq gene expression data  
- Running multiple feature selection (FS) algorithms  
- Measuring time complexity, memory, energy, and carbon emission  
- Training classical ML and deep learning classifiers  
- Logging metrics & experiment results  
- Producing a full benchmarking framework for scientific analysis

The project follows a **clean, data-science oriented structure**, separating configs, scripts, modules, and results for maximum clarity and reproducibility.

---

## 🔧 1. Project Structure

```
RNA-SEQ-FS-SURVEY/
│
├── notebooks/
│   └── prepare_data.ipynb
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── run_chi_square.py
│   ├── run_mutual_info.py
│   ├── run_fcbf.py
│   ├── run_mrmr.py
│   ├── run_lasso.py
│   ├── run_svm_rfe.py
│   ├── run_ga_svm.py
│   ├── run_pso_svm.py
│   ├── run_classical_ml.py
│   ├── run_classification_dl.py
│   └── sync_to_sheets.py
│
├── fs/
│   ├── chi2.py
│   ├── mutual_info.py
│   ├── fcbf.py
│   ├── mrmr.py
│   ├── lasso_fs.py
│   ├── svm_rfe.py
│   ├── ga_svm.py
│   ├── pso_svm.py
│   └── utils_fs.py
│
├── deep/
│   ├── mlp.py
│   ├── lstm.py
│   ├── gru.py
│   ├── cnn1d.py
│   ├── transformer.py
│   ├── vae.py
│   ├── gan.py
│   ├── train_dl.py
│   └── utils_dl.py
│
├── models/
│   └── saved_weights/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── results/
│   ├── fs_masks/
│   ├── fs_metrics/
│   ├── classification_metrics/
│   ├── dl_logs/
│   └── figures/
│
├── configs/
│   ├── paths.py
│   ├── fs_config.py
│   ├── model_config.py
│   └── sheet_config.py
│
├── toolkit/
│   └── metric_toolkit.py
│
└── README.md
```

---

## 🚀 2. Installation

```bash
pip install -r requirements.txt
```

---

## 🧬 3. Data Preparation

```bash
python scripts/prepare_dataset.py
```

---

## ⚙️ 4. Run Feature Selection

```bash
python scripts/run_chi_square.py
python scripts/run_lasso.py
python scripts/run_svm_rfe.py
python scripts/run_ga_svm.py
```

---

## 🤖 5. Train Classical ML Models

```bash
python scripts/run_classical_ml.py
```

---

## 🧠 6. Train Deep Learning Models

```bash
python scripts/run_classification_dl.py
```

## 📜 License
MIT License
