# Data card — gse31210_egfr

- Role: b7_harder_task (B7, QĐ177); platform: GPL570
- Source: NCBI GEO GSE31210 series matrix (GPL570); probe → gene by GPL570.annot.gz (column 'Gene symbol'), functions of geo_luad_lusc_target_cohorts_5fold.ipynb
- Label rule: 1 = 'EGFR mutation +'; 0 = 'EGFR/KRAS/ALK -', 'KRAS mutation +', 'ALK-fusion +' (QĐ185); 'normal lung' excluded; characteristics read by the key prefix of each cell (Y98)
- After QC: n = 224, wild-type/negative = 98, mutant/positive = 126 (positive rate 0.5625); patients = 224
- Gene space |G| = 16479; scale: {'p99': 11146.892401999994, 'threshold': 100.0, 'scale': 'linear → log2(max(x, 1))'}
- Label exclusions: 20; QC exclusions: 2

## SHA-256

- `expression.h5`: `6d2a9fbfc65194d2acb7b3b520006a54af244eec2f1fe06255d5e03a81019d0e`
- `metadata.csv`: `2679de2dc105205b880e30de5c537b81dc8585bd24c3427e54dc4ee330514b00`
- `repeated_splits.csv`: `91ff5e92e75eff26159bcba1310abb787b4b2c536e460c0faac4913ddc5dfa05`
- `excluded_samples_gse31210_egfr.csv`: `91609a5db249d09a3e16fea8e43d2c26b903d776ee1a9c8b650683472fd63584`
- `b7_labels_all.csv`: `dddfd91c0a30c3ebf45fd38feb6cc1c0b986072512a1d57e0218ce067acfa2aa`
