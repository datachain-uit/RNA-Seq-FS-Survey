# Data card — tcga_luad_egfr

- Role: b7_harder_task (B7, QĐ177); platform: RNA-seq (log2 TPM+1)
- Source: GDC Data Release 46.0 - August 10, 2026; TCGA-LUAD Masked Somatic Mutation (open), 618 MAF files; expression = tcga_luad_lusc_primary_tumor (log2(TPM + 1), same genes and QC)
- Label rule: 1 = >= 1 activating EGFR kinase-domain variant (ex19 in-frame del, L858R, L861Q, G719X, S768I, ex20 in-frame ins); 0 = no non-synonymous EGFR variant; other EGFR variants and samples without an open MAF excluded (QĐ186)
- After QC: n = 493, wild-type/negative = 435, mutant/positive = 58 (positive rate 0.1176); patients = 493
- Gene space |G| = 19938; scale: log2(TPM + 1) (as source)
- Label exclusions: 22; QC exclusions: 0

## SHA-256

- `expression.h5`: `f688eb7b5c6785933915f5572b1a8143780953f0e36f4b1f3048a7f53176960d`
- `metadata.csv`: `63ca7cade4d45b2ca64dae7e97dbc635d0b0b6101aa1dec09761ed8c13ebbdc5`
- `repeated_splits.csv`: `6302d7f2e752fc80e7fa2b6675ed5451efddf53b82ba35323fa12c0475d6a7f6`
- `excluded_samples_tcga_luad_egfr.csv`: `319997af3f3d89bc5d4077171d754ffa865c2fa73ca7e913e8c2e37b21561a04`
- `b7_labels_all.csv`: `8e7807ca5457a3e09944c06f73c6170b4333e178bb551cb94636133d1072ef3f`

## FILTER of EGFR, TP53 rows (recorded only; labels unchanged)

- EGFR · nan: 89
- TP53 · nan: 264
