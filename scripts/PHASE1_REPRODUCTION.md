# Phase 1 result consolidation

This directory contains the reproducible code added for the three completed review tasks.

## Files

- `build_review_completion_artifacts.py`: inventories the Phase 0 stage-split and permutation evidence, writes `docs/phase0_results_manifest.{md,csv}`, and writes `docs/data_card.md`.
- `run_rq0_prime.py`: runs RQ0' repeated stratified 5x5 CV on the four GEO LUAD-LUSC cohorts. Feature selection occurs within each training fold. It writes fold-level metrics, a summary, configuration and README to `results/phase 1/rq0_prime/`.
- `tests/test_build_review_completion_artifacts.py`: regression tests for the Phase 0 manifest builder and the balanced nT=5 sampler.
- `provenance.py`: shared provenance recorder (Review 20/10/2026, §7.7). Every experiment script calls it at the start of `main()`; see "Provenance" below.
- `build_rq0_prime_workbook.py`: builds `RQ0-prime-performance-ceiling.xlsx` (cohort-level ceiling verdict + summary).
- `tests/test_rq0_ceiling.py`, `tests/test_provenance.py`: regression tests for the RQ0' ceiling rule (O14) and provenance (O3, Y8).

## Requirements

Python 3.10+ with `numpy`, `pandas`, `tables` (PyTables for HDF5), and `scikit-learn`.

## Run

From the repository root. Data and results live outside the repository; point the scripts at them with `KLTN_DATA_ROOT` (it must contain `experiment-data-geo-target/` etc.):

```powershell
$env:KLTN_DATA_ROOT = "D:\KLTN"
python -m unittest tests.test_build_review_completion_artifacts tests.test_rq0_ceiling tests.test_provenance
python scripts/build_review_completion_artifacts.py
python scripts/run_rq0_prime.py
python scripts/build_rq0_prime_workbook.py
```

## Provenance

Experiment scripts stop unless the working directory is a git clone with no uncommitted changes to tracked files; they then write the commit hash, script arguments, library versions, machine and input SHA-256 next to their outputs. Run scripts from this repository (the git clone), never from a copy in the data folder. Only as a last resort can a run be forced with `$env:KLTN_ALLOW_UNTRACKED = "1"`; such outputs record `commit_hash: null` and `tracked: false` and must not be cited. Notebook inventory and the notebook provenance cell: `docs/notebook_manifest.md`.

RQ0' ceiling verdict is made per cohort at nT=5 from best one gene and IHC-2 only (`ceiling_by_cohort.csv`). RQ0' uses five shuffled StratifiedKFold repeats (seeds 42-46), evaluates B0, best one gene, IHC-2, IHC-4 and random-k (1, 2, 3, 5, 10, 25, 50), and repeats each method at full training size and nT=5 samples per class. It is an independent benchmark and does not rerun the existing Phase 1 classifier pipelines.
