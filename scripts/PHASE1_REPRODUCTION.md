# Phase 1 result consolidation

This directory contains the reproducible code added for the three completed review tasks.

## Files

- `build_review_completion_artifacts.py`: inventories the Phase 0 stage-split and permutation evidence, writes `docs/phase0_results_manifest.{md,csv}`, and writes `docs/data_card.md`.
- `run_rq0_prime.py`: runs RQ0' repeated stratified 5x5 CV on the four GEO LUAD-LUSC cohorts. Feature selection occurs within each training fold. It writes fold-level metrics, a summary, configuration and README to `results/phase 1/rq0_prime/`.
- `tests/test_build_review_completion_artifacts.py`: regression tests for the Phase 0 manifest builder and the balanced nT=5 sampler.

## Requirements

Python 3.10+ with `numpy`, `pandas`, `tables` (PyTables for HDF5), and `scikit-learn`.

## Run

From the repository root, after placing the prepared GEO input under `experiment-data-geo-target/`:

```powershell
python -m unittest tests.test_build_review_completion_artifacts
python scripts/build_review_completion_artifacts.py
python scripts/run_rq0_prime.py
```

RQ0' uses five shuffled StratifiedKFold repeats (seeds 42-46), evaluates B0, best one gene, IHC-2, IHC-4 and random-k (1, 2, 3, 5, 10, 25, 50), and repeats each method at full training size and nT=5 samples per class. It is an independent benchmark and does not rerun the existing Phase 1 classifier pipelines.
