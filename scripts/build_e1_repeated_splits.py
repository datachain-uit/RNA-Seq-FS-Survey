"""Create and validate precomputed patient-grouped 5x5 E1 outer splits.

The resulting ``repeated_splits.csv`` files belong beside each cohort in the
Kaggle input Dataset.  E1 feature selection and classification consume these
fixed assignments and therefore never create a split at run time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.e1_splits import build_repeated_split_manifest, validate_repeated_split_manifest  # noqa: E402
from survey_v2.runlog import DATA_ROOT, start_run, write_config  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--out-name", default="repeated_splits.csv")
    ap.add_argument("--force", action="store_true", help="replace an existing manifest only after validation")
    return ap.parse_args(argv)


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params=vars(args))
    report = []
    for name in args.datasets:
        ds = load_dataset(DATA_ROOT, name)
        path = ds.data_dir / args.out_name
        if path.exists() and not args.force:
            existing = __import__("pandas").read_csv(path)
            errors = validate_repeated_split_manifest(existing)
            if errors:
                raise ValueError(f"{path} exists but is invalid: {errors}; pass --force after investigation")
            manifest = existing
        else:
            manifest = build_repeated_split_manifest(ds.sample_ids, ds.groups, ds.y, seeds=C.REPEAT_SEEDS)
            manifest.to_csv(path, index=False, encoding="utf-8")
        errors = validate_repeated_split_manifest(manifest)
        if errors:
            raise ValueError(f"{name}: generated manifest invalid: {errors}")
        report.append({"dataset": name, "path": str(path), "rows": len(manifest), "sha256": sha256(path)})
        print(f"[{name}] {len(manifest)} rows | {path}", flush=True)
    out = DATA_ROOT / C.PERM_DATA_DIR / "e1_repeated_splits_manifest.json"
    out.write_text(json.dumps({"protocol": "5 repeats x 5 patient-grouped outer folds", "seeds": list(C.REPEAT_SEEDS),
                               "files": report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_config(out.parent, prov, {"experiment": "E1_precomputed_splits", "files": report},
                 data_files=[Path(item["path"]) for item in report])
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
