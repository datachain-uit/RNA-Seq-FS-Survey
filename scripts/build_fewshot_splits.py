"""Tạo và kiểm tra bản chia few-shot (test_T / pool_T × tập con n_T) cho các cohort GEO — dùng chung E1b, E7, RQ2, RQ4.

Ghi ``<thư mục dữ liệu cohort>/fewshot_splits.csv`` (hoặc ``--out-root/<cohort>/fewshot_splits.csv``) và
``$KLTN_DATA_ROOT/experiment-data-geo-target/fewshot_splits_manifest.json`` (SHA-256 từng tệp). Giao thức: KBTN v2 Mục 6
(5 lần chia 1/3 : 2/3 phân tầng, seed 42–46; 4 tập con mỗi mức n_T ∈ {5, 10, 20, 40, pool} mỗi lớp, lồng nhau; GSE10245
chỉ n_T ∈ {5, 10}).

    python scripts/build_fewshot_splits.py
    python scripts/build_fewshot_splits.py --datasets gse37745 --out-root $KLTN_DATA_ROOT/splits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.fewshot import build_fewshot_manifest, validate_fewshot_manifest  # noqa: E402
from survey_v2.runlog import DATA_ROOT, start_run, write_config  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--out-root", type=Path, default=None, help="mặc định: ghi cạnh dữ liệu của cohort")
    ap.add_argument("--out-name", default="fewshot_splits.csv")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def fewshot_path(ds, out_root: Path | None, out_name: str = "fewshot_splits.csv") -> Path:
    return (Path(out_root) / ds.name / out_name) if out_root else ds.data_dir / out_name


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    report = []
    for name in args.datasets:
        ds = load_dataset(DATA_ROOT, name)
        path = fewshot_path(ds, args.out_root, args.out_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.force:
            man = pd.read_csv(path, dtype={"n_t": str})
            errors = validate_fewshot_manifest(man)
            if errors:
                raise ValueError(f"{path} đã có nhưng không hợp lệ: {errors}; dùng --force sau khi kiểm tra")
        else:
            man = build_fewshot_manifest(name, ds.sample_ids, ds.groups, ds.y)
            man.to_csv(path, index=False, encoding="utf-8")
        levels = sorted({v for v in man.n_t.unique()}, key=lambda t: 10**9 if t == "pool" else int(t))
        report.append({"dataset": name, "path": str(path), "rows": int(len(man)), "n_t_levels": levels,
                       "n_splits": int(man.split.nunique()), "n_subsets": int(man.subset.nunique()), "sha256": sha256(path),
                       "patient_id_assumed_from_sample_id": ds.notes.get("patient_id_assumed_from_sample_id")})
        print(f"[{name}] {len(man)} dòng | n_T {levels} | {path}", flush=True)
    out = (args.out_root or DATA_ROOT / C.GEO_TARGET_DIR) / "fewshot_splits_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol": "5 splits test_T/pool_T (1/3:2/3, stratified) x 4 nested subsets per n_T per class",
                               "seeds": list(C.REPEAT_SEEDS[: C.FEWSHOT_SPLITS]), "n_t_grid": [v if v is not None else "pool" for v in C.N_T_GRID],
                               "files": report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_config(out.parent, prov, {"experiment": "fewshot_splits", "files": report},
                 data_files=[Path(r["path"]) for r in report])
    print("wrote:", out)


if __name__ == "__main__":
    main()
