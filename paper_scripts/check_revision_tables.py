#!/usr/bin/env python3
"""check_revision_tables.py -- after `KLTN_DATA_ROOT=. python scripts/make_revision_tables.py b1 b2 b3 b4 b5 b6 b7` on a clean
clone of the tag, compare every CSV file of results/survey_v2/B*_tables/ with the committed version (git HEAD).

Values are compared column by column: numbers with absolute tolerance --atol (default 1e-9; floating-point
summation may differ in the last digits between machines), everything else exactly. Line endings are ignored.
Prints one line per file (OK or DIFF) and exits with code 1 if any file differs or is missing.

    python paper_scripts/check_revision_tables.py
"""
import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TABLE_DIRS = ["B1_tables", "B2_tables", "B3_tables", "B4_tables", "B5_tables", "B6_tables", "B7_tables"]


def committed(rel: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def same(a: pd.DataFrame, b: pd.DataFrame, atol: float) -> tuple[bool, str]:
    if list(a.columns) != list(b.columns):
        return False, f"columns differ: {list(a.columns)} vs {list(b.columns)}"
    if len(a) != len(b):
        return False, f"rows {len(a)} vs {len(b)}"
    for c in a.columns:
        x, y = a[c], b[c]
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        numeric = xn.notna().sum() == x.notna().sum() and yn.notna().sum() == y.notna().sum() and x.notna().any()
        if numeric:
            xa, ya = xn.to_numpy(float), yn.to_numpy(float)
            ok = (np.isnan(xa) == np.isnan(ya)).all() and np.allclose(np.nan_to_num(xa), np.nan_to_num(ya), rtol=0, atol=atol)
            if not ok:
                return False, f"column {c}: max |diff| {np.nanmax(np.abs(xa - ya)):.3g}"
        elif not x.fillna("<NA>").astype(str).equals(y.fillna("<NA>").astype(str)):
            return False, f"column {c}: text differs"
    return True, ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atol", type=float, default=1e-9)
    a = ap.parse_args(argv)
    bad = 0
    n = 0
    for d in TABLE_DIRS:
        rel_dir = f"results/survey_v2/{d}"
        listed = subprocess.run(["git", "ls-files", rel_dir], cwd=REPO, capture_output=True, text=True).stdout.split()
        for rel in sorted(x for x in listed if x.endswith(".csv")):
            n += 1
            old = committed(rel)
            path = REPO / rel
            if old is None or not path.exists():
                print(f"MISSING {rel}")
                bad += 1
                continue
            ok, why = same(pd.read_csv(io.BytesIO(old)), pd.read_csv(path), a.atol)
            print(f"{'OK  ' if ok else 'DIFF'} {rel}" + (f"  ({why})" if why else ""))
            bad += not ok
    print(f"{n - bad}/{n} OK")
    return 1 if bad or n == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
