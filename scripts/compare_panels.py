"""Kiểm tra bằng nhau giữa hai bộ panel pha A (QĐ19, Review 29/10/2026).

Dùng để quyết định có tái dùng panel pha A chạy trước 29/10 (code v2 + commit 2508a2a) hay không: chạy lại MỘT lần lặp
(`--repeats 0`) bằng v3.1, rồi so từng (cohort, method, k, repeat, fold). Một phương pháp chỉ được tái dùng khi
**mọi** ô so sánh có cùng tập gen (và cùng thứ tự với phương pháp xếp hạng). Không có "gần bằng".

    python scripts/compare_panels.py --old /path/E1_feature_selection_2026-10-28 \\
        --new $KLTN_DATA_ROOT/results/survey_v2/E1_feature_selection --datasets tcga_luad_lusc_primary_tumor --repeats 0

Đầu ra: `panel_equality.csv` (từng ô) và `panel_equality_summary.csv` (từng cohort × method) trong --out (mặc định: --new/../panel_equality),
kèm in bảng tóm tắt. Mã thoát 0 khi mọi phương pháp được so đều khớp, 1 khi có phương pháp lệch (để dùng trong notebook/CI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402

KEY = ["cohort", "repeat", "fold", "method", "k"]


def read_panels(root: Path, datasets) -> pd.DataFrame:
    frames = []
    for name in datasets:
        f = Path(root) / name / "selected_genes.csv"
        if not f.exists():
            print(f"[compare] thiếu {f}", file=sys.stderr)
            continue
        g = pd.read_csv(f)
        if "cohort" not in g.columns:
            g["cohort"] = name
        frames.append(g[KEY + ["rank", "gene"]])
    if not frames:
        raise SystemExit(f"không đọc được selected_genes.csv nào dưới {root}")
    return pd.concat(frames, ignore_index=True)


def panel_table(df: pd.DataFrame) -> pd.DataFrame:
    """Một dòng mỗi (cohort, repeat, fold, method, k): tập gen (frozenset) và dãy gen theo rank."""
    df = df.sort_values(KEY + ["rank"])
    return df.groupby(KEY, sort=False).agg(genes=("gene", lambda s: tuple(s.astype(str))),
                                          n=("gene", "size")).reset_index()


def compare(old: pd.DataFrame, new: pd.DataFrame, repeats, methods=None, ranked=C.RANKING_METHODS) -> pd.DataFrame:
    a, b = panel_table(old), panel_table(new)
    a = a[a.repeat.isin(repeats)]
    b = b[b.repeat.isin(repeats)]
    if methods:
        a, b = a[a.method.isin(methods)], b[b.method.isin(methods)]
    m = a.merge(b, on=KEY, how="outer", suffixes=("_old", "_new"), indicator="presence")
    rows = []
    for r in m.itertuples(index=False):
        if r.presence != "both":
            rows.append({**{k: getattr(r, k) for k in KEY}, "present_in": str(r.presence).replace("left_only", "old_only").replace("right_only", "new_only"),
                         "same_set": False, "same_order": False, "jaccard": float("nan"), "n_old": r.n_old, "n_new": r.n_new})
            continue
        so, sn = set(r.genes_old), set(r.genes_new)
        rows.append({**{k: getattr(r, k) for k in KEY}, "present_in": "both", "same_set": so == sn,
                     "same_order": tuple(r.genes_old) == tuple(r.genes_new),
                     "jaccard": len(so & sn) / len(so | sn) if so | sn else float("nan"), "n_old": r.n_old, "n_new": r.n_new})
    out = pd.DataFrame(rows)
    out["required_equal"] = out.apply(lambda r: r.same_order if r.method in ranked else r.same_set, axis=1)
    return out.sort_values(KEY).reset_index(drop=True)


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    g = cells.groupby(["cohort", "method"], sort=True)
    s = g.agg(n_cells=("required_equal", "size"), n_equal=("required_equal", "sum"), n_same_set=("same_set", "sum"),
              n_missing=("present_in", lambda x: int((x != "both").sum())), jaccard_min=("jaccard", "min")).reset_index()
    s["fraction_equal"] = s.n_equal / s.n_cells
    s["verdict"] = s.apply(lambda r: "REUSE" if r.n_equal == r.n_cells and r.n_missing == 0 else "RERUN", axis=1)
    return s


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, required=True, help="thư mục kết quả cũ chứa <cohort>/selected_genes.csv")
    ap.add_argument("--new", type=Path, required=True, help="thư mục kết quả chạy lại bằng v3.1")
    ap.add_argument("--datasets", nargs="+", default=[C.SOURCE_DATASET])
    ap.add_argument("--repeats", nargs="+", type=int, default=[0])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cells = compare(read_panels(args.old, args.datasets), read_panels(args.new, args.datasets), args.repeats, args.methods)
    summary = summarise(cells)
    out = args.out or (Path(args.new).parent / "panel_equality")
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "panel_equality.csv", index=False)
    summary.to_csv(out / "panel_equality_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summary.to_string(index=False))
    print(f"\nwrote: {out}")
    return 0 if (summary.verdict == "REUSE").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
