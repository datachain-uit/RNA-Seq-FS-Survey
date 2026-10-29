"""Kiểm tra nhất quán giới tính theo biểu hiện gen với metadata cho MỌI mẫu của một bộ (QĐ22, 29/10/2026).

Hai trục: XIST (cao ở nữ) và trung bình các gen liên kết Y có trong bộ (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY; cao ở nam).
Mỗi trục được chia hai cụm bằng k-means (k = 2) **trên chính cohort đó** (tự hiệu chuẩn theo nền tảng GPL570 / GPL6884 /
RNA-seq); trục bị coi là "không phân biệt" nếu hai tâm cụm cách nhau < MIN_GAP đơn vị log2 hoặc một cụm có < MIN_CLUSTER mẫu.

Phân loại theo biểu hiện: F-like (XIST cao, Y thấp) · M-like (XIST thấp, Y cao) · both-high · both-low.
Đối chiếu metadata (cột có tên chứa sex/gender; giá trị bắt đầu bằng f/m):
  match          F-like với nữ, M-like với nam
  opposite       F-like với nam, hoặc M-like với nữ (cả hai trục đều trái metadata)      -> CÁCH LY (QĐ22)
  inconsistent   both-high (XIST và gen Y cùng cao), bất kể metadata                       -> CÁCH LY (QĐ22)
  inconclusive   both-low, hoặc chỉ một trục trái metadata, hoặc một trục không phân biệt -> GIỮ, ghi cờ
  no-metadata    không có cột giới tính (vd. GSE10245): chỉ báo cáo F-like/M-like/both-high; both-high vẫn cách ly

Bộ chỉ có gen mã hoá protein (TCGA v2: không có XIST) — chế độ MỘT trục (QĐ22b, 29/10): chỉ dùng trục gen Y
(--male-axis mean: trung bình các gen Y; --male-axis RPS4Y1: riêng RPS4Y1, marker mạnh nhất); category M-like(Y) / F-like(Y) chỉ khi
giá trị nằm rõ trong một cụm (cách tâm cụm kia hơn 1/4 khoảng cách hai tâm), còn lại intermediate(Y) -> inconclusive (giữ, ghi cờ);
`opposite` = mẫu ghi nữ nằm rõ trong cụm nam (hoặc ngược lại) -> cách ly; không có `inconsistent` ở chế độ này. Cột sex của TCGA lấy từ GDC
(sex_at_birth) và ghi vào metadata.csv bằng script đã commit.

Không sửa nhãn, không sửa giới tính trong metadata; mọi mẫu cách ly ghi vào excluded_samples_<cohort>.csv kèm lý do.

    python scripts/check_sex_consistency.py                       # 5 bộ, ghi $KLTN_DATA_ROOT/qc/sex_check_<bộ>.csv
    python scripts/check_sex_consistency.py --datasets gse42127 --show-all
    python scripts/check_sex_consistency.py --datasets tcga_luad_lusc_primary_tumor --male-axis RPS4Y1   # một trục
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset  # noqa: E402
from survey_v2.runlog import DATA_ROOT  # noqa: E402

FEMALE_GENES = ("XIST",)
MALE_GENES = ("RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY")
MIN_GAP = 2.0        # log2 units between the two cluster centres for an axis to be informative
MIN_CLUSTER = 3      # smallest cluster size for an axis to be informative


def split_axis(values: np.ndarray) -> tuple[float | None, float, float]:
    """Hai cụm k-means trên một trục; trả về (ngưỡng = trung điểm hai tâm hoặc None nếu không phân biệt, tâm thấp, tâm cao)."""
    from sklearn.cluster import KMeans
    v = np.asarray(values, dtype=float).reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(v)
    centres = np.sort(km.cluster_centers_.ravel())
    sizes = np.bincount(km.labels_, minlength=2)
    if centres[1] - centres[0] < MIN_GAP or sizes.min() < MIN_CLUSTER:
        return None, float(centres[0]), float(centres[1])
    return float(centres.mean()), float(centres[0]), float(centres[1])


def classify(xist_high: bool | None, y_high: bool | None) -> str:
    if xist_high is None or y_high is None:
        return "uninformative"
    if xist_high and not y_high:
        return "F-like"
    if y_high and not xist_high:
        return "M-like"
    return "both-high" if xist_high else "both-low"


def verdict(category: str, meta_sex: str | None) -> str:
    if category == "both-high":
        return "inconsistent"
    if meta_sex not in ("female", "male"):
        return "no-metadata"
    if category == "F-like":
        return "match" if meta_sex == "female" else "opposite"
    if category == "M-like":
        return "match" if meta_sex == "male" else "opposite"
    return "inconclusive"          # both-low, uninformative


def normalise_sex(v) -> str | None:
    s = str(v).strip().lower()
    if not s or s in ("nan", "none", "na", "unknown", "n/a"):
        return None
    if s.startswith("f") or s.startswith("w") or s in ("nữ", "nu"):
        return "female"
    if s.startswith("m") or s in ("nam",):
        return "male"
    return None


def classify_single(y: float, thr: float | None, lo: float, hi: float, margin: float = 0.25) -> str:
    """Một trục: chỉ kết luận khi giá trị nằm RÕ trong một cụm (cách tâm cụm kia hơn `margin` × khoảng cách hai tâm);
    vùng giữa hai cụm -> intermediate(Y) (không kết luận)."""
    if thr is None:
        return "uninformative"
    gap = hi - lo
    if y >= hi - margin * gap:
        return "M-like(Y)"
    if y <= lo + margin * gap:
        return "F-like(Y)"
    return "intermediate(Y)"


def verdict_single(category: str, meta_sex: str | None) -> str:
    """Chế độ một trục (không có XIST): `opposite` chỉ khi metadata có và mẫu nằm rõ trong cụm của giới kia; không có `inconsistent`."""
    if category in ("uninformative", "intermediate(Y)"):
        return "inconclusive"
    if meta_sex not in ("female", "male"):
        return "no-metadata"
    want = "male" if category == "M-like(Y)" else "female"
    return "match" if meta_sex == want else "opposite"


def sex_table(ds, meta: pd.DataFrame, male_axis: str = "mean") -> tuple[pd.DataFrame, dict]:
    genes = set(ds.genes.tolist())
    f_cols = [g for g in FEMALE_GENES if g in genes]
    m_cols = [g for g in MALE_GENES if g in genes]
    info = {"female_genes": f_cols, "male_genes": m_cols, "male_axis": male_axis}
    if not m_cols:
        return pd.DataFrame(), {**info, "error": "không có gen Y nào trong bộ"}
    if male_axis != "mean" and male_axis not in m_cols:
        return pd.DataFrame(), {**info, "error": f"gen {male_axis} không có trong bộ"}
    gi = {g: int(np.flatnonzero(ds.genes == g)[0]) for g in f_cols + m_cols}
    ymean = ds.X[:, [gi[g] for g in m_cols]].mean(axis=1) if male_axis == "mean" else ds.X[:, gi[male_axis]]
    thr_y, lo_y, hi_y = split_axis(ymean)
    info.update({"y_threshold": thr_y, "y_centres": (lo_y, hi_y)})
    single = not f_cols                       # TCGA v2: chỉ gen mã hoá protein, không có XIST -> một trục (QĐ22b)
    info["mode"] = "single-axis (Y only)" if single else "two-axis (XIST, Y)"
    if not single:
        xist = ds.X[:, gi["XIST"]]
        thr_x, lo_x, hi_x = split_axis(xist)
        info.update({"xist_threshold": thr_x, "xist_centres": (lo_x, hi_x)})
    # vùng trung gian của trục Y: dưới ngưỡng nhưng trên tâm cụm thấp + max(3 SD cụm thấp, 1/4 khoảng cách hai tâm) — chỉ để ghi cờ
    low_sd = float(np.std(ymean[ymean <= (thr_y if thr_y is not None else np.inf)])) if thr_y is not None else float("nan")
    sex_cols = [c for c in meta.columns if "sex" in c.lower() or "gender" in c.lower()]
    info["metadata_sex_column"] = sex_cols[0] if sex_cols else None
    meta_sex = meta[sex_cols[0]].map(normalise_sex) if sex_cols else pd.Series([None] * len(meta), index=meta.index)
    rows = []
    for i, sid in enumerate(ds.sample_ids):
        yh = None if thr_y is None else bool(ymean[i] > thr_y)
        ms = meta_sex.get(sid) if sid in meta_sex.index else None
        if single:
            cat = classify_single(float(ymean[i]), thr_y, lo_y, hi_y)
            ver = verdict_single(cat, ms)
            x_val = float("nan")
        else:
            xh = None if thr_x is None else bool(xist[i] > thr_x)
            cat, ver = classify(xh, yh), verdict(classify(xh, yh), ms)
            x_val = float(xist[i])
        intermediate = bool(thr_y is not None and (not yh) and ymean[i] > lo_y + max(3 * low_sd, 0.25 * (hi_y - lo_y)))
        rows.append({"sample_id": sid, "label": int(ds.y[i]), "XIST": x_val, "y_axis": float(ymean[i]),
                     **{g: float(ds.X[i, gi[g]]) for g in m_cols},
                     "category": cat, "y_intermediate": intermediate, "meta_sex": ms, "verdict": ver,
                     "action": {"opposite": "quarantine", "inconsistent": "quarantine"}.get(ver, "keep")})
    return pd.DataFrame(rows), info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--out", type=Path, default=None, help="mặc định $KLTN_DATA_ROOT/qc")
    ap.add_argument("--show-all", action="store_true", help="in mọi mẫu, không chỉ mẫu bị cờ")
    ap.add_argument("--male-axis", default="mean", help="'mean' (trung bình gen Y) hoặc tên một gen Y, vd. RPS4Y1")
    args = ap.parse_args(argv)
    out = args.out or (DATA_ROOT / "qc")
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in args.datasets:
        ds = load_dataset(DATA_ROOT, name, gene_space=None)
        meta = pd.read_csv(ds.data_dir / "metadata.csv")
        meta = meta.set_index("sample_id") if "sample_id" in meta.columns else pd.read_csv(ds.data_dir / "metadata.csv", index_col=0)
        meta.index = meta.index.astype(str)
        table, info = sex_table(ds, meta, male_axis=args.male_axis)
        if table.empty:
            print(f"[{name}] bỏ qua: {info.get('error')}", flush=True)
            continue
        table.to_csv(out / f"sex_check_{name}.csv", index=False)
        counts = table.verdict.value_counts().to_dict()
        summary.append({"dataset": name, "n": len(table), "mode": info["mode"], "male_axis": info["male_axis"],
                        "metadata_sex_column": info["metadata_sex_column"],
                        "xist_threshold": info.get("xist_threshold"), "y_threshold": info["y_threshold"], **counts,
                        "n_intermediate": int(table.y_intermediate.sum()), "n_quarantine": int((table.action == "quarantine").sum())})
        print(f"[{name}] n = {len(table)}; {info['mode']}; ngưỡng XIST = {info.get('xist_threshold')} (tâm {info.get('xist_centres')}), "
              f"ngưỡng Y ({info['male_axis']}) = {info['y_threshold']} (tâm {info['y_centres']}); cột giới tính: {info['metadata_sex_column']}; "
              f"verdict: {counts}; trung gian trên trục Y: {int(table.y_intermediate.sum())}", flush=True)
        flagged = table if args.show_all else table[table.action.eq("quarantine") | table.verdict.eq("inconclusive") | table.y_intermediate]
        if len(flagged):
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                print(flagged.round(2).to_string(index=False), flush=True)
    pd.DataFrame(summary).to_csv(out / "sex_check_summary.csv", index=False)
    print("wrote:", out, flush=True)


if __name__ == "__main__":
    main()
