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

Bộ chỉ có gen mã hoá protein (TCGA v2: không có XIST) — chế độ MỘT trục gen Y, bỏ phiếu theo từng gen (QĐ22b, 29/10):
mỗi gen Y có ngưỡng riêng = trung điểm giữa phân vị 95 của nhóm ghi nữ và phân vị 5 của nhóm ghi nam (hiệu chuẩn nửa giám sát
bằng cột giới tính của metadata; bền với vài mẫu ghi sai vì chỉ dùng phân vị); khi không có cột giới tính hoặc một nhóm < 30 mẫu
thì dùng trung điểm hai cụm k-means của từng gen. Mẫu là M-like(Y) khi ≥ (số gen − 1) gen vượt ngưỡng, F-like(Y) khi ≤ 1 gen vượt,
còn lại intermediate(Y) -> inconclusive (giữ, ghi cờ). `opposite` = mẫu ghi nữ là M-like(Y) hoặc mẫu ghi nam là F-like(Y) -> cách ly;
không có `inconsistent` ở chế độ này. Lý do không dùng trung bình gen Y với biên 1/4: u của nam thường mất một phần nhiễm sắc thể Y
(biểu hiện Y giảm nhưng vẫn trên mức nữ ở mọi gen), trục trung bình gộp các mẫu này với nữ (5 "opposite" giả trên TCGA ngày 29/10).

Không sửa nhãn, không sửa giới tính trong metadata; mọi mẫu cách ly ghi vào excluded_samples_<cohort>.csv kèm lý do.

    python scripts/check_sex_consistency.py                       # 5 bộ, ghi $KLTN_DATA_ROOT/qc/sex_check_<bộ>.csv
    python scripts/check_sex_consistency.py --datasets gse42127 --show-all
    python scripts/check_sex_consistency.py --datasets tcga_luad_lusc_primary_tumor                      # một trục (bỏ phiếu)
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


def single_axis_thresholds(Y: np.ndarray, meta_sex: pd.Series, min_group: int = 30) -> tuple[np.ndarray, str]:
    """Ngưỡng từng gen Y. Nửa giám sát: trung điểm (phân vị 95 nhóm ghi nữ, phân vị 5 nhóm ghi nam) — phân vị 95/5 thay vì 99/1
    để vài mẫu ghi sai không kéo ngưỡng (bền cả với cohort vài chục mẫu); không đủ hai nhóm (< 30) -> trung điểm k-means từng gen."""
    sexes = meta_sex.to_numpy(dtype=object)
    f = np.array([x == "female" for x in sexes]); m = np.array([x == "male" for x in sexes])
    if f.sum() >= min_group and m.sum() >= min_group:
        thr = (np.quantile(Y[f], 0.95, axis=0) + np.quantile(Y[m], 0.05, axis=0)) / 2
        return thr, "midpoint(female q95, male q05)"
    thr = np.array([split_axis(Y[:, j])[0] if split_axis(Y[:, j])[0] is not None else np.nan for j in range(Y.shape[1])])
    return thr, "kmeans midpoint per gene"


def classify_single(votes: int, n_genes: int) -> str:
    if n_genes < 3:
        return "uninformative"
    if votes >= n_genes - 1:
        return "M-like(Y)"
    if votes <= 1:
        return "F-like(Y)"
    return "intermediate(Y)"


def verdict_single(category: str, meta_sex: str | None) -> str:
    """Chế độ một trục (không có XIST): `opposite` chỉ khi metadata có và mẫu thuộc nhóm của giới kia; không có `inconsistent`."""
    if category in ("uninformative", "intermediate(Y)"):
        return "inconclusive"
    if meta_sex not in ("female", "male"):
        return "no-metadata"
    want = "male" if category == "M-like(Y)" else "female"
    return "match" if meta_sex == want else "opposite"


def sex_table(ds, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    genes = set(ds.genes.tolist())
    f_cols = [g for g in FEMALE_GENES if g in genes]
    m_cols = [g for g in MALE_GENES if g in genes]
    info = {"female_genes": f_cols, "male_genes": m_cols}
    if not m_cols:
        return pd.DataFrame(), {**info, "error": "không có gen Y nào trong bộ"}
    gi = {g: int(np.flatnonzero(ds.genes == g)[0]) for g in f_cols + m_cols}
    Y = ds.X[:, [gi[g] for g in m_cols]]
    ymean = Y.mean(axis=1)
    sex_cols = [c for c in meta.columns if "sex" in c.lower() or "gender" in c.lower()]
    info["metadata_sex_column"] = sex_cols[0] if sex_cols else None
    meta_sex = meta[sex_cols[0]].map(normalise_sex) if sex_cols else pd.Series([None] * len(meta), index=meta.index)
    meta_sex = meta_sex.reindex(ds.sample_ids)
    single = not f_cols                       # TCGA v2: chỉ gen mã hoá protein, không có XIST -> một trục (QĐ22b)
    info["mode"] = "single-axis (Y vote)" if single else "two-axis (XIST, Y)"
    if single:
        thr_g, how = single_axis_thresholds(Y, meta_sex)
        info.update({"gene_thresholds": {g: (None if np.isnan(t) else float(t)) for g, t in zip(m_cols, thr_g)}, "threshold_rule": how})
        valid = ~np.isnan(thr_g)
        votes = (Y[:, valid] > thr_g[valid]).sum(axis=1)
        n_genes = int(valid.sum())
    else:
        xist = ds.X[:, gi["XIST"]]
        thr_x, lo_x, hi_x = split_axis(xist)
        thr_y, lo_y, hi_y = split_axis(ymean)
        info.update({"xist_threshold": thr_x, "xist_centres": (lo_x, hi_x), "y_threshold": thr_y, "y_centres": (lo_y, hi_y)})
        low_sd = float(np.std(ymean[ymean <= thr_y])) if thr_y is not None else float("nan")
    rows = []
    for i, sid in enumerate(ds.sample_ids):
        ms = meta_sex.iloc[i] if i < len(meta_sex) else None
        ms = ms if ms in ("female", "male") else None
        if single:
            cat = classify_single(int(votes[i]), n_genes)
            ver = verdict_single(cat, ms)
            extra = {"XIST": float("nan"), "votes_male": int(votes[i]), "n_genes_voting": n_genes, "y_intermediate": cat == "intermediate(Y)"}
        else:
            xh = None if thr_x is None else bool(xist[i] > thr_x)
            yh = None if thr_y is None else bool(ymean[i] > thr_y)
            cat, ver = classify(xh, yh), verdict(classify(xh, yh), ms)
            extra = {"XIST": float(xist[i]), "votes_male": None, "n_genes_voting": None,
                     "y_intermediate": bool(thr_y is not None and (not yh) and ymean[i] > lo_y + max(3 * low_sd, 0.25 * (hi_y - lo_y)))}
        rows.append({"sample_id": sid, "label": int(ds.y[i]), "y_mean": float(ymean[i]), **{g: float(ds.X[i, gi[g]]) for g in m_cols},
                     **extra, "category": cat, "meta_sex": ms, "verdict": ver,
                     "action": {"opposite": "quarantine", "inconsistent": "quarantine"}.get(ver, "keep")})
    return pd.DataFrame(rows), info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--out", type=Path, default=None, help="mặc định $KLTN_DATA_ROOT/qc")
    ap.add_argument("--show-all", action="store_true", help="in mọi mẫu, không chỉ mẫu bị cờ")
    args = ap.parse_args(argv)
    out = args.out or (DATA_ROOT / "qc")
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in args.datasets:
        ds = load_dataset(DATA_ROOT, name, gene_space=None)
        meta = pd.read_csv(ds.data_dir / "metadata.csv")
        meta = meta.set_index("sample_id") if "sample_id" in meta.columns else pd.read_csv(ds.data_dir / "metadata.csv", index_col=0)
        meta.index = meta.index.astype(str)
        table, info = sex_table(ds, meta)
        if table.empty:
            print(f"[{name}] bỏ qua: {info.get('error')}", flush=True)
            continue
        table.to_csv(out / f"sex_check_{name}.csv", index=False)
        counts = table.verdict.value_counts().to_dict()
        summary.append({"dataset": name, "n": len(table), "mode": info["mode"], "metadata_sex_column": info["metadata_sex_column"],
                        "xist_threshold": info.get("xist_threshold"), "y_threshold": info.get("y_threshold"),
                        "gene_thresholds": info.get("gene_thresholds"), "threshold_rule": info.get("threshold_rule"), **counts,
                        "n_intermediate": int(table.y_intermediate.sum()), "n_quarantine": int((table.action == "quarantine").sum())})
        if info["mode"].startswith("single"):
            print(f"[{name}] n = {len(table)}; {info['mode']}; ngưỡng từng gen ({info['threshold_rule']}): "
                  f"{ {g: (round(t, 2) if t is not None else None) for g, t in info['gene_thresholds'].items()} }; "
                  f"cột giới tính: {info['metadata_sex_column']}; verdict: {counts}", flush=True)
        else:
            print(f"[{name}] n = {len(table)}; {info['mode']}; ngưỡng XIST = {info.get('xist_threshold')} (tâm {info.get('xist_centres')}), "
                  f"ngưỡng Y = {info['y_threshold']} (tâm {info['y_centres']}); cột giới tính: {info['metadata_sex_column']}; "
                  f"verdict: {counts}; trung gian trên trục Y: {int(table.y_intermediate.sum())}", flush=True)
        flagged = table if args.show_all else table[table.action.eq("quarantine") | table.verdict.eq("inconclusive") | table.y_intermediate]
        if len(flagged):
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                print(flagged.round(2).to_string(index=False), flush=True)
    pd.DataFrame(summary).to_csv(out / "sex_check_summary.csv", index=False)
    print("wrote:", out, flush=True)


if __name__ == "__main__":
    main()
