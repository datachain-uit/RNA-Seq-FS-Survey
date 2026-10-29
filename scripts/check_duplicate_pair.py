"""Kiểm tra một cặp mẫu nghi trùng bằng gen giới tính và tương quan (Y22, Review 29/10/2026).

Một cặp có r ≥ 0,99 nhưng metadata ghi hai bệnh nhân khác giới (vd. GSM1019246 / GSM1019247 của GSE37745, r = 0,992)
có thể là (a) hai bệnh nhân thật, (b) một mẫu bị nạp hai lần. Gen giới tính tách hai trường hợp này: XIST cao ở nữ;
RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY cao ở nam. Nếu hai mẫu "khác giới" theo metadata mà biểu hiện gen giới tính giống hệt
nhau (cùng mức XIST, cùng mức gen Y) thì metadata không đáng tin cho cặp này: cách ly cả hai khỏi cohort (ghi vào
excluded_samples_<cohort>.csv) và báo GVHD. Không chạy lại gì khác.

    python scripts/check_duplicate_pair.py --dataset gse37745 --samples GSM1019246 GSM1019247
    python scripts/check_duplicate_pair.py --dataset gse37745 --threshold 0.98      # liệt kê mọi cặp r >= 0,98
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2.data import load_dataset  # noqa: E402
from survey_v2.runlog import DATA_ROOT  # noqa: E402

FEMALE = ("XIST",)
MALE = ("RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY")


def sex_table(ds, samples) -> pd.DataFrame:
    idx = {s: i for i, s in enumerate(ds.sample_ids)}
    genes = [g for g in FEMALE + MALE if g in set(ds.genes)]
    cols = [int(np.flatnonzero(ds.genes == g)[0]) for g in genes]
    rows = []
    for s in samples:
        if s not in idx:
            raise SystemExit(f"{s} không có trong {ds.name}")
        x = ds.X[idx[s], cols]
        rows.append({"sample_id": s, "label": int(ds.y[idx[s]]), **{g: float(v) for g, v in zip(genes, x)}})
    t = pd.DataFrame(rows).set_index("sample_id")
    # Suy đoán giới tính theo biểu hiện: điểm = mean(gen Y) − XIST (đơn vị log2); dương rõ -> nam, âm rõ -> nữ
    male_cols = [g for g in MALE if g in t.columns]
    if male_cols and "XIST" in t.columns:
        t["sex_score"] = t[male_cols].mean(axis=1) - t["XIST"]
        t["sex_inferred"] = np.where(t["sex_score"] > 0, "male", "female")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gse37745")
    ap.add_argument("--samples", nargs="+", default=None, help="các mẫu cần xem (>= 2)")
    ap.add_argument("--threshold", type=float, default=None, help="liệt kê mọi cặp có Pearson r >= ngưỡng (mặc định: bỏ qua)")
    args = ap.parse_args(argv)
    ds = load_dataset(DATA_ROOT, args.dataset, gene_space=None)
    meta = pd.read_csv(ds.data_dir / "metadata.csv")
    if "sample_id" in meta.columns:
        meta = meta.set_index("sample_id")
    else:
        meta = pd.read_csv(ds.data_dir / "metadata.csv", index_col=0)
    sex_cols = [c for c in meta.columns if "sex" in c.lower() or "gender" in c.lower()]
    print(f"{ds.name}: n = {ds.n}, p = {ds.p}; cột giới tính trong metadata: {sex_cols or 'không có'}")
    if args.threshold is not None:
        Xc = ds.X - ds.X.mean(axis=1, keepdims=True)
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
        R = Xc @ Xc.T
        iu = np.triu_indices(ds.n, 1)
        hits = [(ds.sample_ids[i], ds.sample_ids[j], float(R[i, j])) for i, j in zip(*iu) if R[i, j] >= args.threshold]
        print(f"cặp có r >= {args.threshold}: {len(hits)}")
        for a, b, r in sorted(hits, key=lambda t: -t[2]):
            print(f"  {a}  {b}  r = {r:.4f}")
        if args.samples is None:
            args.samples = sorted({s for a, b, _ in hits for s in (a, b)})
    if not args.samples:
        return
    t = sex_table(ds, args.samples)
    for c in sex_cols:
        t[f"meta_{c}"] = [meta.loc[s, c] if s in meta.index else None for s in t.index]
    if len(args.samples) == 2:
        a, b = [int(np.flatnonzero(ds.sample_ids == s)[0]) for s in args.samples]
        r = float(np.corrcoef(ds.X[a], ds.X[b])[0, 1])
        print(f"Pearson r({args.samples[0]}, {args.samples[1]}) = {r:.4f}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.round(3).to_string())
    if "sex_inferred" in t.columns and sex_cols:
        for s in t.index:
            meta_sex = str(t.loc[s, f"meta_{sex_cols[0]}"]).strip().lower()
            inferred = t.loc[s, "sex_inferred"]
            flag = "" if (meta_sex[:1] == inferred[:1]) else "   <-- KHÁC metadata"
            print(f"{s}: metadata = {meta_sex}, biểu hiện = {inferred}{flag}")


if __name__ == "__main__":
    main()
