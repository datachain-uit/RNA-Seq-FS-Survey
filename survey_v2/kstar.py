"""k* gộp qua cohort theo QĐ8 + QĐ11 — đoạn code G của Review 24/10/2026, đưa vào gói để make_survey_tables dùng.

Kiểm định theo thứ tự cố định trên k (KSTAR_SEQUENCE: 50 → 25 → 10 → 5 → 4 → 3 → 2 → 1); một mức k chỉ "đạt" khi cận
dưới một phía 95 % của ΔAUROC(panel − B0) gộp qua cohort > −Δ (Δ = 0,02) theo CẢ bản IVW (phương sai Nadeau–Bengio)
lẫn bản trọng số bằng nhau; dừng ở lần đầu không đạt (Hsu & Berger 1999). k*_both = k cuối cùng đạt theo cả hai;
k*_ivw = theo IVW. Trong survey, cùng cách tính cho MỌI phương pháp (không chỉ L1-LR), kèm trọng số từng cohort và I².

Đầu vào: bảng dài theo fold (cohort, repeat, fold, method, k, auroc); B0 là method == "b0" (k bỏ trống hoặc bất kỳ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

IDX = ["cohort", "repeat", "fold"]


def per_cohort(df: pd.DataFrame, method: str, k: int, ref: str = "b0", ratio: float = C.NB_TEST_TRAIN_RATIO,
               alpha: float = C.ALPHA) -> pd.DataFrame:
    a = df[(df["method"] == method) & (df["k"] == k)].set_index(IDX)["auroc"]
    b = df[df["method"] == ref].groupby(IDX)["auroc"].first()
    d = (a - b).dropna().rename("diff").reset_index()
    rows = []
    for c, g in d.groupby("cohort"):
        J, m, s = len(g), float(g["diff"].mean()), float(g["diff"].std(ddof=1)) if len(g) > 1 else np.nan
        var = (1.0 / J + ratio) * s**2 if J > 1 else np.nan
        lb = m - stats.t.ppf(1 - alpha, J - 1) * np.sqrt(var) if J > 1 else np.nan
        rows.append({"cohort": c, "J": J, "diff": m, "sd": s, "var": var, "lb": lb})
    return pd.DataFrame(rows, columns=["cohort", "J", "diff", "sd", "var", "lb"])


def pooled(pc: pd.DataFrame, alpha: float = C.ALPHA) -> dict:
    z = stats.norm.ppf(1 - alpha)
    pc = pc.dropna(subset=["var"])
    pc = pc[pc["var"] > 0]
    if pc.empty:
        return {"diff_ivw": np.nan, "lb_ivw": np.nan, "diff_eq": np.nan, "se_eq": np.nan, "lb_eq": np.nan, "I2": np.nan}
    w = 1.0 / pc["var"]
    d_ivw = float((w * pc["diff"]).sum() / w.sum())
    se_ivw = float(np.sqrt(1.0 / w.sum()))
    q = float((w * (pc["diff"] - d_ivw) ** 2).sum())
    d_eq = float(pc["diff"].mean())
    se_eq = float(np.sqrt(pc["var"].sum()) / len(pc))
    weights = {f"w_{c}": float(x) for c, x in zip(pc["cohort"], w / w.sum())}
    return {"diff_ivw": d_ivw, "lb_ivw": d_ivw - z * se_ivw, "diff_eq": d_eq, "se_eq": se_eq, "lb_eq": d_eq - z * se_eq,
            "I2": max(0.0, (q - (len(pc) - 1)) / q) if q > 0 else 0.0, "max_weight": float((w / w.sum()).max()), **weights}


def k_star(df: pd.DataFrame, method: str, cohorts, k_sequence=C.KSTAR_SEQUENCE, delta: float = C.KSTAR_DELTA,
           require_all_cohorts: bool = True):
    """Bảng theo k và hai giá trị k*: theo IVW và theo cả hai cách gộp (None nếu không đạt ở k đầu tiên)."""
    sub = df[df["cohort"].isin(cohorts)]
    rows, k_ivw, k_both = [], None, None
    alive_ivw = alive_both = True
    # chuỗi cố định trên các mức k mà phương pháp có kết quả (wrapper chỉ ở WRAPPER_K): vẫn là thứ tự đặt trước, từ lớn tới nhỏ
    tested = set(sub.loc[sub["method"] == method, "k"].dropna().astype(int))
    k_sequence = [k for k in k_sequence if k in tested]
    for k in k_sequence:
        pc = per_cohort(sub, method, k)
        if set(pc["cohort"]) != set(cohorts):
            if require_all_cohorts:
                rows.append({"method": method, "k": k, "missing_cohorts": sorted(set(cohorts) - set(pc["cohort"]))})
                alive_ivw = alive_both = False
                continue
        p = pooled(pc)
        ok_ivw = bool(p["lb_ivw"] > -delta) if not np.isnan(p["lb_ivw"]) else False
        ok_eq = bool(p["lb_eq"] > -delta) if not np.isnan(p["lb_eq"]) else False
        alive_ivw = alive_ivw and ok_ivw
        alive_both = alive_both and ok_ivw and ok_eq
        k_ivw = k if alive_ivw else k_ivw
        k_both = k if alive_both else k_both
        lbs = {f"lb_{c}": v for c, v in zip(pc["cohort"], pc["lb"])}
        rows.append({"method": method, "k": k, **p, "pass_ivw": ok_ivw, "pass_eq": ok_eq,
                     "alive_ivw": alive_ivw, "alive_both": alive_both, **lbs})
    return pd.DataFrame(rows), k_ivw, k_both


def verdict_h1(k_ivw, k_both, k_sequence_used=()) -> str:
    """Quy tắc QĐ11 (đặt cho H1 của L1-LR; với phương pháp khác chỉ là mô tả). Panel cố định hoặc lưới k thiếu mức 25
    không áp dụng được quy tắc 'bác bỏ'."""
    if k_both is not None and k_both <= 10:
        return "ủng hộ (k*_gộp <= 10 theo cả hai cách gộp)"
    if k_ivw is not None and k_ivw <= 10:
        return "ủng hộ một phần (chỉ bản IVW đạt; ghi rõ cohort chi phối)"
    if k_ivw is None:
        return "bác bỏ (không đạt ở k lớn nhất)" if 25 in set(k_sequence_used) else "không xác định (không đạt ở các k đã thử)"
    if k_ivw > 25:
        return "bác bỏ (chuỗi không tới được k = 25)"
    return "chưa kết luận"


def kstar_table(df: pd.DataFrame, methods, cohorts=C.POOLED_KSTAR_COHORTS, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bảng theo k cho mọi phương pháp, bảng tóm tắt k* theo phương pháp)."""
    tabs, summ = [], []
    for m in methods:
        t, k_ivw, k_both = k_star(df, m, cohorts, **kw)
        tabs.append(t)
        seq_used = [int(k) for k in t["k"]] if len(t) else []
        summ.append({"method": m, "k_star_ivw": k_ivw, "k_star_both": k_both, "verdict_vs_b0": verdict_h1(k_ivw, k_both, seq_used),
                     "k_sequence_used": ">".join(str(int(k)) for k in t["k"]) if len(t) else "", "cohorts": ",".join(cohorts)})
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame(), pd.DataFrame(summ)


def per_cohort_kstar(df: pd.DataFrame, methods, cohorts, k_sequence=C.KSTAR_SEQUENCE, delta: float = C.KSTAR_DELTA) -> pd.DataFrame:
    """k* từng cohort (mô tả): cận dưới t(0,95; J−1) từng cohort, chuỗi cố định."""
    rows = []
    for c in cohorts:
        for m in methods:
            k_star_c, alive = None, True
            tested = set(df.loc[(df["cohort"] == c) & (df["method"] == m), "k"].dropna().astype(int))
            for k in [k for k in k_sequence if k in tested]:
                pc = per_cohort(df[df["cohort"] == c], m, k)
                if pc.empty or np.isnan(pc["lb"].iloc[0]):
                    alive = False
                    continue
                ok = bool(pc["lb"].iloc[0] > -delta)
                alive = alive and ok
                k_star_c = k if alive else k_star_c
            rows.append({"cohort": c, "method": m, "k_star_cohort": k_star_c})
    return pd.DataFrame(rows)
