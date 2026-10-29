"""Panel tri thức và panel công bố (tham chiếu cố định, KBTN v2 Mục 5.1; QĐ2, QĐ12).

* IHC-2 = {NKX2-1, TP63}; IHC-4 = IHC-2 ∪ {NAPSA, KRT5} (mức mRNA của dấu ấn hoá mô miễn dịch; Yatabe et al. 2019).
* Panel công bố (Girard et al. 2016 phần AC vs SCC; Wilkerson et al. 2013) đọc từ ``survey_v2/panels/published_panels.csv``
  (cột ``panel, gene, source``). Tệp mẫu chỉ có IHC; **đội code điền danh sách gen từ bài gốc** và ghi DOI vào cột source.
  Gói code không tự đặt danh sách gen cho panel công bố (nguyên tắc không bịa số liệu).
* Gen thiếu trên nền tảng được ghi lại (``missing``) và panel được đánh giá với các gen còn lại, kèm cờ ``complete``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import config as C

PANELS_CSV = Path(__file__).resolve().parent / C.PUBLISHED_PANELS_CSV


def builtin_panels() -> dict[str, list[str]]:
    return {"ihc_2": list(C.IHC_2), "ihc_4": list(C.IHC_4)}


def published_panels(path: Path = PANELS_CSV) -> dict[str, list[str]]:
    """Panel từ CSV; tên method = ``panel_<panel>``. Dòng bắt đầu bằng ``#`` bị bỏ qua."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#")) if r.get("gene")]
    for r in rows:
        name = r["panel"].strip()
        if name in ("ihc_2", "ihc_4"):
            continue                                   # IHC là built-in, không ghi đè
        key = name if name.startswith("panel_") else f"panel_{name}"
        out.setdefault(key, []).append(r["gene"].strip())
    return out


def all_fixed_panels(path: Path = PANELS_CSV) -> dict[str, list[str]]:
    d = builtin_panels()
    d.update(published_panels(path))
    return d


def panel_indices(genes: np.ndarray, panel: list[str]) -> tuple[np.ndarray, list[str]]:
    """Chỉ số cột của các gen trong panel có mặt trong `genes`; kèm danh sách gen thiếu (thứ tự panel được giữ)."""
    index = {g: j for j, g in enumerate(genes)}
    found = [index[g] for g in panel if g in index]
    missing = [g for g in panel if g not in index]
    return np.asarray(found, dtype=int), missing
