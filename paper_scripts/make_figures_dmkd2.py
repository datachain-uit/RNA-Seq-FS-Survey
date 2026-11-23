#!/usr/bin/env python3
"""
make_figures_dmkd2.py -- the figures of E1, E7 and E6 with lettering of at least 8 pt at a print width of 174 mm
(tag survey-v3-dmkd.2; QĐ178).  Earlier layout, superseded by make_figures_dmkd.py; the file names follow the
numbering of that tag.

* Figure_8.pdf  (E1, AUROC against k) and Figure_12.pdf (E6, AUROC / stability / cost at k = 10) are drawn by the
  SAME functions of make_figures.py from the SAME official tables (results/survey_v2/tables, commit e15fb7c):
  data and computation unchanged. Only the export step changes: the figure is set to the print width of the
  article (--print-width-mm, default 174 mm) and every text element (axis labels, tick labels, titles, legends,
  annotations) is raised to at least --min-pt (default 8 pt), then written as vector PDF.
* Figure_11.pdf is redrawn from the split-level labels of B1 (results/survey_v2/B1_tables/b1_e7_split_labels.csv,
  20 splits; QĐ175): cell = mean over splits of Δ = strategy − target-only (split value = mean of the 4 subsets),
  mark = label of the reading rule fixed on 12/11 (+ positive transfer, − negative transfer, ≈ practically
  equivalent, none = inconclusive). It replaces the descriptive 5-split flags of e7_transfer_table.csv.

Usage:
    python paper_scripts/make_figures_dmkd2.py --results results/survey_v2/tables \
        --b1 results/survey_v2/B1_tables --out out/Images_dmkd2

After saving, every PDF is checked: the smallest text size at print width must be >= --min-pt (exit code 1
otherwise); the sizes are recorded in figures_sources_dmkd2.json.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as mf  # noqa: E402  (sets the matplotlib style of the manuscript)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.text import Text  # noqa: E402

MM_PER_IN = 25.4
REPORT = {}
OPTS = {"width_in": 174 / MM_PER_IN, "min_pt": 8.0}


def _texts(fig):
    fig.canvas.draw()
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def _relegend(fig):
    """Legends placed below the panels: rebuild with fewer columns so that 8 pt labels fit the print width."""
    for lg in list(fig.legends):
        handles, labels = lg.legend_handles, [t.get_text() for t in lg.get_texts()]
        lg.remove()
        fig.legend(handles=handles, labels=labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
                   handlelength=2.2, columnspacing=1.0, fontsize=OPTS["min_pt"])


def save_print(fig, out, name, sources, tune=None):
    """Replacement of make_figures.save: print width, text >= min_pt, vector PDF, check."""
    w, h = fig.get_size_inches()
    W = OPTS["width_in"]
    fig.set_size_inches(W, h * W / w)
    if tune:
        tune(fig)
    for t in _texts(fig):
        if t.get_fontsize() < OPTS["min_pt"]:
            t.set_fontsize(OPTS["min_pt"])
    _relegend(fig)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    fname = mf.FILE.get(name, name)
    path = os.path.join(out, fname)
    meta = {"Author": mf.AUTHOR, "Title": mf.TITLES.get(name, name),
            "Subject": "Drawn from " + "; ".join(sources) + " (tag survey-v3-dmkd.2; text >= %g pt at %.0f mm)" % (OPTS["min_pt"], W * MM_PER_IN)}
    sizes = [t.get_fontsize() for t in _texts(fig)]
    fig.savefig(path, bbox_inches="tight", metadata=meta)
    plt.close(fig)
    REPORT[fname] = {"sources": sources, "min_text_pt": min(sizes), "n_text": len(sizes),
                     "print_width_mm": round(W * MM_PER_IN, 1)}
    print("wrote", path, "| smallest text %.1f pt" % min(sizes))


def _tune_fig8(fig):
    """Fewer tick labels on the log axis; cohort titles on two lines (name, platform) so that 8 pt titles do not touch."""
    for ax in fig.axes:
        if ax.get_xscale() == "log":
            ks = [1, 2, 5, 10, 50]
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
        t = ax.get_title(loc="left")
        if " (" in t:
            ax.set_title(t.replace(" (", "\n("), loc="left", fontweight="bold")


def _tune_fig12(fig):
    """At 8 pt the method names written next to the points overlap; identify the methods by marker and colour in the
    legend instead (same encoding as Figure 8). Points, axes and data unchanged."""
    from matplotlib.lines import Line2D
    from matplotlib.text import Annotation
    names = set()
    for ax in fig.axes:
        for t in list(ax.texts):
            if isinstance(t, Annotation):
                names.add(t.get_text())
                t.remove()
    inv = {v: k for k, v in mf.NAME.items()}
    methods = [m for m in mf.METHODS if mf.NAME[m] in names or inv.get(mf.NAME[m]) in names]
    old = fig.legends[0] if fig.legends else None
    handles = list(old.legend_handles) if old else []
    labels = [t.get_text() for t in old.get_texts()] if old else []
    if old:
        old.remove()
    for m in methods:
        fam, _, mk = mf.STYLE[m]
        handles.append(Line2D([0], [0], color=mf.HUE[fam], marker=mk, linestyle="none", markersize=6))
        labels.append(mf.NAME[m])
    fig.legend(handles=handles, labels=labels, loc="lower center")


def fig11_split_labels(b1_dir, out):
    lab = pd.read_csv(os.path.join(b1_dir, "b1_e7_split_labels.csv"))
    lab["n_t"] = lab.n_t.astype(str)
    strategies = [("pooled", "pooled"), ("select_on_source", "select-on-source"), ("source_only", "source-only")]
    ks = [2, 3, 4, 5, 10]
    nts = ["5", "10", "20"]
    mark = {"positive transfer": "+", "negative transfer": "−", "practically equivalent": "≈", "inconclusive": ""}
    fig, axes = plt.subplots(3, 4, figsize=(OPTS["width_in"], 6.0))
    norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.10)
    cmap = plt.get_cmap("RdBu")
    last = None
    for i, (s, stitle) in enumerate(strategies):
        for j, c in enumerate(mf.GEO):
            ax = axes[i, j]
            g = lab[(lab.cohort == c) & (lab.strategy == s)]
            M = np.full((len(nts), len(ks)), np.nan)
            L = [["" for _ in ks] for _ in nts]
            for _, r in g.iterrows():
                if r.n_t in nts and int(r.k) in ks:
                    a, b = nts.index(r.n_t), ks.index(int(r.k))
                    M[a, b] = r.mean_delta
                    L[a][b] = mark[r.label]
            last = ax.imshow(np.clip(M, -0.10, 0.10), cmap=cmap, norm=norm, aspect="auto")
            for a in range(len(nts)):
                for b in range(len(ks)):
                    v = M[a, b]
                    if math.isnan(v):
                        ax.text(b, a, "–", ha="center", va="center", color=mf.INK3)
                        continue
                    txt = ("%+d" % round(100 * v)).replace("-", "−")    # ΔAUROC × 100 (percentage points of AUROC)
                    ax.text(b, a, f"{txt}\n{L[a][b]}" if L[a][b] else txt, ha="center", va="center",
                            color="white" if abs(v) > 0.06 else mf.INK, linespacing=0.9)
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_yticks(range(len(nts)))
            ax.set_yticklabels(nts)
            ax.grid(False)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(mf.COHORT[c].split(" ")[0], loc="left", fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{stitle}\n$n_T$ per class")
            if i == 2:
                ax.set_xlabel("budget k")
    fig.tight_layout(h_pad=1.0, w_pad=0.6, rect=(0, 0.06, 0.9, 1))
    cax = fig.add_axes([0.915, 0.2, 0.018, 0.65])
    cb = fig.colorbar(last, cax=cax)
    cb.set_label("mean ΔAUROC over 20 splits\n(strategy − target-only; colour clipped at ±0.10)")
    fig.text(0.0, 0.0, "Cell: 100 × mean ΔAUROC. Mark: + positive transfer, − negative transfer, ≈ practically equivalent,\n"
             "none = inconclusive (two-sided exact Wilcoxon over 20 splits, Holm within strategy; margin ±0.01).", ha="left", va="bottom")
    sources = ["B1_tables/b1_e7_split_labels.csv (mean_delta, label; 20 splits)"]
    mf.TITLES["fig_e7_split_labels.pdf"] = "Fig. 11. E7 (20 splits): transfer strategies against target-only L1-LR, split-level labels"
    mf.FILE["fig_e7_split_labels.pdf"] = "Figure_11.pdf"
    W = OPTS["width_in"]
    for t in _texts(fig):
        if t.get_fontsize() < OPTS["min_pt"]:
            t.set_fontsize(OPTS["min_pt"])
    sizes = [t.get_fontsize() for t in _texts(fig)]
    path = os.path.join(out, "Figure_11.pdf")
    fig.savefig(path, bbox_inches="tight", metadata={"Author": mf.AUTHOR, "Title": mf.TITLES["fig_e7_split_labels.pdf"],
                                                     "Subject": "Drawn from " + "; ".join(sources) + " (tag survey-v3-dmkd.2)"})
    plt.close(fig)
    REPORT["Figure_11.pdf"] = {"sources": sources, "min_text_pt": min(sizes), "n_text": len(sizes), "print_width_mm": round(W * MM_PER_IN, 1)}
    print("wrote", path, "| smallest text %.1f pt" % min(sizes))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/survey_v2/tables (official tables, commit e15fb7c)")
    ap.add_argument("--b1", required=True, help="results/survey_v2/B1_tables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-width-mm", type=float, default=174.0)
    ap.add_argument("--min-pt", type=float, default=8.0)
    a = ap.parse_args(argv)
    OPTS.update(width_in=a.print_width_mm / MM_PER_IN, min_pt=a.min_pt)
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    tunes = {"fig_e1_auroc_vs_k.pdf": _tune_fig8, "fig_e6_tradeoff.pdf": _tune_fig12}
    mf.save = lambda fig, out, name, sources: save_print(fig, out, name, sources, tune=tunes.get(name))
    mf.fig_e1_auroc_vs_k(a.results, a.out)
    mf.fig_e6(a.results, a.out)
    fig11_split_labels(a.b1, a.out)
    with open(os.path.join(a.out, "figures_sources_dmkd2.json"), "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=1, ensure_ascii=False)
    bad = {k: v["min_text_pt"] for k, v in REPORT.items() if v["min_text_pt"] < a.min_pt - 1e-9}
    if bad:
        print("TEXT BELOW %g pt:" % a.min_pt, bad)
        return 1
    print("OK: every text >= %g pt at %.0f mm" % (a.min_pt, a.print_width_mm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
