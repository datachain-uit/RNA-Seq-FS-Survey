#!/usr/bin/env python3
"""
make_figures_dmkd.py -- result figures 5-12 of the DMKD manuscript, drawn for the text width of the
Springer template (sn-jnl: \\textwidth = 372 TeX pt = 370.61 PDF pt = 130.7 mm) from the official result tables
(commit e15fb7c, label [v03-11]).

    python3 make_figures_dmkd.py --results <tables_official_2026-11-03 | results/survey_v2/tables> \
                                 --e3 <E2_E3 dir: e3_scaling_n.csv, e3_scaling_p.csv, e3_wrapper_evals.csv> \
                                 --out Images/ [--b1 results/survey_v2/B1_tables] [--only 11]

--b1 (for the B1 version of Fig. 11, after the fix O49): Fig. 11 is drawn from B1_tables/b1_e7_split_labels.csv
(20 splits; cell = mean_delta, outline = label: solid for positive or negative transfer, dashed for practically
equivalent, none for inconclusive) instead of e7_transfer_table.csv (5 splits; outline = positive_transfer or
negative_transfer).  The layout, the lettering and the checks are the same.

Why a new script (issue O53, 18/11/2026): the figures of make_figures.py were drawn for a page of 175-190 mm and
are inserted at 131 mm, so their lettering prints at 3-5 pt; make_figures_dmkd2.py (tag survey-v3-dmkd.2) raises
the lettering to 8 pt at 174 mm, which again shrinks at 131 mm. Here every figure is laid out natively at the text
width and inserted with width=\\textwidth, that is at scale 1:
  * page width exactly 370.61 pt in PDF (PostScript) points = 372 TeX points (\\textwidth of sn-jnl; a page of
    372 PDF points would print at 0.9963), height <= 520 pt (text block of sn-jnl: 372 x 552.7 TeX pt);
  * main lettering 8 pt, tick and legend labels 7.5 pt, nothing below 7 pt (Springer: 8-12 pt at final size),
    sub- and superscripts included (mathtext shrinks them to 0.875 of the label: 7 pt for an 8 pt label);
    one sans-serif family (Liberation Sans, metric-compatible with Arial; DejaVu Sans only for missing glyphs);
  * data, scales and the colour of each family are those of make_figures.py; the panels are re-arranged where
    the old arrangement cannot carry 7-8 pt lettering (minor comment 24 of the pre-submission review):
    Fig. 8 has one column per family instead of 13 lines per panel; Fig. 11 prints 100 x Delta AUROC and marks
    a flagged cell by its outline instead of a sign glued to the number; Fig. 12 gives stability its own axis
    instead of marker area and identifies the methods by marker and colour instead of overlapping names.
After drawing, every figure is checked: smallest text >= 7 pt, no text outside the page, no two texts overlapping,
page width 370.61 pt, every glyph of the written PDF >= 7 pt (check_figure_lettering.py, needs pdfminer.six), and
the plotted values equal the values of the tables (re-read from the CSV files).
The script exits with code 1 if a check fails. Document information: Title, Author, Keywords, Creator (no Subject).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

PT = 1 / 72.0              # Matplotlib and PDF points (PostScript points)
W_IN = 372 / 72.27         # text width of sn-jnl: 372 TeX points = 370.61 PostScript points
W_PDF_PT = W_IN * 72
H_MAX_IN = 520 * PT
MIN_PT = 7.0
BASE, SMALL = 8.0, 7.5

# Sub- and superscripts of mathtext ($n_T$, $N_\mathrm{eval}$) print at SHRINK_FACTOR times the size of the label,
# 0.7 by default (5.6 pt for an 8 pt label).  0.875 prints them at 7 pt, the floor of the lettering (O53).
import matplotlib._mathtext as _mathtext  # noqa: E402
_mathtext.SHRINK_FACTOR = MIN_PT / BASE

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE, "legend.fontsize": SMALL,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.title_fontsize": BASE,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold", "mathtext.sf": "Liberation Sans", "mathtext.cal": "Liberation Sans",
    "mathtext.tt": "DejaVu Sans Mono",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e3e2dd", "grid.linewidth": 0.5, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 3.2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.titlepad": 3.0, "axes.labelpad": 2.0,
    "legend.handlelength": 2.3, "legend.columnspacing": 1.0, "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3, "legend.frameon": False,
    "figure.dpi": 72, "savefig.dpi": 600,
})

HUE = {"Filter": "#2a78d6", "Embedded": "#eb6834", "Constrained": "#1baf7a", "Wrapper": "#6250d6"}
FAMILIES = ["Filter", "Embedded", "Constrained", "Wrapper"]
STYLE = {  # (family, linestyle, marker) -- as in make_figures.py
    "f_test": ("Filter", "-", "o"), "mutual_info": ("Filter", "--", "s"), "mrmr": ("Filter", "-.", "^"),
    "relieff": ("Filter", ":", "D"),
    "l1_lr": ("Embedded", "-", "o"), "elastic_net": ("Embedded", "--", "s"), "rf_importance": ("Embedded", "-.", "^"),
    "svm_rfe": ("Embedded", ":", "D"),
    "abess": ("Constrained", "-", "o"), "stability_selection": ("Constrained", "--", "s"), "ktsp": ("Constrained", "-.", "^"),
    "ga": ("Wrapper", "-", "o"), "bpso": ("Wrapper", "--", "s"),
}
NAME = {
    "f_test": "F-test", "mutual_info": "MI", "mrmr": "mRMR", "relieff": "ReliefF",
    "l1_lr": "L1-LR", "elastic_net": "Elastic Net", "rf_importance": "RF importance", "svm_rfe": "SVM-RFE",
    "abess": "abess (L0)", "stability_selection": "Stability selection", "ktsp": "k-TSP",
    "ga": "GA", "bpso": "BPSO", "b0": "B0 (all genes)", "random_k": "random-k",
    "ihc_2": "IHC-2", "ihc_4": "IHC-4", "panel_girard2016": "Girard 2016", "panel_wilkerson2013": "Wilkerson 2013",
}
COHORT = {"tcga_luad_lusc_primary_tumor": "TCGA", "gse10245": "GSE10245", "gse30219": "GSE30219",
          "gse37745": "GSE37745", "gse42127": "GSE42127"}
COHORTS = ["tcga_luad_lusc_primary_tumor", "gse10245", "gse30219", "gse37745", "gse42127"]
GEO = ["gse10245", "gse30219", "gse37745", "gse42127"]
METHODS = list(STYLE.keys())
NONWRAP = [m for m in METHODS if STYLE[m][0] != "Wrapper"]
INK = "#0b0b0b"; INK2 = "#52514e"; INK3 = "#9a9893"
AUTHOR = "Thu Nguyen"
CREATOR = "Matplotlib %s, Scripts/make_figures_dmkd.py" % matplotlib.__version__

REPORT = {}
FAILED = []


def rd(d, f):
    return pd.read_csv(os.path.join(d, f), encoding="utf-8-sig")


def mline(m, **kw):
    fam, ls, mk = STYLE[m]
    return Line2D([0], [0], color=HUE[fam], linestyle=ls, marker=mk, label=NAME[m], **kw)


def ref_handles(*which):
    h = {"b0": Line2D([0], [0], color=INK, linestyle="--", label="B0 (all genes), band = ±SD"),
         "random": Line2D([0], [0], color=INK2, linestyle=":", marker="x", label="random-k, band = ±SD"),
         "random_plain": Line2D([0], [0], color=INK2, linestyle=":", marker="x", label="random-k"),
         "ihc2": Line2D([0], [0], color=INK, marker="*", linestyle="none", markersize=6.5, label="IHC-2 (k = 2)"),
         "ihc4": Line2D([0], [0], color=INK, marker="P", linestyle="none", markersize=5.5, label="IHC-4 (k = 4)")}
    return [h[w] for w in which]


def grouped(*groups):
    """Pad the groups to one length so that a column-major legend shows one group per column."""
    n = max(len(g) for g in groups)
    out = []
    for g in groups:
        out += list(g) + [Line2D([], [], linestyle="none", label=" ") for _ in range(n - len(g))]
    return out


def family_members(fam, methods=None):
    return [m for m in (methods or METHODS) if STYLE[m][0] == fam]


def new_fig(h_in):
    fig = plt.figure(figsize=(W_IN, h_in), layout="constrained")
    fig.get_layout_engine().set(w_pad=3 * PT, h_pad=3 * PT)
    return fig


def logk_axis(ax, ks, labels=None):
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in (labels or ks)])
    ax.minorticks_off()


def tag(line, key):
    """Remember what a plotted line shows, for the check against the tables."""
    line.set_gid(json.dumps(key))
    return line


# ------------------------------------------------------------------ checks
def _drawn_texts(fig):
    out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for axis in ((ax.xaxis, ax.yaxis) if ax.axison else ()):
            if not axis.get_visible():
                continue
            for tick in axis._update_ticks():
                for lab in (tick.label1, tick.label2):
                    if lab.get_visible():
                        out.append(lab)
            out.append(axis.label)
            out.append(axis.get_offset_text())
        out += [ax.title, ax._left_title, ax._right_title] + list(ax.texts)
        lg = ax.get_legend()
        if lg is not None:
            out += list(lg.get_texts()) + [lg.get_title()]
    for lg in fig.legends:
        out += list(lg.get_texts()) + [lg.get_title()]
    out += list(fig.texts)
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def check_layout(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    s = 72.0 / fig.dpi                                   # display units -> points
    fb = fig.bbox
    texts = _drawn_texts(fig)
    probs = []
    boxes = []
    for t in texts:
        if t.get_fontsize() < MIN_PT - 1e-9:
            probs.append("text below %g pt: %r (%.2f pt)" % (MIN_PT, t.get_text(), t.get_fontsize()))
        bb = t.get_window_extent(r)
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            probs.append("text outside the page: %r" % t.get_text())
        boxes.append((t, bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            w = (min(a.x1, b.x1) - max(a.x0, b.x0)) * s
            h = (min(a.y1, b.y1) - max(a.y0, b.y0)) * s
            if w > 0.4 and h > 0.4:
                probs.append("texts overlap: %r / %r (%.1f x %.1f pt)" % (boxes[i][0].get_text(), boxes[j][0].get_text(), w, h))
    sizes = sorted({round(t.get_fontsize(), 2) for t in texts})
    return probs, sizes, len(texts)


def save(fig, out, fname, title, keywords, sources, data_check):
    probs, sizes, n = check_layout(fig, fname)
    probs += data_check(fig)
    w, h = fig.get_size_inches()
    if h > H_MAX_IN + 1e-9:
        probs.append("page higher than %.0f pt: %.1f pt" % (H_MAX_IN * 72, h * 72))
    path = os.path.join(out, fname)
    fig.savefig(path, metadata={"Title": title, "Author": AUTHOR, "Keywords": keywords, "Creator": CREATOR})
    plt.close(fig)
    try:
        import pikepdf
        with pikepdf.open(path) as pdf:
            mb = [float(x) for x in pdf.pages[0].MediaBox]
        pw, ph = mb[2] - mb[0], mb[3] - mb[1]
        if abs(pw - W_PDF_PT) > 0.05:
            probs.append("page width %.2f pt, expected %.2f pt" % (pw, W_PDF_PT))
    except ImportError:
        pw, ph = w * 72, h * 72
    glyphs = None
    try:                                                   # every glyph of the PDF, sub- and superscripts included
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from check_figure_lettering import check_pdf
        res = check_pdf(path, MIN_PT)
        g = res.get("(page text)") or next(iter(res.values()))
        glyphs = {"glyphs": g["glyphs"], "min_pt": g["min_pt"], "median_pt": g["median_pt"],
                  "below_min": g["below_min"]}
        for it in g["small"]:
            probs.append("glyph below %g pt in the PDF: %r (%.2f pt) in %r" % (MIN_PT, it["text"], it["size"],
                                                                             it["context"]))
    except ImportError:
        print("      (glyph sizes of the PDF not checked: pdfminer.six missing)")
    REPORT[fname] = {"title": title, "sources": sources, "page_pt": [round(pw, 2), round(ph, 2)],
                     "font_sizes_pt": sizes, "pdf_glyphs": glyphs, "n_texts": n, "problems": probs}
    status = "OK" if not probs else "FAIL"
    print("%-4s %s  %.0f x %.0f pt, fonts %s pt, %d texts" % (status, path, pw, ph, sizes, n))
    for p in probs:
        print("     ", p)
    if probs:
        FAILED.append(fname)


def lines_check(fig, lookup):
    """Every tagged line must carry the values of the table (lookup(key) -> expected y array)."""
    probs = []
    n = 0
    for ax in fig.axes:
        for ln in ax.get_lines():
            gid = ln.get_gid()
            if not gid:
                continue
            key = json.loads(gid)
            exp = np.asarray(lookup(key), dtype=float)
            got = np.asarray(ln.get_ydata(), dtype=float)
            n += 1
            if exp.shape != got.shape or not np.allclose(exp, got, equal_nan=True, rtol=0, atol=1e-12):
                probs.append("plotted values differ from the table: %s" % gid)
    if n == 0:
        probs.append("no tagged line")
    return probs


# ------------------------------------------------------------------ Fig. 5 (E3)
def fig5(e3dir, out):
    n = pd.read_csv(os.path.join(e3dir, "e3_scaling_n.csv"), encoding="utf-8-sig")
    p = pd.read_csv(os.path.join(e3dir, "e3_scaling_p.csv"), encoding="utf-8-sig")
    w = pd.read_csv(os.path.join(e3dir, "e3_wrapper_evals.csv"), encoding="utf-8-sig")
    fig = new_fig(4.55)
    gs = fig.add_gridspec(2, 2)
    axn, axp, axw = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    axl = fig.add_subplot(gs[1, 1]); axl.axis("off")
    for ax, df, col, title, xlabel, ticks, labels in [
            (axn, n, "n", "(a) Training patients n  (p = 5,000)", "training patients n", [100, 200, 400, 812], None),
            (axp, p, "p", "(b) Genes p  (n = 812)", "genes after the pre-filter p", [1000, 2000, 5000, 10000, 19938],
             ["1k", "2k", "5k", "10k", "19.9k"])]:
        for m in NONWRAP:
            g = df[df.method == m].sort_values(col)
            if g.empty:
                continue
            fam, ls, mk = STYLE[m]
            tag(ax.plot(g[col].values, g.wall_s.values, color=HUE[fam], linestyle=ls, marker=mk)[0], [col, m])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
        ax.set_xticks(ticks); ax.set_xticklabels(labels or [f"{t:,}" for t in ticks])
        ax.set_yticks([0.01, 1, 100]); ax.set_yticklabels(["0.01", "1", "100"])
        ax.set_ylim(0.005, 4000)
        ax.set_xlabel(xlabel); ax.set_title(title, loc="left", fontweight="bold")
    axn.set_ylabel("wall time of selection (s)")
    for m in ["ga", "bpso"]:
        g = w[w.method == m].sort_values("n_eval_nominal")
        fam, ls, mk = STYLE[m]
        tag(axw.plot(g.n_eval_nominal.values, g.wall_s.values, color=HUE[fam], linestyle=ls, marker=mk, label=NAME[m])[0],
            ["eval", m])
    for (x, y, lab, dy) in [(200, 1.683197, "10 × 20", -11), (1500, 17.051599, "30 × 50", 6)]:
        axw.annotate(lab, (x, y), textcoords="offset points", xytext=(4 if x == 200 else -4, dy),
                     ha="left" if x == 200 else "right", fontsize=SMALL, color=INK2)
    axw.set_xscale("log"); axw.set_yscale("log"); axw.minorticks_off()
    axw.set_xticks([200, 1500]); axw.set_xticklabels(["200", "1,500"]); axw.set_xlim(140, 2200)
    axw.set_yticks([1, 10, 100]); axw.set_yticklabels(["1", "10", "100"]); axw.set_ylim(0.8, 60)
    axw.set_xlabel("fitness evaluations $N_\\mathrm{eval}$ = P × T"); axw.set_ylabel("wall time of selection (s)")
    axw.set_title("(c) Wrappers: evaluations", loc="left", fontweight="bold")
    axw.legend(loc="upper left", handlelength=2.0)
    axl.legend(handles=[mline(m) for m in NONWRAP], loc="center left", ncol=2, title="Selectors in (a), (b)",
               alignment="left")

    def lookup(key):
        col, m = key
        if col == "eval":
            return w[w.method == m].sort_values("n_eval_nominal").wall_s.values
        df = n if col == "n" else p
        return df[df.method == m].sort_values(col).wall_s.values

    def data_check(f):
        probs = lines_check(f, lookup)
        # the two pop x generations labels must match the table
        for _, r in w.iterrows():
            lab = "%d × %d" % (r["pop"], r["generations"])
            if lab not in ("10 × 20", "30 × 50") or abs(r.n_eval_nominal - r["pop"] * r["generations"]) > 0:
                probs.append("wrapper label mismatch: %s" % lab)
        if abs(w[(w.method == "ga") & (w.n_eval_nominal == 200)].wall_s.iloc[0] - 1.683197) > 1e-6 or \
           abs(w[(w.method == "ga") & (w.n_eval_nominal == 1500)].wall_s.iloc[0] - 17.051599) > 1e-6:
            probs.append("annotation anchor differs from e3_wrapper_evals.csv")
        return probs

    save(fig, out, "Figure_5.pdf", "Fig. 5. E3: wall time of the selection step against n, p and N_eval",
         "E3; computational cost; scaling", ["E2_E3/e3_scaling_n.csv", "E2_E3/e3_scaling_p.csv", "E2_E3/e3_wrapper_evals.csv"],
         data_check)


# ------------------------------------------------------------------ Fig. 6 (E4)
def fig6(results, out):
    s = rd(results, "e4_stability.csv")
    ks = [1, 2, 3, 4, 5, 10, 25, 50]
    fs = [0.5, 0.7, 0.9]
    cs = ["tcga_luad_lusc_primary_tumor", "gse37745"]
    fig = new_fig(5.15)
    gs = fig.add_gridspec(2, 2)
    for j, c in enumerate(cs):
        ax = fig.add_subplot(gs[0, j])
        g = s[(s.cohort == c) & (s.fraction == 0.7)]
        for m in METHODS + ["random_k"]:
            gm = g[g.method == m].set_index("k").reindex(ks)
            if gm.nogueira.notna().sum() == 0:
                continue
            if m == "random_k":
                tag(ax.plot(ks, gm.nogueira.values, color=INK2, linestyle=":", marker="x")[0], ["k", c, m])
                continue
            fam, ls, mk = STYLE[m]
            eb = ax.errorbar(ks, gm.nogueira.values, yerr=[gm.nogueira - gm.nogueira_ci_low, gm.nogueira_ci_high - gm.nogueira],
                             color=HUE[fam], linestyle=ls, marker=mk, markersize=2.8, elinewidth=0.5, capsize=1.2)
            tag(eb.lines[0], ["k", c, m])
        logk_axis(ax, ks)
        ax.set_ylim(-0.05, 1.02)
        ax.set_title(f"{COHORT[c]}, f = 0.7", loc="left", fontweight="bold")
        ax.set_xlabel("budget k (genes)")
        if j == 0:
            ax.set_ylabel("Nogueira index $\\hat{\\Phi}$ [95% CI]")
        ax = fig.add_subplot(gs[1, j])
        for m in METHODS + ["random_k"]:
            vals = []
            for f in fs:
                gm = s[(s.cohort == c) & (s.fraction == f) & (s.method == m) & (s.k == 10)]
                vals.append(float(gm.nogueira.iloc[0]) if len(gm) else np.nan)
            if all(math.isnan(v) for v in vals):
                continue
            if m == "random_k":
                tag(ax.plot(fs, vals, color=INK2, linestyle=":", marker="x")[0], ["f", c, m])
                continue
            fam, ls, mk = STYLE[m]
            tag(ax.plot(fs, vals, color=HUE[fam], linestyle=ls, marker=mk)[0], ["f", c, m])
        ax.set_xticks(fs); ax.set_xlim(0.45, 0.95); ax.set_ylim(-0.05, 1.02)
        ax.set_title(f"{COHORT[c]}, k = 10", loc="left", fontweight="bold")
        ax.set_xlabel("subsample fraction f")
        if j == 0:
            ax.set_ylabel("Nogueira index $\\hat{\\Phi}$")
    fig.legend(handles=grouped(*[[mline(m) for m in family_members(f)] for f in FAMILIES[:3]],
                               [mline(m) for m in family_members("Wrapper")] + ref_handles("random_plain")),
               loc="outside lower center", ncol=4)

    def lookup(key):
        kind, c, m = key
        if kind == "k":
            return s[(s.cohort == c) & (s.fraction == 0.7) & (s.method == m)].set_index("k").reindex(ks).nogueira.values
        return [float(s[(s.cohort == c) & (s.fraction == f) & (s.method == m) & (s.k == 10)].nogueira.iloc[0])
                if len(s[(s.cohort == c) & (s.fraction == f) & (s.method == m) & (s.k == 10)]) else np.nan for f in fs]

    save(fig, out, "Figure_6.pdf", "Fig. 6. E4: Nogueira index on TCGA and GSE37745", "E4; stability; Nogueira index",
         ["e4_stability.csv"], lambda f: lines_check(f, lookup))


# ------------------------------------------------------------------ Fig. 7 (E5)
def fig7(results, out):
    a = rd(results, "e5a_redundancy.csv")
    o = rd(results, "e5c_overlap_k10_gse37745.csv").set_index("method")
    fig = new_fig(6.45)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.15], width_ratios=[1.15, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    axl = fig.add_subplot(gs[0, 1]); axl.axis("off")
    ax.axhline(0, color=INK3, linewidth=0.6)
    rhos = [0.5, 0.8, 0.95]
    for m in NONWRAP + ["random_k"]:
        g = a[(a.method == m) & (a.k == 10)].sort_values("rho")
        if g.empty:
            continue
        if m == "random_k":
            tag(ax.plot(g.rho.values, g.delta_nogueira_vs_e4.values, color=INK2, linestyle=":", marker="x")[0], [m])
            continue
        fam, ls, mk = STYLE[m]
        tag(ax.plot(g.rho.values, g.delta_nogueira_vs_e4.values, color=HUE[fam], linestyle=ls, marker=mk)[0], [m])
    ax.set_xticks(rhos); ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("correlation ρ of the injected copies")
    ax.set_ylabel("Δ$\\hat{\\Phi}$ (raw − E4)")
    ax.set_title("(a) Redundancy (E5a)", loc="left", fontweight="bold")
    axl.legend(handles=[mline(m) for m in NONWRAP] + ref_handles("random_plain"), loc="center left", ncol=2)
    axh = fig.add_subplot(gs[1, :])
    order = [m for m in METHODS if m in o.index]
    M = o.loc[order, order].values.astype(float)
    full = np.where(np.tril(np.ones_like(M), k=-1) > 0, M, np.nan)   # lower triangle (the matrix is symmetric)
    L = full[1:, :-1]                                                 # rows 2..13, columns 1..12: no empty row or column
    im = axh.imshow(L, cmap="Blues", vmin=0, vmax=1)
    axh.set_xticks(range(len(order) - 1)); axh.set_xticklabels([NAME[m] for m in order[:-1]], rotation=90)
    axh.set_yticks(range(len(order) - 1)); axh.set_yticklabels([NAME[m] for m in order[1:]])
    for i in range(len(order) - 1):
        for j in range(i + 1):
            v = L[i, j]
            axh.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=MIN_PT, color="white" if v > 0.55 else INK)
    axh.grid(False); axh.tick_params(length=0)
    for sp in axh.spines.values():
        sp.set_visible(False)
    axh.set_title("(b) Overlap between methods (E5c): mean Jaccard index over 25 folds", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=axh, fraction=0.035, pad=0.01, aspect=30)
    cb.set_label("mean Jaccard index")

    def data_check(f):
        probs = lines_check(f, lambda key: a[(a.method == key[0]) & (a.k == 10)].sort_values("rho").delta_nogueira_vs_e4.values)
        if not np.allclose(M, M.T, equal_nan=True):
            probs.append("overlap matrix not symmetric: the lower triangle would hide values")
        arr = np.asarray(im.get_array(), dtype=float)
        exp = o.loc[order[1:], order[:-1]].values.astype(float)
        exp = np.where(np.tril(np.ones_like(exp), k=0) > 0, exp, np.nan)
        if not np.allclose(np.nan_to_num(arr, nan=-1), np.nan_to_num(exp, nan=-1)):
            probs.append("heatmap values differ from e5c_overlap_k10_gse37745.csv")
        return probs

    save(fig, out, "Figure_7.pdf", "Fig. 7. E5: redundancy and overlap on GSE37745 at k = 10", "E5; redundancy; overlap; Jaccard",
         ["e5a_redundancy.csv", "e5c_overlap_k10_gse37745.csv"], data_check)


# ------------------------------------------------------------------ Fig. 8 (E1)
def fig8(results, out):
    s = rd(results, "e1_summary.csv")
    s = s[s.classifier == "lr_l2"]
    b0 = rd(results, "e1_b0_table.csv").set_index("cohort")
    ks = [1, 2, 3, 4, 5, 10, 25, 50]
    members = {f: [m for m in METHODS if STYLE[m][0] == f] for f in FAMILIES}
    fig = new_fig(6.45)
    gs = fig.add_gridspec(len(COHORTS) + 1, 4, height_ratios=[1] * len(COHORTS) + [0.66])
    axes = [[None] * 4 for _ in COHORTS]
    for i, c in enumerate(COHORTS):
        g = s[s.cohort == c]
        r = g[g.method == "random_k"].set_index("k").reindex(ks)
        m0, s0 = [float(x) for x in b0.loc[c, "AUROC"].split("±")]
        lo = math.floor((float(np.nanmin(r.auroc_mean)) - 0.03) * 20) / 20
        for j, fam in enumerate(FAMILIES):
            ax = fig.add_subplot(gs[i, j], sharey=axes[i][0] if j else None)
            axes[i][j] = ax
            ax.fill_between(ks, r.auroc_mean - r.auroc_sd, np.minimum(1.0, r.auroc_mean + r.auroc_sd), color=INK3, alpha=0.2, linewidth=0)
            tag(ax.plot(ks, r.auroc_mean.values, color=INK2, linestyle=":", marker="x", markersize=2.6)[0], [c, "random_k"])
            ax.axhspan(m0 - s0, min(1.0, m0 + s0), color="#cde2fb", alpha=0.5, linewidth=0)
            ax.axhline(m0, color=INK, linestyle="--", linewidth=0.8)
            drawn = 0
            for m in members[fam]:
                gm = g[g.method == m].set_index("k").reindex(ks)
                if gm.auroc_mean.notna().sum() == 0:
                    continue
                fam_, ls, mk = STYLE[m]
                ok = gm.auroc_mean.notna().values
                kk = np.array(ks)[ok]
                tag(ax.plot(kk, gm.auroc_mean.values[ok], color=HUE[fam_], linestyle=ls, marker=mk, markersize=2.6)[0], [c, m])
                drawn += 1
            for m, k, mk, ms in [("ihc_2", 2, "*", 6.5), ("ihc_4", 4, "P", 5.5)]:
                gm = g[(g.method == m) & (g.k == k)]
                if len(gm):
                    ax.plot([k], [gm.auroc_mean.iloc[0]], color=INK, marker=mk, markersize=ms, linestyle="none", zorder=5)
            if drawn == 0:
                ax.text(0.97, 0.06, "not run", transform=ax.transAxes, ha="right", va="bottom", fontsize=SMALL, color=INK2)
            logk_axis(ax, [1, 4, 10, 50])
            ax.set_xlim(0.85, 60)
            ax.set_ylim(max(0.5, lo), 1.0)
            if i < len(COHORTS) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("budget k")
            if j:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(f"{COHORT[c]}\nAUROC")
            if i == 0:
                ax.set_title(fam, loc="left", fontweight="bold")
    for j, fam in enumerate(FAMILIES):
        axl = fig.add_subplot(gs[len(COHORTS), j]); axl.axis("off")
        axl.legend(handles=[mline(m, markersize=3.2) for m in members[fam]], loc="upper left", ncol=1, handlelength=2.3,
                   borderaxespad=0.0)
    refs = [Line2D([0], [0], color=INK, linestyle="--", label="B0 (all genes) ± SD"),
            Line2D([0], [0], color=INK2, linestyle=":", marker="x", label="random-k ± SD"),
            Line2D([0], [0], color=INK, marker="*", linestyle="none", markersize=6.5, label="IHC-2"),
            Line2D([0], [0], color=INK, marker="P", linestyle="none", markersize=5.5, label="IHC-4")]
    fig.legend(handles=refs, loc="outside lower center", ncol=4)

    def lookup(key):
        c, m = key
        g = s[(s.cohort == c) & (s.method == m)].set_index("k").reindex(ks)
        v = g.auroc_mean.values
        return v if m == "random_k" else v[~np.isnan(v)]

    def data_check(f):
        probs = lines_check(f, lookup)
        n_lines = sum(1 for ax in f.axes for ln in ax.get_lines() if ln.get_gid())
        expected = len(COHORTS) * 4 + sum(1 for c in COHORTS for m in METHODS if s[(s.cohort == c) & (s.method == m)].auroc_mean.notna().any())
        if n_lines != expected:
            probs.append("Fig. 8: %d tagged lines, expected %d" % (n_lines, expected))
        return probs

    save(fig, out, "Figure_8.pdf", "Fig. 8. E1: AUROC of LR-L2 as a function of the budget k", "E1; AUROC; budget k",
         ["e1_summary.csv (classifier = lr_l2)", "e1_b0_table.csv"], data_check)


# ------------------------------------------------------------------ Fig. 9 (pooled lower bound)
def fig9(results, out):
    b = rd(results, "e1_kstar_pooled_by_k.csv")
    b = b[~b.method.str.startswith("panel_")]
    ks = [1, 2, 3, 4, 5, 10, 25, 50]
    fig = new_fig(5.3)
    gs = fig.add_gridspec(2, 1)
    ax0 = None
    for i, (col, title) in enumerate([("lb_ivw", "(a) Inverse-variance weights"), ("lb_eq", "(b) Equal weights")]):
        ax = fig.add_subplot(gs[i, 0], sharex=ax0)
        ax0 = ax0 or ax
        ax.axhline(-0.02, color=INK, linestyle="--", linewidth=0.8)
        ax.axhline(0, color=INK3, linewidth=0.6)
        for m in METHODS + ["ihc_2", "ihc_4"]:
            g = b[b.method == m].set_index("k").reindex(ks)
            if g[col].notna().sum() == 0:
                continue
            ok = g[col].notna().values
            kk = np.array(ks)[ok]
            if m in STYLE:
                fam, ls, mk = STYLE[m]
                tag(ax.plot(kk, g[col].values[ok], color=HUE[fam], linestyle=ls, marker=mk)[0], [col, m])
            else:
                tag(ax.plot(kk, g[col].values[ok], color=INK, marker="*" if m == "ihc_2" else "P",
                            markersize=6.5 if m == "ihc_2" else 5.5, linestyle="none", zorder=5)[0], [col, m])
        logk_axis(ax, ks)
        ax.set_ylim(-0.12, 0.025)
        ax.set_yticks([-0.10, -0.05, 0.0]); ax.set_yticklabels(["−0.10", "−0.05", "0"])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("95% lower bound of ΔAUROC")
        ax.text(1.04, -0.0185, "margin −0.02", ha="left", va="bottom", fontsize=SMALL, color=INK)
        if i == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("budget k (genes)")
    ihc = [Line2D([0], [0], color=INK, marker="*", linestyle="none", markersize=6.5, label="IHC-2"),
           Line2D([0], [0], color=INK, marker="P", linestyle="none", markersize=5.5, label="IHC-4")]
    fig.legend(handles=grouped(*[[mline(m) for m in family_members(f)] for f in FAMILIES[:3]],
                               [mline(m) for m in family_members("Wrapper")] + ihc),
               loc="outside lower center", ncol=4)

    def lookup(key):
        col, m = key
        v = b[b.method == m].set_index("k").reindex(ks)[col].values
        return v[~np.isnan(v)]

    save(fig, out, "Figure_9.pdf", "Fig. 9. Pooled lower bound of AUROC(panel_k) - AUROC(B0)", "E1; minimum sufficient budget; non-inferiority",
         ["e1_kstar_pooled_by_k.csv"], lambda f: lines_check(f, lookup))


# ------------------------------------------------------------------ Fig. 10 (E1b)
def fig10(results, out):
    b = rd(results, "e1b_learning_curves.csv")
    b["n_t"] = b["n_t"].astype(str)
    levels = ["5", "10", "20", "40", "pool"]
    xs = {"5": 0, "10": 1, "20": 2, "40": 3, "pool": 4}
    fig = new_fig(5.35)
    gs = fig.add_gridspec(2, 2)
    for idx, c in enumerate(GEO):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        g = b[b.cohort == c]
        r = g[(g.method == "random_k") & (g.k == 10)].set_index("n_t").reindex(levels)
        okr = r.auroc_mean.notna().values
        x = [xs[l] for l, ok in zip(levels, okr) if ok]
        y = r.auroc_mean.values[okr]; sd = r.auroc_sd.values[okr]
        ax.fill_between(x, y - sd, np.minimum(1.0, y + sd), color=INK3, alpha=0.2, linewidth=0)
        tag(ax.plot(x, y, color=INK2, linestyle=":", marker="x")[0], [c, "random_k", 10])
        for m, k, mk, ls in [("ihc_4", 4, "P", "-"), ("panel_girard2016", 42, "v", "--")]:
            rr = g[(g.method == m) & (g.k == k)].set_index("n_t").reindex(levels)
            ok = rr.auroc_mean.notna().values
            tag(ax.plot([xs[l] for l, o in zip(levels, ok) if o], rr.auroc_mean.values[ok], color=INK, linestyle=ls,
                        marker=mk, markersize=3.8, linewidth=0.8)[0], [c, m, k])
        for m in NONWRAP:
            rr = g[(g.method == m) & (g.k == 10)].set_index("n_t").reindex(levels)
            if rr.auroc_mean.notna().sum() == 0:
                continue
            ok = rr.auroc_mean.notna().values
            fam, ls, mk = STYLE[m]
            tag(ax.plot([xs[l] for l, o in zip(levels, ok) if o], rr.auroc_mean.values[ok], color=HUE[fam], linestyle=ls,
                        marker=mk)[0], [c, m, 10])
        avail = [l for l in levels if l in set(g.n_t)]
        ax.set_xticks([xs[l] for l in avail]); ax.set_xticklabels(avail)
        ax.set_xlim(-0.3, 4.3)
        ax.set_title(COHORT[c], loc="left", fontweight="bold")
        if idx // 2 == 1:
            ax.set_xlabel("labelled patients per class $n_T$")
        if idx % 2 == 0:
            ax.set_ylabel("AUROC, k = 10")
        ax.set_ylim(max(0.55, math.floor((float(np.nanmin(y)) - 0.05) * 20) / 20), 1.0)
    handles = ref_handles("random") + [
        Line2D([0], [0], color=INK, linestyle="-", marker="P", markersize=3.8, label="IHC-4 (k = 4)"),
        Line2D([0], [0], color=INK, linestyle="--", marker="v", markersize=3.8, label="Girard 2016 (42 genes)")]
    fig.legend(handles=grouped(handles, *[[mline(m) for m in family_members(f, NONWRAP)] for f in FAMILIES[:3]]),
               loc="outside lower center", ncol=4)

    def lookup(key):
        c, m, k = key
        v = b[(b.cohort == c) & (b.method == m) & (b.k == k)].set_index("n_t").reindex(levels).auroc_mean.values
        return v[~np.isnan(v)]

    save(fig, out, "Figure_10.pdf", "Fig. 10. E1b: few-shot learning curves at k = 10", "E1b; few-shot; learning curve",
         ["e1b_learning_curves.csv"], lambda f: lines_check(f, lookup))


# ------------------------------------------------------------------ Fig. 11 (E7)
B1_STYLE = {"positive transfer": "flag", "negative transfer": "flag", "practically equivalent": "equiv",
            "inconclusive": None}


def fig11(results, out, b1=None):
    strategies = [("pooled", "pooled"), ("select_on_source", "select-on-source"), ("source_only", "source-only")]
    ks = [2, 3, 4, 5, 10]
    nts = [5, 10, 20]
    if b1:                                   # B1: 20 splits, four labels of the reading rule (after O49)
        e = pd.read_csv(os.path.join(b1, "b1_e7_split_labels.csv"), encoding="utf-8-sig")
        e = e.rename(columns={"mean_delta": "diff_mean"})
        bad_labels = sorted(set(e.label) - set(B1_STYLE))
        if bad_labels:
            raise ValueError("unknown labels in b1_e7_split_labels.csv: %s" % bad_labels)
        e["style"] = e.label.map(B1_STYLE)
        src_name = "b1_e7_split_labels.csv"
        sources = ["B1_tables/b1_e7_split_labels.csv (mean_delta, label; 20 splits)"]
        title = "Fig. 11. E7 (20 splits): transfer strategies against target-only L1-LR"
        cb_label = "mean ΔAUROC over 20 splits = strategy − target-only (colour clipped at ±0.10)"
        handles = [Patch(facecolor="white", edgecolor=INK, linewidth=1.2, label="positive or\nnegative transfer"),
                   Patch(facecolor="white", edgecolor=INK, linewidth=1.0, linestyle=(0, (2, 1.5)),
                         label="practically\nequivalent")]
    else:                                    # E7: 5 splits, descriptive flags of the official table
        e = rd(results, "e7_transfer_table.csv")
        e["style"] = [("flag" if (bool(p) or bool(n)) else None) for p, n in zip(e.positive_transfer, e.negative_transfer)]
        src_name = "e7_transfer_table.csv"
        sources = ["e7_transfer_table.csv"]
        title = "Fig. 11. E7: transfer strategies against target-only L1-LR"
        cb_label = "ΔAUROC = strategy − target-only (colour clipped at ±0.10)"
        handles = [Patch(facecolor="white", edgecolor=INK, linewidth=1.2, label="flagged cell")]
    fig = new_fig(4.15)
    gs = fig.add_gridspec(len(strategies) + 1, len(GEO), height_ratios=[1, 1, 1, 0.16])
    norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.10)
    cmap = plt.get_cmap("RdBu")
    last = None
    images = []
    for i, (st, stitle) in enumerate(strategies):
        for j, c in enumerate(GEO):
            ax = fig.add_subplot(gs[i, j])
            g = e[(e.cohort == c) & (e.strategy == st)]
            M = np.full((len(nts), len(ks)), np.nan)
            F = [[None for _ in ks] for _ in nts]
            for _, r in g.iterrows():
                a_, b_ = nts.index(int(r.n_t)), ks.index(int(r.k))
                M[a_, b_] = r.diff_mean
                F[a_][b_] = r["style"] if isinstance(r["style"], str) else None
            last = ax.imshow(np.clip(M, -0.10, 0.10), cmap=cmap, norm=norm, aspect="auto")
            images.append((last, c, st, M.copy()))
            for a_ in range(len(nts)):
                for b_ in range(len(ks)):
                    v = M[a_, b_]
                    if math.isnan(v):
                        ax.text(b_, a_, "–", ha="center", va="center", fontsize=MIN_PT, color=INK3)
                        continue
                    ax.text(b_, a_, ("%+d" % round(100 * v)).replace("-", "−").replace("+0", "0"), ha="center",
                            va="center", fontsize=MIN_PT, color="white" if abs(v) > 0.06 else INK)
                    if F[a_][b_]:
                        rect = Rectangle((b_ - 0.5 + 0.06, a_ - 0.5 + 0.06), 0.88, 0.88, fill=False, edgecolor=INK,
                                         linewidth=1.2 if F[a_][b_] == "flag" else 1.0, zorder=4,
                                         linestyle="-" if F[a_][b_] == "flag" else (0, (2, 1.5)))
                        rect.set_gid(json.dumps([c, st, nts[a_], ks[b_], F[a_][b_]]))
                        ax.add_patch(rect)
            ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks])
            ax.set_yticks(range(len(nts))); ax.set_yticklabels([str(nn) for nn in nts])
            ax.grid(False); ax.tick_params(length=0, pad=1.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(COHORT[c], loc="left", fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{stitle}\n$n_T$ per class")
            else:
                ax.tick_params(labelleft=False)
            if i == len(strategies) - 1:
                ax.set_xlabel("budget k")
            else:
                ax.tick_params(labelbottom=False)
    cax = fig.add_subplot(gs[len(strategies), 1:3])
    cb = fig.colorbar(last, cax=cax, orientation="horizontal")
    cb.set_ticks([-0.10, -0.05, 0, 0.05, 0.10]); cb.set_ticklabels(["−0.10", "−0.05", "0", "0.05", "0.10"])
    cb.set_label(cb_label)
    leg = fig.add_subplot(gs[len(strategies), 3]); leg.axis("off")
    leg.legend(handles=handles, loc="center left", handlelength=1.4, borderaxespad=0)
    txt = fig.add_subplot(gs[len(strategies), 0]); txt.axis("off")
    txt.text(0.0, 0.5, "value = 100 × ΔAUROC", transform=txt.transAxes, ha="left", va="center", fontsize=SMALL)

    def data_check(f):
        probs = []
        for im, c, st, M in images:
            g = e[(e.cohort == c) & (e.strategy == st)]
            exp = np.full((len(nts), len(ks)), np.nan)
            for _, r in g.iterrows():
                exp[nts.index(int(r.n_t)), ks.index(int(r.k))] = r.diff_mean
            if not np.allclose(np.nan_to_num(np.asarray(im.get_array(), dtype=float), nan=9), np.nan_to_num(np.clip(exp, -0.1, 0.1), nan=9)):
                probs.append("heatmap colours differ from %s: %s %s" % (src_name, c, st))
            if len(g) != np.isfinite(exp).sum():
                probs.append("duplicate cells in %s: %s %s" % (src_name, c, st))
        drawn = set()                                      # outlines drawn = outlines of the table, cell by cell
        for ax in f.axes:
            for pa in ax.patches:
                if pa.get_gid():
                    drawn.add(tuple(json.loads(pa.get_gid())))
        want = {(r.cohort, r.strategy, int(r.n_t), int(r.k), r["style"]) for _, r in e.iterrows()
                if isinstance(r["style"], str) and r.cohort in GEO and int(r.n_t) in nts and int(r.k) in ks}
        if drawn != want:
            probs.append("outlined cells differ from %s: %d drawn, %d in the table" % (src_name, len(drawn), len(want)))
        return probs

    save(fig, out, "Figure_11.pdf", title, "E7; transfer learning; few-shot", sources, data_check)


# ------------------------------------------------------------------ Fig. 12 (E6)
def fig12(results, out):
    t = rd(results, "e6_tradeoff.csv")
    t = t[(t.k == 10) & (~t.method.str.startswith("panel_")) & (t.method != "random_k")]
    cs = ["tcga_luad_lusc_primary_tumor", "gse37745"]
    fig = new_fig(5.0)
    gs = fig.add_gridspec(2, 2)
    plotted = []
    for j, c in enumerate(cs):
        g = t[t.cohort == c]
        top = fig.add_subplot(gs[0, j])
        bot = fig.add_subplot(gs[1, j], sharex=top)
        for ax, col in [(top, "auroc_mean"), (bot, "nogueira")]:
            for _, r in g.iterrows():
                fam, ls, mk = STYLE[r.method]
                par = bool(r.pareto_optimal)
                ax.scatter([r.wall_s_e2], [r[col]], s=30, color=HUE[fam], marker=mk, alpha=0.9,
                           edgecolor=INK if par else "white", linewidth=1.1 if par else 0.4, zorder=3)
                plotted.append((c, r.method, col, float(r.wall_s_e2), float(r[col])))
            ax.set_xscale("log"); ax.minorticks_off()
            ax.set_xticks([0.1, 1, 10, 100, 1000]); ax.set_xticklabels(["0.1", "1", "10", "100", "1,000"])
        top.set_title(f"{COHORT[c]}, k = 10", loc="left", fontweight="bold")
        top.tick_params(labelbottom=False)
        lo = math.floor((g.auroc_mean.min() - 0.003) * 200) / 200
        top.set_ylim(lo, min(1.0, g.auroc_mean.max() + 0.004))
        bot.set_ylim(-0.03, 1.0)
        bot.set_xlabel("selection time (s), log")
        if j == 0:
            top.set_ylabel("AUROC (E1)")
            bot.set_ylabel("Nogueira $\\hat{\\Phi}$ (E4)")
    present = [m for m in METHODS if m in set(t[t.cohort.isin(cs)].method)]

    def mk(m):
        return Line2D([0], [0], color=HUE[STYLE[m][0]], marker=STYLE[m][2], linestyle="none", markersize=5.0,
                      markeredgecolor="white", markeredgewidth=0.4, label=NAME[m])
    pareto = Line2D([0], [0], color="#cccccc", marker="o", linestyle="none", markersize=5.0, markeredgecolor=INK,
                    markeredgewidth=1.1, label="Pareto-optimal")
    fig.legend(handles=grouped(*[[mk(m) for m in family_members(f, present)] for f in FAMILIES[:3]],
                               [mk(m) for m in family_members("Wrapper", present)] + [pareto]),
               loc="outside lower center", ncol=4)

    def data_check(f):
        probs = []
        for c, m, col, x, y in plotted:
            row = t[(t.cohort == c) & (t.method == m)]
            if len(row) != 1 or abs(float(row.wall_s_e2.iloc[0]) - x) > 1e-12 or abs(float(row[col].iloc[0]) - y) > 1e-12:
                probs.append("point differs from e6_tradeoff.csv: %s %s %s" % (c, m, col))
        offs = [ax.collections for ax in f.axes]
        npts = sum(len(cl) for cl in offs)
        if npts != len(plotted):
            probs.append("Fig. 12: %d markers, %d table rows" % (npts, len(plotted)))
        return probs

    save(fig, out, "Figure_12.pdf", "Fig. 12. E6: AUROC, stability and cost at k = 10", "E6; Pareto front; trade-off",
         ["e6_tradeoff.csv"], data_check)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--e3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", type=int, help="figure numbers, e.g. --only 8 12")
    ap.add_argument("--b1", help="B1_tables folder: draw Fig. 11 from b1_e7_split_labels.csv (after O49)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    jobs = {5: lambda: fig5(a.e3, a.out), 6: lambda: fig6(a.results, a.out), 7: lambda: fig7(a.results, a.out),
            8: lambda: fig8(a.results, a.out), 9: lambda: fig9(a.results, a.out), 10: lambda: fig10(a.results, a.out),
            11: lambda: fig11(a.results, a.out, a.b1), 12: lambda: fig12(a.results, a.out)}
    for k in sorted(jobs):
        if a.only and k not in a.only:
            continue
        jobs[k]()
    with open(os.path.join(a.out, "figures_dmkd_report.json"), "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=1, ensure_ascii=False)
    if FAILED:
        print("FAILED:", FAILED)
        return 1
    print("OK: every figure %.2f pt wide, every glyph >= %g pt, no overlap, values equal to the tables"
          % (W_PDF_PT, MIN_PT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
