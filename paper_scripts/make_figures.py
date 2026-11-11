#!/usr/bin/env python3
"""
make_figures.py -- draw every experimental figure of the manuscript from the
official result files (tables_official_2026-11-03, commit e15fb7c; label [v03-11]).

Usage:
    python3 Scripts/make_figures.py --results <tables_official_2026-11-03> \
                                    --e3 <E2_E3 dir with e3_scaling_{n,p}.csv, e3_wrapper_evals.csv> \
                                    --out Images/

Rules: only mean +/- SD (never the t-based auroc_ci_* columns, Y40); SD bands are
clipped at 1.0; every figure is drawn from one or two named CSV files, recorded in
Images/figures_sources.json.  Colour encoding: one hue per family (filter blue,
embedded orange, constrained aqua, wrapper violet) and line style / marker per
member, so that identity never rests on colour alone; references in neutral ink.
Document information of every PDF: Author = Thu Nguyen, Title, Subject = source CSV files
(Scripts/set_figure_metadata.py adds the same fields as XMP).
Output file names = figure numbers of the manuscript (Figure_5.pdf ... Figure_12.pdf, order of appearance;
IPM asks for Figure_1, Figure_2, ...); the mapping is the dictionary FILE below.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e3e2dd", "grid.linewidth": 0.5, "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "figure.dpi": 150,
})

HUE = {"Filter": "#2a78d6", "Embedded": "#eb6834", "Constrained": "#1baf7a", "Wrapper": "#6250d6"}
STYLE = {  # (family, linestyle, marker)
    "f_test": ("Filter", "-", "o"), "mutual_info": ("Filter", "--", "s"), "mrmr": ("Filter", "-.", "^"), "relieff": ("Filter", ":", "D"),
    "l1_lr": ("Embedded", "-", "o"), "elastic_net": ("Embedded", "--", "s"), "rf_importance": ("Embedded", "-.", "^"), "svm_rfe": ("Embedded", ":", "D"),
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
COHORT = {"tcga_luad_lusc_primary_tumor": "TCGA (RNA-seq)", "gse10245": "GSE10245 (GPL570)", "gse30219": "GSE30219 (GPL570)",
          "gse37745": "GSE37745 (GPL570)", "gse42127": "GSE42127 (GPL6884)"}
COHORTS = ["tcga_luad_lusc_primary_tumor", "gse10245", "gse30219", "gse37745", "gse42127"]
GEO = ["gse10245", "gse30219", "gse37745", "gse42127"]
METHODS = list(STYLE.keys())
INK = "#0b0b0b"; INK2 = "#52514e"; INK3 = "#9a9893"
SOURCES = {}


def rd(d, f):
    return pd.read_csv(os.path.join(d, f), encoding="utf-8-sig")


def method_handles(methods):
    hs = []
    for m in methods:
        fam, ls, mk = STYLE[m]
        hs.append(Line2D([0], [0], color=HUE[fam], linestyle=ls, marker=mk, label=NAME[m]))
    return hs


AUTHOR = "Thu Nguyen"
# Output file name of each figure = its number in the manuscript (order of appearance, build of
# 08/11/2026). If the order of the figures changes, update FILE, TITLES and the \includegraphics lines.
FILE = {
    "fig_e3_scaling.pdf": "Figure_5.pdf",
    "fig_e4_stability.pdf": "Figure_6.pdf",
    "fig_e5_redundancy_overlap.pdf": "Figure_7.pdf",
    "fig_e1_auroc_vs_k.pdf": "Figure_8.pdf",
    "fig_kstar_lb.pdf": "Figure_9.pdf",
    "fig_e1b_curves.pdf": "Figure_10.pdf",
    "fig_e7_heatmap.pdf": "Figure_11.pdf",
    "fig_e6_tradeoff.pdf": "Figure_12.pdf",
}
TITLES = {
    "fig_e3_scaling.pdf": "Fig. 5. E3: wall time of the selection step against n, p and N_eval",
    "fig_e4_stability.pdf": "Fig. 6. E4: Nogueira index on TCGA and GSE37745",
    "fig_e5_redundancy_overlap.pdf": "Fig. 7. E5: redundancy and overlap on GSE37745 at k = 10",
    "fig_e1_auroc_vs_k.pdf": "Fig. 8. E1: AUROC of LR-L2 as a function of the budget k",
    "fig_kstar_lb.pdf": "Fig. 9. Pooled lower bound of AUROC(panel_k) - AUROC(B0)",
    "fig_e1b_curves.pdf": "Fig. 10. E1b: few-shot learning curves at k = 10",
    "fig_e7_heatmap.pdf": "Fig. 11. E7: transfer strategies against target-only L1-LR",
    "fig_e6_tradeoff.pdf": "Fig. 12. E6: AUROC, stability and cost at k = 10",
}


def save(fig, out, name, sources):
    fname = FILE.get(name, name)
    path = os.path.join(out, fname)
    meta = {"Author": AUTHOR, "Title": TITLES.get(name, name),
            "Subject": "Drawn from " + "; ".join(sources) + " (tables_official_2026-11-03, commit e15fb7c, label [v03-11])"}
    fig.savefig(path, bbox_inches="tight", metadata=meta)
    plt.close(fig)
    SOURCES[fname] = sources
    print("wrote", path)


# --------------------------------------------------------------------------
def fig_e1_auroc_vs_k(results, out):
    s = rd(results, "e1_summary.csv")
    s = s[s.classifier == "lr_l2"]
    b0 = rd(results, "e1_b0_table.csv").set_index("cohort")
    ks = [1, 2, 3, 4, 5, 10, 25, 50]
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 4.9))
    for i, c in enumerate(COHORTS):
        g = s[s.cohort == c]
        r = g[g.method == "random_k"].set_index("k").reindex(ks)
        m0, s0 = [float(x) for x in b0.loc[c, "AUROC"].split("±")]
        zoom_lo = 1.0
        for row in [0, 1]:
            ax = axes[row, i]
            ax.fill_between(ks, r.auroc_mean - r.auroc_sd, np.minimum(1.0, r.auroc_mean + r.auroc_sd), color=INK3, alpha=0.18, linewidth=0)
            ax.plot(ks, r.auroc_mean, color=INK2, linestyle=":", marker="x", markersize=3)
            ax.axhspan(m0 - s0, min(1.0, m0 + s0), color="#cde2fb", alpha=0.45, linewidth=0)
            ax.axhline(m0, color=INK, linestyle="--", linewidth=0.9)
            for m in METHODS:
                gm = g[g.method == m].set_index("k").reindex(ks)
                if gm.auroc_mean.notna().sum() == 0:
                    continue
                fam, ls, mk = STYLE[m]
                ax.plot(ks, gm.auroc_mean, color=HUE[fam], linestyle=ls, marker=mk, markersize=2.4, linewidth=0.85, alpha=0.95)
                zoom_lo = min(zoom_lo, float(np.nanmin(gm.auroc_mean[gm.index >= 2])))
            for m, k, mk in [("ihc_2", 2, "*"), ("ihc_4", 4, "P")]:
                gm = g[(g.method == m) & (g.k == k)]
                if len(gm):
                    ax.plot([k], [gm.auroc_mean.iloc[0]], color=INK, marker=mk, markersize=5.5, linestyle="none", zorder=5)
            ax.set_xscale("log")
            ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks], fontsize=5.6)
            ax.minorticks_off()
            if row == 0:
                lo = max(0.55, float(np.nanmin(r.auroc_mean) - 0.03))
                ax.set_ylim(lo, 1.0)
                ax.set_title(COHORT[c], loc="left", fontweight="bold", fontsize=7)
            else:
                ax.set_ylim(max(0.80, zoom_lo - 0.01), 1.0)
                ax.set_xlabel("budget k (genes)")
                ax.set_title("zoom", loc="left", fontsize=6.5, color=INK2)
        axes[0, 0].set_ylabel("AUROC (mean over folds)")
        axes[1, 0].set_ylabel("AUROC, zoomed")
    handles = [Line2D([0], [0], color=INK, linestyle="--", label="B0 (all genes), band = ±SD"),
               Line2D([0], [0], color=INK2, linestyle=":", marker="x", label="random-k, band = ±SD"),
               Line2D([0], [0], color=INK, marker="*", linestyle="none", markersize=6, label="IHC-2 (k = 2)"),
               Line2D([0], [0], color=INK, marker="P", linestyle="none", markersize=6, label="IHC-4 (k = 4)")]
    handles += method_handles(METHODS)
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.09), handlelength=2.4, columnspacing=0.9)
    fig.tight_layout(h_pad=0.8, w_pad=0.5)
    save(fig, out, "fig_e1_auroc_vs_k.pdf", ["e1_summary.csv (auroc_mean, auroc_sd; classifier = lr_l2)", "e1_b0_table.csv (AUROC)"])


# --------------------------------------------------------------------------
def fig_kstar_lb(results, out):
    b = rd(results, "e1_kstar_pooled_by_k.csv")
    b = b[~b.method.str.startswith("panel_")]
    ks = [1, 2, 3, 4, 5, 10, 25, 50]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, col, title in [(axes[0], "lb_ivw", "inverse-variance weights"), (axes[1], "lb_eq", "equal weights")]:
        ax.axhline(-0.02, color=INK, linestyle="--", linewidth=0.9)
        ax.axhline(0, color=INK3, linewidth=0.6)
        for m in METHODS + ["ihc_2", "ihc_4"]:
            g = b[b.method == m].set_index("k").reindex(ks)
            if g[col].notna().sum() == 0:
                continue
            if m in STYLE:
                fam, ls, mk = STYLE[m]
                ax.plot(ks, g[col], color=HUE[fam], linestyle=ls, marker=mk, markersize=2.6, linewidth=0.9)
            else:
                ax.plot(ks, g[col], color=INK, marker="*" if m == "ihc_2" else "P", linestyle="none", markersize=6, zorder=5)
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
        ax.set_ylim(-0.12, 0.03)
        ax.set_title(f"Pooled lower bound, {title}", loc="left", fontweight="bold")
        ax.set_xlabel("budget k (genes)")
    axes[0].set_ylabel("95% lower bound of ΔAUROC (panel − B0)")
    axes[0].text(1.05, -0.017, "bound −0.02", fontsize=6, color=INK)
    handles = method_handles(METHODS) + [Line2D([0], [0], color=INK, marker="*", linestyle="none", markersize=6, label="IHC-2"),
                                         Line2D([0], [0], color=INK, marker="P", linestyle="none", markersize=6, label="IHC-4")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.12), handlelength=2.4, columnspacing=0.9)
    fig.tight_layout()
    save(fig, out, "fig_kstar_lb.pdf", ["e1_kstar_pooled_by_k.csv (lb_ivw, lb_eq)"])


# --------------------------------------------------------------------------
def fig_e1b(results, out):
    b = rd(results, "e1b_learning_curves.csv")
    b["n_t"] = b["n_t"].astype(str)
    levels = ["5", "10", "20", "40", "pool"]
    xs = {"5": 0, "10": 1, "20": 2, "40": 3, "pool": 4}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=False)
    for ax, c in zip(axes, GEO):
        g = b[(b.cohort == c)]
        r = g[(g.method == "random_k") & (g.k == 10)].set_index("n_t").reindex(levels)
        x = [xs[l] for l in levels if not math.isnan(r.loc[l, "auroc_mean"])]
        y = r.auroc_mean.dropna().values; sd = r.auroc_sd.dropna().values
        ax.fill_between(x, y - sd, np.minimum(1.0, y + sd), color=INK3, alpha=0.18, linewidth=0)
        ax.plot(x, y, color=INK2, linestyle=":", marker="x")
        for m, k, mk in [("ihc_4", 4, "P"), ("panel_girard2016", 42, "v")]:
            rr = g[(g.method == m) & (g.k == k)].set_index("n_t").reindex(levels)
            xx = [xs[l] for l in levels if not math.isnan(rr.loc[l, "auroc_mean"])]
            ax.plot(xx, rr.auroc_mean.dropna().values, color=INK, linestyle="-" if m == "ihc_4" else "--", marker=mk, markersize=3.5, linewidth=0.8)
        for m in METHODS:
            if m in ("ga", "bpso"):
                continue
            rr = g[(g.method == m) & (g.k == 10)].set_index("n_t").reindex(levels)
            if rr.auroc_mean.notna().sum() == 0:
                continue
            xx = [xs[l] for l in levels if not math.isnan(rr.loc[l, "auroc_mean"])]
            fam, ls, mk = STYLE[m]
            ax.plot(xx, rr.auroc_mean.dropna().values, color=HUE[fam], linestyle=ls, marker=mk, markersize=2.6, linewidth=0.9)
        avail = [l for l in levels if l in set(g.n_t)]
        ax.set_xticks([xs[l] for l in avail]); ax.set_xticklabels(avail)
        ax.set_xlim(-0.3, 4.3)
        ax.set_title(COHORT[c].split(" ")[0], loc="left", fontweight="bold")
        ax.set_xlabel("labelled patients per class, $n_T$")
        lo = max(0.55, float(np.nanmin(y) - 0.05))
        ax.set_ylim(lo, 1.0)
    axes[0].set_ylabel("AUROC on the fixed test third (k = 10)")
    handles = [Line2D([0], [0], color=INK2, linestyle=":", marker="x", label="random-k (band = ±SD)"),
               Line2D([0], [0], color=INK, linestyle="-", marker="P", markersize=4, label="IHC-4 (k = 4, no selection)"),
               Line2D([0], [0], color=INK, linestyle="--", marker="v", markersize=4, label="Girard 2016 (42 genes, no selection)")]
    handles += method_handles([m for m in METHODS if m not in ("ga", "bpso")])
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.16), handlelength=2.4, columnspacing=0.9)
    fig.tight_layout()
    save(fig, out, "fig_e1b_curves.pdf", ["e1b_learning_curves.csv (auroc_mean, auroc_sd at k = 10; fixed panels at their own k)"])


# --------------------------------------------------------------------------
def fig_e7(results, out):
    e = rd(results, "e7_transfer_table.csv")
    strategies = [("pooled", "pooled (source + $L_T$)"), ("select_on_source", "select-on-source"), ("source_only", "source-only (zero-shot)")]
    ks = [2, 3, 4, 5, 10]; nts = [5, 10, 20]
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 4.4))
    norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.10)
    cmap = plt.get_cmap("RdBu")
    last = None
    for i, (s, stitle) in enumerate(strategies):
        for j, c in enumerate(GEO):
            ax = axes[i, j]
            g = e[(e.cohort == c) & (e.strategy == s)]
            M = np.full((len(nts), len(ks)), np.nan)
            for _, r in g.iterrows():
                M[nts.index(int(r.n_t)), ks.index(int(r.k))] = r.diff_mean
            im = ax.imshow(np.clip(M, -0.10, 0.10), cmap=cmap, norm=norm, aspect="auto")
            last = im
            for a in range(len(nts)):
                for b_ in range(len(ks)):
                    v = M[a, b_]
                    if math.isnan(v):
                        ax.text(b_, a, "–", ha="center", va="center", fontsize=6, color=INK3)
                        continue
                    row = g[(g.n_t == nts[a]) & (g.k == ks[b_])].iloc[0]
                    mark = "+" if bool(row.positive_transfer) else ("−" if bool(row.negative_transfer) else "")
                    txt = f"{v:+.2f}" if abs(v) < 0.995 else f"{v:+.2f}"
                    col = "white" if abs(v) > 0.06 else INK
                    ax.text(b_, a, f"{txt}{mark}", ha="center", va="center", fontsize=5.6, color=col)
            ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks])
            ax.set_yticks(range(len(nts))); ax.set_yticklabels([str(n) for n in nts])
            ax.grid(False)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(COHORT[c].split(" ")[0], loc="left", fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{stitle}\n$n_T$ per class", fontsize=6.5)
            if i == 2:
                ax.set_xlabel("budget k")
    cb = fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cb.set_label("ΔAUROC = strategy − target-only (clipped at ±0.10)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    save(fig, out, "fig_e7_heatmap.pdf", ["e7_transfer_table.csv (diff_mean, positive_transfer, negative_transfer)"])


# --------------------------------------------------------------------------
def fig_e3(e3dir, out):
    n = pd.read_csv(os.path.join(e3dir, "e3_scaling_n.csv"), encoding="utf-8-sig")
    p = pd.read_csv(os.path.join(e3dir, "e3_scaling_p.csv"), encoding="utf-8-sig")
    w = pd.read_csv(os.path.join(e3dir, "e3_wrapper_evals.csv"), encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1, 0.75]})
    for ax, df, xcol, title, xlabel in [(axes[0], n, "n", "time vs n  (p = 5,000, k = 10)", "training patients n"),
                                        (axes[1], p, "p", "time vs p  (n = 812, k = 10)", "genes after pre-filter p")]:
        for m in METHODS:
            g = df[df.method == m].sort_values(xcol)
            if g.empty:
                continue
            fam, ls, mk = STYLE[m]
            ax.plot(g[xcol], g.wall_s, color=HUE[fam], linestyle=ls, marker=mk, markersize=2.8, linewidth=0.9)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_title(title, loc="left", fontweight="bold")
        xt = sorted(df[xcol].unique())
        ax.set_xticks(xt); ax.set_xticklabels([f"{x:,}" for x in xt], rotation=0)
        ax.minorticks_off()
    axes[0].set_ylabel("wall time of selection (s, median of 3)")
    ax = axes[2]
    for m in ["ga", "bpso"]:
        g = w[w.method == m].sort_values("n_eval_nominal")
        fam, ls, mk = STYLE[m]
        ax.plot(g.n_eval_nominal, g.wall_s, color=HUE[fam], linestyle=ls, marker=mk, markersize=3.2, linewidth=0.9, label=NAME[m])
        for _, r in g.iterrows():
            ax.annotate(f"{int(r['pop'])}×{int(r.generations)}", (r.n_eval_nominal, r.wall_s), textcoords="offset points", xytext=(3, -8), fontsize=5.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("fitness evaluations $N_{\\mathrm{eval}}$ (P × T)"); ax.set_title("wrappers vs $N_{\\mathrm{eval}}$", loc="left", fontweight="bold")
    ax.set_xticks([200, 1500]); ax.set_xticklabels(["200", "1,500"]); ax.minorticks_off()
    ax.legend(frameon=False, loc="upper left")
    handles = method_handles([m for m in METHODS if m not in ("ga", "bpso")])
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12), handlelength=2.4, columnspacing=0.9)
    fig.tight_layout()
    save(fig, out, "fig_e3_scaling.pdf", ["E2_E3/e3_scaling_n.csv (wall_s)", "E2_E3/e3_scaling_p.csv (wall_s)", "E2_E3/e3_wrapper_evals.csv (wall_s, n_eval_nominal)"])


# --------------------------------------------------------------------------
def fig_e4(results, out):
    s = rd(results, "e4_stability.csv")
    ks = [1, 2, 3, 4, 5, 10, 25, 50]
    cs = ["tcga_luad_lusc_primary_tumor", "gse37745"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for j, c in enumerate(cs):
        ax = axes[0, j]
        g = s[(s.cohort == c) & (s.fraction == 0.7)]
        for m in METHODS + ["random_k"]:
            gm = g[g.method == m].set_index("k").reindex(ks)
            if gm.nogueira.notna().sum() == 0:
                continue
            if m == "random_k":
                ax.plot(ks, gm.nogueira, color=INK2, linestyle=":", marker="x")
                continue
            fam, ls, mk = STYLE[m]
            ax.errorbar(ks, gm.nogueira, yerr=[gm.nogueira - gm.nogueira_ci_low, gm.nogueira_ci_high - gm.nogueira],
                        color=HUE[fam], linestyle=ls, marker=mk, markersize=2.6, linewidth=0.9, elinewidth=0.5, capsize=1.2, alpha=0.95)
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
        ax.set_ylim(-0.05, 1.0)
        ax.set_title(f"{COHORT[c].split(' ')[0]}: stability vs k  (f = 0.7)", loc="left", fontweight="bold")
        ax.set_xlabel("budget k (genes)")
        if j == 0:
            ax.set_ylabel("Nogueira index Φ̂ [95% CI]")
        ax = axes[1, j]
        for m in METHODS + ["random_k"]:
            vals = []
            for f in [0.5, 0.7, 0.9]:
                gm = s[(s.cohort == c) & (s.fraction == f) & (s.method == m) & (s.k == 10)]
                vals.append(float(gm.nogueira.iloc[0]) if len(gm) else np.nan)
            if all(math.isnan(v) for v in vals):
                continue
            if m == "random_k":
                ax.plot([0.5, 0.7, 0.9], vals, color=INK2, linestyle=":", marker="x")
                continue
            fam, ls, mk = STYLE[m]
            ax.plot([0.5, 0.7, 0.9], vals, color=HUE[fam], linestyle=ls, marker=mk, markersize=2.8, linewidth=0.9)
        ax.set_xticks([0.5, 0.7, 0.9]); ax.set_xlim(0.45, 0.95); ax.set_ylim(-0.05, 1.0)
        ax.set_title(f"{COHORT[c].split(' ')[0]}: stability vs subsample fraction  (k = 10)", loc="left", fontweight="bold")
        ax.set_xlabel("subsample fraction f")
        if j == 0:
            ax.set_ylabel("Nogueira index Φ̂")
    handles = method_handles(METHODS) + [Line2D([0], [0], color=INK2, linestyle=":", marker="x", label="random-k")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.07), handlelength=2.4, columnspacing=0.9)
    fig.tight_layout(h_pad=1.2)
    save(fig, out, "fig_e4_stability.pdf", ["e4_stability.csv (nogueira, nogueira_ci_low, nogueira_ci_high)"])


# --------------------------------------------------------------------------
def fig_e5(results, out):
    a = rd(results, "e5a_redundancy.csv")
    o = rd(results, "e5c_overlap_k10_gse37745.csv").set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.axhline(0, color=INK3, linewidth=0.6)
    for m in [m for m in METHODS if m not in ("ga", "bpso")] + ["random_k"]:
        g = a[(a.method == m) & (a.k == 10)].sort_values("rho")
        if g.empty:
            continue
        if m == "random_k":
            ax.plot(g.rho, g.delta_nogueira_vs_e4, color=INK2, linestyle=":", marker="x")
            continue
        fam, ls, mk = STYLE[m]
        ax.plot(g.rho, g.delta_nogueira_vs_e4, color=HUE[fam], linestyle=ls, marker=mk, markersize=2.8, linewidth=0.9)
    ax.set_xticks([0.5, 0.8, 0.95]); ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("correlation ρ of the 50 injected copies")
    ax.set_ylabel("ΔΦ̂ (raw index − E4 at f = 0.7)")
    ax.set_title("(a) GSE37745, k = 10: effect of redundancy", loc="left", fontweight="bold")
    ax = axes[1]
    order = [m for m in METHODS if m in o.index]
    M = o.loc[order, order].values.astype(float)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([NAME[m] for m in order], rotation=60, ha="right", fontsize=5.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME[m] for m in order], fontsize=5.8)
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j:
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.6, color="white" if v > 0.55 else INK)
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(b) GSE37745, k = 10: Jaccard overlap between methods", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("mean Jaccard over 25 folds", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    handles = method_handles([m for m in METHODS if m not in ("ga", "bpso")]) + [Line2D([0], [0], color=INK2, linestyle=":", marker="x", label="random-k")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.14), handlelength=2.4, columnspacing=0.9)
    fig.tight_layout()
    save(fig, out, "fig_e5_redundancy_overlap.pdf", ["e5a_redundancy.csv (delta_nogueira_vs_e4 at k = 10)", "e5c_overlap_k10_gse37745.csv"])


# --------------------------------------------------------------------------
def fig_e6(results, out):
    t = rd(results, "e6_tradeoff.csv")
    t = t[(t.k == 10) & (~t.method.str.startswith("panel_")) & (t.method != "random_k")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, c in zip(axes, ["tcga_luad_lusc_primary_tumor", "gse37745"]):
        g = t[t.cohort == c]
        for _, r in g.iterrows():
            fam, ls, mk = STYLE[r.method]
            size = 10 + 90 * max(0.0, float(r.nogueira))
            ax.scatter(r.wall_s_e2, r.auroc_mean, s=size, color=HUE[fam], marker=mk, alpha=0.85,
                       edgecolor=INK if bool(r.pareto_optimal) else "none", linewidth=1.0 if bool(r.pareto_optimal) else 0, zorder=3)
            ax.annotate(NAME[r.method], (r.wall_s_e2, r.auroc_mean), textcoords="offset points", xytext=(4, 3), fontsize=5.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlabel("selection wall time at n = 812, p = 5,000 (s, log)")
        ax.set_title(f"{COHORT[c].split(' ')[0]}, k = 10", loc="left", fontweight="bold")
        ax.set_ylim(min(0.97, g.auroc_mean.min() - 0.004), min(1.0, g.auroc_mean.max() + 0.004))
    axes[0].set_ylabel("AUROC (mean over 25 folds)")
    handles = [Patch(facecolor=HUE[f], label=f) for f in HUE]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="#cccccc", markeredgecolor=INK, markersize=7, label="Pareto-optimal (outlined)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#cccccc", markersize=3, label="marker area ∝ Nogueira Φ̂ (f = 0.7)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, out, "fig_e6_tradeoff.pdf", ["e6_tradeoff.csv (auroc_mean, nogueira, wall_s_e2, pareto_optimal at k = 10)"])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--e3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_e1_auroc_vs_k(a.results, a.out)
    fig_kstar_lb(a.results, a.out)
    fig_e1b(a.results, a.out)
    fig_e7(a.results, a.out)
    fig_e3(a.e3, a.out)
    fig_e4(a.results, a.out)
    fig_e5(a.results, a.out)
    fig_e6(a.results, a.out)
    with open(os.path.join(a.out, "figures_sources.json"), "w", encoding="utf-8") as f:
        json.dump(SOURCES, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
