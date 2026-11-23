#!/usr/bin/env python3
r"""
make_supplementary_tables.py -- tables of the sensitivity analyses SA1-SA4, of the second task E8 and of the
re-analyses of the stored results (18/11/2026, QD208-QD214; QD173 items 1-5, QD198-QD201), and tables of E7 over
20 test/pool splits, of the two transfer methods with a model of the source and of n_T* over 20 splits
(21/11/2026, QD226, QD227, QD236-QD238; item 6 of QD173).

The script reads result tables only; it runs no experiment.  Every table file that it writes names its input file(s)
and columns in its first comment lines, and every derived number used in the running text is written to
Tables/numbers_supplement.json (logged in working/03_claims.md, Sections W and X).

Inputs (parameters, so that the script can be run from a checkout of the code repository as well):
  --official  tables of the main benchmark, label [v03-11] (e1_summary.csv, e1_tests_vs_random_b0.csv,
              e1_kstar_pooled_by_k.csv, e1_kstar_pooled.csv, e1b_learning_curves.csv, e1b_stability.csv,
              e1_ablation_noihc.csv, e4_stability.csv)
              default: input/results/_received_2026-11-03/tables_official_2026-11-03
  --extra     tables of the additional analyses, tag survey-v3-dmkd.2 (folders B2_tables ... B7_tables, B7_gate)
              default: input/results/_received_2026-11-16/tables_survey-v3-dmkd.2
  --k2        gene-level evidence of the two-gene zero-shot rule (e7_k2_gene_directions.csv)
              default: input/results/_received_2026-11-09/E7_k2_evidence
  --b1        tables of the 20-split runs, branch dmkd-revision at commit 61f6252 (folders B1_tables, B1_E7_20, B6_tables;
              in the repository: results/survey_v2)
              default: input/results/_received_2026-11-19/tables_dmkd-revision_61f6252
  --derived   optional: the advisor's derived numbers of 19/11/2026, used for cross-checks only (skipped when absent)
              default: input/results/_received_2026-11-19/derived
  --out       folder of the table files            default: paper/Springer/Tables

Names used in the paper (the repository keeps the folder names of the right-hand column):
  SA1  reference model B0 under a wider grid of C and on the pre-filtered genes     B4_tables
  SA2  published signatures cut to ten genes inside the training fold               B5_tables
  SA3  k-TSP pairs classified by the voting rule of the pairs                       B2_tables
  SA4  two variants of the zero-shot strategy                                       B3_tables
  E8   second task: EGFR mutation in lung adenocarcinoma                            B7_gate, B7_tables
  E7 over 20 test/pool splits, n_T* over 20 splits (no separate name in the paper)  B1_tables, B1_E7_20
  ptLasso and Trans-GLM in E7 (no separate name in the paper)                       B6_tables

Tables written (numbers of the manuscript of 21/11/2026 in working/05_decisions_and_design.md, Section 64)
  article      tab_e7_20splits.tex (E7 at k = 10 with the labels over 20 splits), tab_e8.tex (E8 at k = 10)
  supplement   tab_app_e1_<cohort>.tex (AUROC at every budget with the marks of the corrected tests),
               tab_app_e1_{bacc,mcc,logloss,brier}.tex, tab_app_e1b_secondary.tex and tab_app_e1b_stability.tex (two
               landscape tables of two cohorts in each file, labels tab:app_e1b_secondary[_b], tab:app_e1b_stability[_b]),
               tab_app_ablation_full.tex, tab_app_kstar_sens.tex, tab_app_sens_reference.tex (SA1), tab_app_sig10.tex,
               tab_app_sig10_fewshot.tex, tab_app_ktsp_vote.tex, tab_app_ktsp_vote_fewshot.tex, tab_app_zeroshot.tex,
               tab_app_zeroshot_k2.tex (two-gene rule and SA4 at k = 2), tab_app_e8_tcga.tex, tab_app_e8_gse31210.tex,
               tab_app_e8_rank.tex, tab_app_e8_secondary.tex, tab_app_e7_20splits.tex (every cell of E7 over 20 splits,
               with the five-split difference), tab_app_b6.tex, tab_app_b6_naive.tex, tab_app_nt_star.tex
  Until 20/11/2026 tab_sens_reference.tex and tab_zeroshot_k2.tex were tables of the article; the five-split tables
  tab_e7.tex and tab_app_e7.tex of make_tables.py are no longer used by the manuscript (QD226).
  Layout of 21/11/2026 (QD241): no table of the supplementary file is printed below 6.9 pt.  Landscape: tab_app_e7_20splits,
  tab_app_b6_naive, tab_app_zeroshot, tab_app_sig10 (interval beneath the difference), the two parts of tab_app_e1b_secondary
  and of tab_app_e1b_stability.  The values of every table are unchanged.

Checks made while writing
  * tab_app_e1_<cohort>.tex (Tables S5-S9 of Online Resource 1 since 21/11/2026): with the marks removed, every cell
    equals the cell of the table file that was in place before (same mean and SD), so only the marks are new.
  * k* re-analysis: the pooled differences, standard errors and bounds recomputed from the per-cohort tests
    (e1_tests_vs_random_b0.csv) equal the stored values of e1_kstar_pooled_by_k.csv to 1e-9 in all 92 rows, and the
    fixed-sequence rule applied to the stored bounds returns the stored k* of e1_kstar_pooled.csv for every method.
  * Transfer labels: the label of each of the 165 + 110 cells follows from its stored Holm p-value, mean difference
    and bootstrap interval by the rule fixed on 12/11/2026; the equality gates of the 20-split runs passed with a
    maximum difference of 0; no cell of pooled or select-on-source is labelled negative transfer; in every cell
    labelled positive or negative transfer, the five-split difference has the same sign.
  * n_T*: the rule applied to the five-split table returns its stored n_t_star in every cell.

Usage (from the project root or from paper/Springer/):  python3 paper/Springer/Scripts/make_supplementary_tables.py
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
import sys

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

FAMILIES = [("Filter", ["f_test", "mutual_info", "mrmr", "relieff"]),
            ("Embedded", ["l1_lr", "elastic_net", "rf_importance", "svm_rfe"]),
            ("Constrained", ["abess", "stability_selection", "ktsp"]),
            ("Wrapper", ["ga", "bpso"])]
NAME = {"f_test": "F-test", "mutual_info": "MI", "mrmr": "mRMR", "relieff": "ReliefF", "l1_lr": "L1-LR",
        "elastic_net": "Elastic Net", "rf_importance": "RF importance", "svm_rfe": "SVM-RFE", "abess": "abess (L0)",
        "stability_selection": "Stability selection", "ktsp": "$k$-TSP", "ga": "GA", "bpso": "BPSO",
        "ihc_2": "IHC-2", "ihc_4": "IHC-4", "panel_girard2016": "Girard 2016", "panel_wilkerson2013": "Wilkerson 2013",
        "random_k": "random-$k$", "b0": "B0"}
SELECTORS = [m for _, ms in FAMILIES for m in ms]
NONWRAPPER = [m for m in SELECTORS if m not in ("ga", "bpso")]
COHORTS = ["tcga_luad_lusc_primary_tumor", "gse10245", "gse30219", "gse37745", "gse42127"]
CNAME = {"tcga_luad_lusc_primary_tumor": "TCGA", "gse10245": "GSE10245", "gse30219": "GSE30219", "gse37745": "GSE37745",
         "gse42127": "GSE42127", "tcga_luad_egfr": "TCGA-LUAD", "gse31210_egfr": "GSE31210"}
CSHORT = {"tcga_luad_lusc_primary_tumor": "tcga", "gse10245": "gse10245", "gse30219": "gse30219", "gse37745": "gse37745",
          "gse42127": "gse42127"}
GEO = COHORTS[1:]
GPL570 = ["gse10245", "gse30219", "gse37745"]
BUDGETS = ["1", "2", "3", "4", "5", "10", "25", "50"]
NUMBERS = {}
WRITTEN = []


def rd(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f3(x, plus=False):
    """Three decimals; a typographic minus is produced by math mode in the table cells that need it."""
    v = float(x)
    s = ("%+.3f" if plus else "%.3f") % v
    if s in ("-0.000", "+0.000") and not plus:
        s = "0.000"
    return s


def pm(mean, sd):
    return "%s $\\pm$ %s" % (f3(mean), f3(sd))


def sgn(x):
    """Signed three-decimal number in math mode (true minus sign)."""
    return "$%+.3f$" % float(x)


def write(out, name, text):
    path = os.path.join(out, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    WRITTEN.append(name)


def family_rows(methods, row):
    """Rows grouped by family with \\addlinespace between families; row(method) -> list of cells or None."""
    lines, first_block = [], True
    for fam, ms in FAMILIES:
        block = []
        for m in ms:
            if m not in methods:
                continue
            cells = row(m)
            if cells is None:
                continue
            block.append((m, cells))
        if not block:
            continue
        if not first_block:
            lines.append("\\addlinespace[2pt]")
        first_block = False
        for i, (m, cells) in enumerate(block):
            lines.append("%s & %s & %s \\\\" % (fam if i == 0 else "", NAME[m], " & ".join(cells)))
    return lines


# ---------------------------------------------------------------------------------------------------------------
#  1. AUROC at every budget, one table per cohort, with the marks of the corrected tests (QD173, item 4);
#     Tables S5-S9 of Online Resource 1 since 21/11/2026 (Tables S2-S6 until 20/11/2026)
# ---------------------------------------------------------------------------------------------------------------
def tables_e1_marks(official, out):
    summ = {(r["cohort"], r["method"], r["k"]): r for r in rd(os.path.join(official, "e1_summary.csv"))
            if r["classifier"] == "lr_l2"}
    tests = {(r["cohort"], r["method"], r["k"]): r for r in rd(os.path.join(official, "e1_tests_vs_random_b0.csv"))}
    counts = {}
    below_b0, above_b0 = 0, 0
    for c in COHORTS:
        old = open(os.path.join(out, "tab_app_e1_%s.tex" % CSHORT[c]), encoding="utf-8").read()
        old_cells = {}
        for line in old.split("\n"):
            if " & " in line and line.rstrip().endswith("\\\\") and "\\textbf{Family}" not in line:
                parts = [p.strip() for p in line.rstrip()[:-2].split("&")]
                old_cells[parts[1]] = [re.sub(r"\$\^\{\\(dagger|ddagger|dagger\\ddagger)\}\$", "", p) for p in parts[2:]]

        def cell(m, k):
            r = summ.get((c, m, k))
            if r is None:
                return "--"
            s = pm(r["auroc_mean"], r["auroc_sd"])
            t = tests.get((c, m, k))
            mark = ""
            if t is not None:
                if t["beats_random_holm"] == "True":
                    mark += "\\dagger"
                    counts[(c, k, "beat")] = counts.get((c, k, "beat"), 0) + (1 if m in SELECTORS else 0)
                if t["differs_from_b0_holm"] == "True":
                    mark += "\\ddagger"
                if m in SELECTORS:
                    counts[(c, k, "n")] = counts.get((c, k, "n"), 0) + 1
            return s + ("$^{%s}$" % mark if mark else "")

        rows = family_rows(SELECTORS, lambda m: [cell(m, k) for k in BUDGETS]
                           if any((c, m, k) in summ for k in BUDGETS) else None)
        rows.append("\\addlinespace[2pt]")
        ref = [("IHC-2", "ihc_2", {"2": "2"}), ("IHC-4", "ihc_4", {"4": "4"}),
               ("Girard 2016 (42 genes)", "panel_girard2016", {"50": None}),
               ("Wilkerson 2013 (52 genes)", "panel_wilkerson2013", {"50": None})]
        for i, (label, m, where) in enumerate(ref):
            own = [k for (cc, mm, k) in summ if cc == c and mm == m]
            assert len(own) == 1, (c, m, own)
            cells = [cell(m, own[0]) if k in where else "--" for k in BUDGETS]
            rows.append("%s & %s & %s \\\\" % ("Reference" if i == 0 else "", label, " & ".join(cells)))
        rows.append(" & random-$k$ & %s \\\\" % " & ".join(pm(summ[(c, "random_k", k)]["auroc_mean"],
                                                                  summ[(c, "random_k", k)]["auroc_sd"]) for k in BUDGETS))
        # check: without the marks, every cell equals the cell of the table that was in place
        n_checked = 0
        for line in rows:
            if " & " not in line:
                continue
            parts = [p.strip() for p in line.rstrip()[:-2].split("&")]
            new = [re.sub(r"\$\^\{[^}]*\}\$", "", p) for p in parts[2:]]
            assert old_cells[parts[1]] == new, (c, parts[1], old_cells[parts[1]], new)
            n_checked += len(new)
        for (cc, m, k), t in tests.items():
            if cc == c and t["differs_from_b0_holm"] == "True":
                if float(t["diff_vs_b0"]) < 0:
                    below_b0 += 1
                else:
                    above_b0 += 1
        # 18/11/2026 (repository copy, commit ae10367): the table files of make_tables.py use table* and have no \tablenote;
        # the note is then empty and only the marks are explained.  Cells and numbers are unchanged.
        m_note = re.search(r"\\tablenote\{(.*)\}\s*\\end\{(?:sidewaystable|table)\*\}", old, re.S)
        note_old = m_note.group(1) if m_note else ""
        note = note_old.rstrip()
        if not note:
            note = "Cells: mean $\\pm$ SD of AUROC (LR-L2) over the 5 $\\times$ 5 patient-level cross-validation."
        # 21/11/2026 (QD240): the missing panel genes are listed in the supplementary file itself (Section S3.1)
        note = note.replace("is given in Section~4 of the article", "is given in Section~\\ref{sup:design_data}")
        if "$\\dagger$" not in note:
            note += (" $\\dagger$: above random-$k$ of the same size after Holm correction (one-sided Nadeau--Bengio test,"
                     " family = method $\\times$ its budgets); $\\ddagger$: differs from B0 after Holm correction (two-sided)."
                     " The two signatures have no random reference of their size, and random-$k$ and B0 are the references"
                     " of the tests.")
        cap = re.search(r"\\caption\{(.*)\}", old).group(1)
        text = ("% source: e1_summary.csv (classifier = lr_l2; auroc_mean, auroc_sd); e1_tests_vs_random_b0.csv (beats_random_holm, differs_from_b0_holm)\n"
                "% generated by Scripts/make_supplementary_tables.py from tables_official_2026-11-03 (label [v03-11]); do not edit by hand.\n"
                "% 18/11/2026 (QD173 item 4, QD209): the cells are those of the table generated on 03/11/2026 (mean +- SD, checked cell by cell);\n"
                "% the marks of the tests against random-k and B0 are new.  Table of Online Resource 1 (ESM_1.tex).\n"
                "\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{@CAP@}\n\\label{tab:app_e1_@SHORT@}\n"
                "\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
                "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
                "\\begin{tabular}{llcccccccc}\n\\toprule\n"
                "\\textbf{Family} & \\textbf{Method} & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=5$ & $k=10$ & $k=25$ & $k=50$ \\\\\n\\midrule\n"
                ).replace("@CAP@", cap).replace("@SHORT@", CSHORT[c])
        text += "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\tablenote{" + note + "}\n\\end{sidewaystable*}\n"
        write(out, "tab_app_e1_%s.tex" % CSHORT[c], text)
        NUMBERS["S2_S6_cells_checked_%s" % CSHORT[c]] = n_checked
    NUMBERS["e1_beats_random_counts"] = {"%s|k=%s" % (CNAME[c], k): "%d/%d" % (counts.get((c, k, "beat"), 0), counts.get((c, k, "n"), 0))
                                         for c in COHORTS for k in BUDGETS}
    NUMBERS["e1_differs_from_b0_below"] = below_b0
    NUMBERS["e1_differs_from_b0_above"] = above_b0


# ---------------------------------------------------------------------------------------------------------------
#  2. Secondary metrics of E1 at k in {2, 5, 10} (QD173, item 1)
# ---------------------------------------------------------------------------------------------------------------
METRICS = [("bacc", "balanced accuracy", "Balanced accuracy"), ("mcc", "Matthews correlation coefficient", "MCC"),
           ("logloss", "log-loss", "Log-loss"), ("brier", "Brier score", "Brier score")]


def tables_e1_secondary(official, out):
    summ = {(r["cohort"], r["method"], r["k"]): r for r in rd(os.path.join(official, "e1_summary.csv"))
            if r["classifier"] == "lr_l2"}
    b0k = {c: [k for (cc, m, k) in summ if cc == c and m == "b0"][0] for c in COHORTS}
    best = {}
    for key, long_name, short in METRICS:
        lines = []
        for k in ("2", "5", "10"):
            lines.append("\\multicolumn{7}{l}{\\textit{Budget} $k = %s$} \\\\" % k)
            lines += family_rows(SELECTORS, lambda m: [pm(summ[(c, m, k)][key + "_mean"], summ[(c, m, k)][key + "_sd"])
                                                        if (c, m, k) in summ else "--" for c in COHORTS]
                                 if any((c, m, k) in summ for c in COHORTS) else None)
            lines.append("\\addlinespace[2pt]")
            lines.append("Reference & random-$k$ & %s \\\\" % " & ".join(
                pm(summ[(c, "random_k", k)][key + "_mean"], summ[(c, "random_k", k)][key + "_sd"]) for c in COHORTS))
            if k == "2":
                lines.append(" & IHC-2 & %s \\\\" % " & ".join(
                    pm(summ[(c, "ihc_2", "2")][key + "_mean"], summ[(c, "ihc_2", "2")][key + "_sd"]) for c in COHORTS))
            lines.append("\\midrule")
        lines.append("\\multicolumn{7}{l}{\\textit{Fixed panels at their own size and the full profile}} \\\\")
        for label, m in (("IHC-4 ($k = 4$)", "ihc_4"), ("Girard 2016 (42 genes)", "panel_girard2016"),
                         ("Wilkerson 2013 (52 genes)", "panel_wilkerson2013"), ("B0 (all genes of $G$)", "b0")):
            cells = []
            for c in COHORTS:
                own = [kk for (cc, mm, kk) in summ if cc == c and mm == m]
                cells.append(pm(summ[(c, m, own[0])][key + "_mean"], summ[(c, m, own[0])][key + "_sd"]))
            lines.append("Reference & %s & %s \\\\" % (label, " & ".join(cells)))
        better = "higher is better" if key in ("bacc", "mcc") else "lower is better"
        text = ("%% source: e1_summary.csv (classifier = lr_l2; %s_mean, %s_sd at k = 2, 5, 10; panels and B0 at their own size)\n"
                "%% generated by Scripts/make_supplementary_tables.py from tables_official_2026-11-03 (label [v03-11]); do not edit by hand.\n"
                "%% 18/11/2026 (QD173 item 1, QD209).  Table of Online Resource 1 (ESM_1.tex).\n"
                "\\begin{table*}[p]\n\\centering\n\\scriptsize\n\\caption{E1: %s of LR-L2 at $k \\in \\{2, 5, 10\\}$}\n\\label{tab:app_e1_%s}\n"
                "\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
                "\\begin{adjustbox}{max width=\\textwidth,max totalheight=0.9\\textheight}\n"
                "\\begin{tabular}{llccccc}\n\\toprule\n"
                "\\textbf{Family} & \\textbf{Method} & \\textbf{TCGA} & \\textbf{GSE10245} & \\textbf{GSE30219} & \\textbf{GSE37745} & \\textbf{GSE42127} \\\\\n\\midrule\n"
                % (key, key, long_name, key))
        text += "\n".join(lines) + ("\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
                                    "\\tablenote{%s (%s), mean $\\pm$ SD over the 25 patient-level folds (wrappers: 15 folds on the three GPL570 cohorts, $k \\in \\{5, 10\\}$)."
                                    " $k$-TSP exists for even $k$ only. The decision threshold of the balanced accuracy and of the MCC is chosen on the internal out-of-fold predictions of the training part, as for every panel of the benchmark."
                                    "}\n\\end{table*}\n" % (short, better))
        write(out, "tab_app_e1_%s.tex" % key, text)
    # numbers for the text: log-loss and Brier of B0 against the ten-gene panels
    for c in COHORTS:
        ll = [float(summ[(c, m, "10")]["logloss_mean"]) for m in NONWRAPPER if (c, m, "10") in summ]
        br = [float(summ[(c, m, "10")]["brier_mean"]) for m in NONWRAPPER if (c, m, "10") in summ]
        mc = [float(summ[(c, m, "10")]["mcc_mean"]) for m in NONWRAPPER if (c, m, "10") in summ]
        ba = [float(summ[(c, m, "10")]["bacc_mean"]) for m in NONWRAPPER if (c, m, "10") in summ]
        NUMBERS["e1_k10_secondary_%s" % CSHORT[c]] = {
            "logloss_min": round(min(ll), 4), "logloss_max": round(max(ll), 4), "logloss_b0": round(float(summ[(c, "b0", b0k[c])]["logloss_mean"]), 4),
            "brier_min": round(min(br), 4), "brier_max": round(max(br), 4), "brier_b0": round(float(summ[(c, "b0", b0k[c])]["brier_mean"]), 4),
            "mcc_min": round(min(mc), 4), "mcc_max": round(max(mc), 4), "mcc_b0": round(float(summ[(c, "b0", b0k[c])]["mcc_mean"]), 4),
            "bacc_min": round(min(ba), 4), "bacc_max": round(max(ba), 4), "bacc_b0": round(float(summ[(c, "b0", b0k[c])]["bacc_mean"]), 4),
            "n_selectors": len(ll)}


# ---------------------------------------------------------------------------------------------------------------
#  3. Secondary metrics of E1b at k = 10 and few-shot stability (QD173, items 1 and 2)
# ---------------------------------------------------------------------------------------------------------------
LEVELS = ["5", "10", "20", "40", "pool"]


def tables_e1b(official, out):
    """E1b at k = 10: secondary metrics and Nogueira index at every target size.
    21/11/2026 (QD241): each table is printed as two landscape tables of two cohorts, so that the body is set at 8 pt or more
    (one upright table of the four cohorts was scaled to 5.6 and 6.1 pt)."""
    lc = {(r["cohort"], r["method"], r["k"], r["n_t"]): r for r in rd(os.path.join(official, "e1b_learning_curves.csv"))}
    methods = NONWRAPPER + ["random_k"]
    pairs = [GEO[:2], GEO[2:]]

    def pair_name(pair):
        return "%s and %s" % (CNAME[pair[0]], CNAME[pair[1]])

    def secondary_rows(pair):
        lines = []
        for ci, c in enumerate(pair):
            if ci:
                lines.append("\\midrule")
            first = True
            for m in methods:
                if not any((c, m, "10", lv) in lc for lv in LEVELS):
                    continue
                cells = []
                for key in ("bacc_mean", "logloss_mean", "brier_mean"):
                    for lv in LEVELS:
                        r = lc.get((c, m, "10", lv))
                        cells.append(f3(r[key]) if r is not None and r[key] != "" else "--")
                lines.append("%s & %s & %s \\\\" % (CNAME[c] if first else "", NAME[m], " & ".join(cells)))
                first = False
        return lines

    head = " & ".join("$%s$" % ("n_T{=}" + lv) if lv != "pool" else "pool" for lv in LEVELS)
    text = ("% source: e1b_learning_curves.csv (bacc_mean, logloss_mean, brier_mean at k = 10 for every n_t level; the file stores no MCC and no SD of these metrics)\n"
            "% generated by Scripts/make_supplementary_tables.py from tables_official_2026-11-03 (label [v03-11]); do not edit by hand.\n"
            "% 18/11/2026 (QD173 item 1, QD209); 21/11/2026 (QD241): two landscape tables.  Tables of Online Resource 1 (ESM_1.tex).\n")
    for i, pair in enumerate(pairs):
        label = "tab:app_e1b_secondary" + ("" if i == 0 else "_b")
        note = ("Mean over the 20 runs of a cell (five splits, four nested subsets), by labelled target size $n_T$ per class."
                " ``--'' = level not available for the cohort. With $n_T \\le 10$ the classifier uses the fixed setting $C = 1$ and the threshold 0.5,"
                " so the balanced accuracy at these levels is not tuned. The stored summary of E1b holds the mean of these three metrics; it holds no MCC."
                if i == 0 else "Continuation of Table~\\ref{tab:app_e1b_secondary}, with the same conventions.")
        text += ("\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{E1b: balanced accuracy, log-loss and Brier score at $k = 10$ on %s}\n\\label{%s}\n"
                 "\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
                 "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
                 "\\begin{tabular}{ll" % (pair_name(pair), label) + "ccccc" * 3 + "}\n\\toprule\n"
                 " & & \\multicolumn{5}{c}{\\textbf{Balanced accuracy}} & \\multicolumn{5}{c}{\\textbf{Log-loss}} & \\multicolumn{5}{c}{\\textbf{Brier score}} \\\\\n"
                 "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\\cmidrule(lr){13-17}\n"
                 "\\textbf{Cohort} & \\textbf{Method} & " + " & ".join([head] * 3) + " \\\\\n\\midrule\n"
                 + "\n".join(secondary_rows(pair)) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
                 "\\tablenote{" + note + "}\n\\end{sidewaystable*}\n")
    write(out, "tab_app_e1b_secondary.tex", text)

    stab = {(r["cohort"], r["method"], r["k"], r["n_t"]): r for r in rd(os.path.join(official, "e1b_stability.csv"))}

    def stability_rows(pair):
        lines = []
        for ci, c in enumerate(pair):
            if ci:
                lines.append("\\midrule")
            first = True
            for m in NONWRAPPER:
                cells = []
                for lv in LEVELS:
                    r = stab.get((c, m, "10", lv))
                    if r is None or r["nogueira"] == "":
                        cells.append("--")
                    else:
                        cells.append("%s [%s; %s]" % (f3(r["nogueira"]), f3(r["nogueira_ci_low"]), f3(r["nogueira_ci_high"])))
                lines.append("%s & %s & %s \\\\" % (CNAME[c] if first else "", NAME[m], " & ".join(cells)))
                first = False
        return lines

    text = ("% source: e1b_stability.csv (nogueira, nogueira_ci_low, nogueira_ci_high at k = 10 for every n_t level)\n"
            "% generated by Scripts/make_supplementary_tables.py from tables_official_2026-11-03 (label [v03-11]); do not edit by hand.\n"
            "% 18/11/2026 (QD173 item 2, QD209); 21/11/2026 (QD241): two landscape tables.  Tables of Online Resource 1 (ESM_1.tex).\n")
    for i, pair in enumerate(pairs):
        label = "tab:app_e1b_stability" + ("" if i == 0 else "_b")
        note = ("Nogueira index $\\hat{\\Phi}$ over the 20 panels of a cell (five splits, four nested subsets), with its asymptotic interval in brackets."
                " The interval is descriptive, because the nested subsets of one split overlap (at the pool level the four subsets of a split are the same patients)."
                " ``--'' = level not available for the cohort."
                if i == 0 else "Continuation of Table~\\ref{tab:app_e1b_stability}, with the same conventions.")
        text += ("\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{E1b: Nogueira index of the few-shot panels at $k = 10$ on %s}\n\\label{%s}\n"
                 "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
                 "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
                 "\\begin{tabular}{llccccc}\n\\toprule\n"
                 "\\textbf{Cohort} & \\textbf{Method} & $n_T = 5$ & $n_T = 10$ & $n_T = 20$ & $n_T = 40$ & pool \\\\\n\\midrule\n" % (pair_name(pair), label)
                 + "\n".join(stability_rows(pair)) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
                 "\\tablenote{" + note + "}\n\\end{sidewaystable*}\n")
    write(out, "tab_app_e1b_stability.tex", text)
    for c in GEO:
        v5 = [float(stab[(c, m, "10", "5")]["nogueira"]) for m in NONWRAPPER]
        vp = [float(stab[(c, m, "10", "pool")]["nogueira"]) for m in NONWRAPPER]
        NUMBERS["e1b_nogueira_k10_%s" % c] = {"n_t5_n_le_0.05": sum(v <= 0.05 for v in v5), "n_t5_min": round(min(v5), 3), "n_t5_max": round(max(v5), 3),
                                               "pool_min": round(min(vp), 3), "pool_max": round(max(vp), 3)}


# ---------------------------------------------------------------------------------------------------------------
#  4. Full ablation without the IHC genes (QD173, item 3)
# ---------------------------------------------------------------------------------------------------------------
def table_ablation(official, out):
    ab = {(r["cohort"], r["method"], r["k"]): r for r in rd(os.path.join(official, "e1_ablation_noihc.csv"))}
    lines, n_excl, n_cells, extreme = [], 0, 0, []
    for ci, c in enumerate(COHORTS):
        if ci:
            lines.append("\\midrule")
        first = True
        for m in NONWRAPPER:
            cells = []
            for k in BUDGETS:
                r = ab.get((c, m, k))
                if r is None:
                    cells.append("--")
                    continue
                excl = float(r["delta_ci_low"]) > 0 or float(r["delta_ci_high"]) < 0
                n_cells += 1
                n_excl += excl
                extreme.append((float(r["delta_mean"]), CNAME[c], m, k))
                cells.append(sgn(r["delta_mean"]) + ("$^{*}$" if excl else ""))
            lines.append("%s & %s & %s \\\\" % (CNAME[c] if first else "", NAME[m], " & ".join(cells)))
            first = False
        r = ab[(c, "b0", "all")]
        lines.append(" & B0 (all genes) & \\multicolumn{8}{c}{%s [%s; %s]} \\\\" % (sgn(r["delta_mean"]), sgn(r["delta_ci_low"]), sgn(r["delta_ci_high"])))
    text = ("% source: e1_ablation_noihc.csv (delta_mean, delta_ci_low, delta_ci_high for every cohort, method and budget; B0 at k = all)\n"
            "% generated by Scripts/make_supplementary_tables.py from tables_official_2026-11-03 (label [v03-11]); do not edit by hand.\n"
            "% 18/11/2026 (QD173 item 3, QD209).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{table*}[p]\n\\centering\n\\scriptsize\n\\caption{Ablation without the four IHC marker genes at every budget}\n\\label{tab:app_ablation_full}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
            "\\begin{adjustbox}{max width=\\textwidth,max totalheight=0.9\\textheight}\n"
            "\\begin{tabular}{llcccccccc}\n\\toprule\n"
            "\\textbf{Cohort} & \\textbf{Method} & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=5$ & $k=10$ & $k=25$ & $k=50$ \\\\\n\\midrule\n")
    text += "\n".join(lines) + ("\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
                                "\\tablenote{Change of AUROC when NKX2-1, TP63, NAPSA and KRT5 are removed from the gene space before selection"
                                " (without the four genes minus full gene space), paired over the 25 folds. $^{*}$ marks a cell whose 95\\% Nadeau--Bengio interval excludes zero;"
                                " the intervals are not corrected for the number of cells. $k$-TSP exists for even $k$ only. The row B0 gives the change of the full-profile model with its interval.}\n\\end{table*}\n")
    write(out, "tab_app_ablation_full.tex", text)
    extreme.sort()
    NUMBERS["ablation_full"] = {"cells": n_cells, "cells_interval_excludes_zero": n_excl,
                                "five_largest_losses": [(round(v, 4), c, m, k) for v, c, m, k in extreme[:5]],
                                "five_largest_gains": [(round(v, 4), c, m, k) for v, c, m, k in extreme[-5:]],
                                "n_abs_gt_0.03": sum(abs(v) > 0.03 for v, _, _, _ in extreme),
                                "n_abs_gt_0.03_k_ge_2": sum(abs(v) > 0.03 and k != "1" for v, _, _, k in extreme),
                                "excluded_zero_cells": [(CNAME[c], m, k, round(float(r["delta_mean"]), 4)) for (c, m, k), r in ab.items()
                                                        if m in NONWRAPPER and (float(r["delta_ci_low"]) > 0 or float(r["delta_ci_high"]) < 0)]}


# ---------------------------------------------------------------------------------------------------------------
#  5. Sensitivity of k* to the margin and to the pooling (QD173, item 5)
# ---------------------------------------------------------------------------------------------------------------
SEQ = ["50", "25", "10", "5", "4", "3", "2", "1"]


def fixed_sequence(passes):
    """passes: dict budget -> bool for the budgets a method has; returns k* (str) or None."""
    kstar = None
    for k in SEQ:
        if k not in passes:
            continue
        if not passes[k]:
            break
        kstar = k
    return kstar


def table_kstar(official, out):
    byk = rd(os.path.join(official, "e1_kstar_pooled_by_k.csv"))
    pooled = {r["method"]: r for r in rd(os.path.join(official, "e1_kstar_pooled.csv"))}
    tests = {(r["cohort"], r["method"], r["k"]): r for r in rd(os.path.join(official, "e1_tests_vs_random_b0.csv"))}
    z = stats.norm.ppf(0.95)
    t2 = stats.t.ppf(0.95, 2)
    rows, max_dev = {}, 0.0
    for r in byk:
        m, k = r["method"], r["k"]
        d, se = [], []
        for c in GPL570:
            t = tests[(c, m, k)]
            J = int(float(t["n_pairs_b0"]))
            d.append(float(t["diff_vs_b0"]))
            se.append((float(t["diff_b0_ci_high"]) - float(t["diff_b0_ci_low"])) / (2 * stats.t.ppf(0.975, J - 1)))
        w = [1 / s ** 2 for s in se]
        W = sum(w)
        d_iv = sum(wi * di for wi, di in zip(w, d)) / W
        se_iv = math.sqrt(1 / W)
        d_eq = sum(d) / 3
        se_eq = math.sqrt(sum(s ** 2 for s in se)) / 3
        for name, v in (("diff_ivw", d_iv), ("lb_ivw", d_iv - z * se_iv), ("diff_eq", d_eq), ("se_eq", se_eq), ("lb_eq", d_eq - z * se_eq)):
            max_dev = max(max_dev, abs(v - float(r[name])))
        Q = sum(wi * (di - d_iv) ** 2 for wi, di in zip(w, d))
        q = Q / 2.0                                    # Q / (K - 1), K = 3 cohorts
        lb_hk = d_iv - t2 * math.sqrt(q) * se_iv        # Hartung-Knapp variance (tau^2 = 0 here, because Q < K - 1)
        lb_hk_mod = d_iv - t2 * math.sqrt(max(1.0, q)) * se_iv  # variance not allowed to fall below the fixed-effect one
        rows.setdefault(m, {})[k] = {"lb_ivw": float(r["lb_ivw"]), "lb_eq": float(r["lb_eq"]), "lb_hk": lb_hk, "lb_hk_mod": lb_hk_mod,
                                     "q": q, "I2": float(r["I2"]), "d_iv": d_iv, "se_iv": se_iv}
    assert max_dev < 1e-9, max_dev
    NUMBERS["kstar_reconstruction_max_abs_deviation"] = max_dev
    NUMBERS["kstar_rows_reconstructed"] = len(byk)
    NUMBERS["kstar_q_max"] = round(max(v["q"] for m in rows for v in rows[m].values()), 4)
    NUMBERS["kstar_t_quantile_2df"] = round(t2, 3)
    cells = [v for m in rows for v in rows[m].values()]
    NUMBERS["kstar_hk_cells"] = {"n": len(cells), "hk_bound_above_ivw_bound": sum(v["lb_hk"] > v["lb_ivw"] for v in cells),
                                 "hk_bound_below_ivw_bound": sum(v["lb_hk"] < v["lb_ivw"] for v in cells),
                                 "q_below_0.317": int(sum(v["q"] < (z / t2) ** 2 for v in cells)), "q_median": round(statistics.median(v["q"] for v in cells), 4)}
    order = SELECTORS + ["ihc_2", "ihc_4"]
    res = {}
    for m in order:
        ks = rows[m]
        res[m] = {
            "ivw_02": fixed_sequence({k: v["lb_ivw"] > -0.02 for k, v in ks.items()}),
            "eq_02": fixed_sequence({k: v["lb_eq"] > -0.02 for k, v in ks.items()}),
            "both_02": fixed_sequence({k: v["lb_ivw"] > -0.02 and v["lb_eq"] > -0.02 for k, v in ks.items()}),
            "ivw_01": fixed_sequence({k: v["lb_ivw"] > -0.01 for k, v in ks.items()}),
            "eq_01": fixed_sequence({k: v["lb_eq"] > -0.01 for k, v in ks.items()}),
            "both_01": fixed_sequence({k: v["lb_ivw"] > -0.01 and v["lb_eq"] > -0.01 for k, v in ks.items()}),
            "hk_02": fixed_sequence({k: v["lb_hk"] > -0.02 for k, v in ks.items()}),
            "hk_mod_02": fixed_sequence({k: v["lb_hk_mod"] > -0.02 for k, v in ks.items()}),
        }
        # the stored k* of the main analysis is reproduced by the rule applied to the stored bounds
        st_iv = pooled[m]["k_star_ivw"]
        st_both = pooled[m]["k_star_both"]
        assert (res[m]["ivw_02"] or "") == (str(int(float(st_iv))) if st_iv else ""), (m, res[m]["ivw_02"], st_iv)
        assert (res[m]["both_02"] or "") == (str(int(float(st_both))) if st_both else ""), (m, res[m]["both_02"], st_both)

    def kk(v):
        return v if v else "--"

    def lb(m, key):
        ks = rows[m]
        k = "10" if "10" in ks else sorted(ks, key=int)[-1]
        return sgn(ks[k][key]) + ("" if k == "10" else "$^{a}$")

    lines = []
    blocks = FAMILIES + [("Reference", ["ihc_2", "ihc_4"])]
    for bi, (fam, ms) in enumerate(blocks):
        if bi:
            lines.append("\\addlinespace[2pt]")
        for i, m in enumerate(ms):
            r = res[m]
            lines.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
                fam if i == 0 else "", NAME[m], kk(r["ivw_02"]), kk(r["both_02"]), kk(r["ivw_01"]), kk(r["both_01"]),
                kk(r["hk_02"]), kk(r["hk_mod_02"]), lb(m, "lb_ivw"), lb(m, "lb_eq"), lb(m, "lb_hk"), lb(m, "lb_hk_mod")))
    text = ("% source: e1_kstar_pooled_by_k.csv (lb_ivw, lb_eq by method and k), e1_kstar_pooled.csv (k_star_ivw, k_star_both), e1_tests_vs_random_b0.csv (diff_vs_b0, diff_b0_ci_low, diff_b0_ci_high, n_pairs_b0 of the three GPL570 cohorts)\n"
            "% generated by Scripts/make_supplementary_tables.py from tables_official_2026-11-03 (label [v03-11]); do not edit by hand.\n"
            "% 18/11/2026 (QD173 item 5, QD210).  Per-cohort standard errors are recovered from the two-sided 95% Nadeau-Bengio intervals (t quantile with n_pairs - 1 degrees of freedom);\n"
            "% the pooled values recomputed from them equal the stored values of e1_kstar_pooled_by_k.csv to 1e-9 in all 92 rows.  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{Sensitivity of the pooled minimum sufficient budget $k^{*}$ to the margin and to the pooling}\n\\label{tab:app_kstar_sens}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
            "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
            "\\begin{tabular}{llcccccccccc}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{$\\boldsymbol{k^{*}}$\\textbf{, margin 0.02}} & \\multicolumn{2}{c}{$\\boldsymbol{k^{*}}$\\textbf{, margin 0.01}} & \\multicolumn{2}{c}{$\\boldsymbol{k^{*}}$\\textbf{, HK, margin 0.02}} & \\multicolumn{4}{c}{\\textbf{Lower bound of the pooled difference at} $\\boldsymbol{k = 10}$} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-12}\n"
            "\\textbf{Family} & \\textbf{Method} & IVW & both & IVW & both & HK & HK, bounded & IVW & equal weights & HK & HK, bounded \\\\\n\\midrule\n")
    text += "\n".join(lines) + ("\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
                                "\\tablenote{Pooled difference AUROC$(\\text{panel}_k) - \\text{AUROC}(\\text{B0})$ over the three GPL570 cohorts, tested along the fixed sequence of the article."
                                " IVW = inverse-variance weights, both = IVW and equal weights (the two columns at margin 0.02 are those of Table~\\artnum{tab:kstar} of the article)."
                                " HK = Hartung--Knapp adjustment of the IVW estimate: the variance is multiplied by $q = Q/(K - 1)$, with $Q$ the heterogeneity statistic of the $K = 3$ cohort differences,"
                                " and the one-sided 95\\%% bound uses the $t$ quantile with two degrees of freedom (%.3f) in place of the normal quantile (1.645)."
                                " ``HK, bounded'' does not let the variance fall below the fixed-effect variance ($q$ replaced by $\\max(1, q)$); $q$ is below 1 for every method and budget (largest value %.2f), so this bound equals the IVW estimate minus %.3f standard errors."
                                " ``--'' = no tested budget passes. $^{a}$ fixed panel, tested once at its own size (IHC-2: $k = 2$, IHC-4: $k = 4$).}\n\\end{sidewaystable*}\n"
                                % (t2, NUMBERS["kstar_q_max"], t2))
    write(out, "tab_app_kstar_sens.tex", text)
    NUMBERS["kstar_sensitivity"] = res
    NUMBERS["kstar_supported_margin_0.01_both"] = [m for m in order if res[m]["both_01"] and int(res[m]["both_01"]) <= 10]
    NUMBERS["kstar_supported_margin_0.01_ivw"] = [m for m in order if res[m]["ivw_01"] and int(res[m]["ivw_01"]) <= 10]
    NUMBERS["kstar_supported_hk"] = [m for m in order if res[m]["hk_02"] and int(res[m]["hk_02"]) <= 10]
    NUMBERS["kstar_supported_hk_bounded"] = [m for m in order if res[m]["hk_mod_02"] and int(res[m]["hk_mod_02"]) <= 10]
    NUMBERS["kstar_lb_k10"] = {m: {k: round(v, 4) for k, v in rows[m]["10"].items() if k.startswith("lb")} for m in SELECTORS if "10" in rows[m]}
    return res


# ---------------------------------------------------------------------------------------------------------------
#  6. SA1: variants of the reference model B0 (article table)
# ---------------------------------------------------------------------------------------------------------------
VARIANTS = [("b0", "B0 (reference of the benchmark)", "all genes of $G$", "$10^{-2}$ to $10$"),
            ("b0_full_wide", "wide grid", "all genes of $G$", "$10^{-4}$ to $10$"),
            ("b0_p5000", "pre-filtered", "5,000", "$10^{-2}$ to $10$"),
            ("b0_p5000_wide", "pre-filtered, wide grid", "5,000", "$10^{-4}$ to $10$")]


def table_sa1(extra, official, out):
    var = {(r["cohort"], r["variant"]): r for r in rd(os.path.join(extra, "B4_tables", "b4_b0_variants.csv"))}
    ks = {r["method"]: r for r in rd(os.path.join(extra, "B4_tables", "b4_kstar_sensitivity.csv"))}
    pooled = {r["method"]: r for r in rd(os.path.join(official, "e1_kstar_pooled.csv"))}
    gate = rd(os.path.join(extra, "B4_tables", "b4_gate_b0.csv"))
    assert all(g["pass"] == "True" and float(g["max_abs_diff_auroc"]) == 0.0 for g in gate)
    a = []
    for c in COHORTS:
        a.append("%s & %s \\\\" % (CNAME[c], " & ".join(pm(var[(c, v)]["auroc_mean"], var[(c, v)]["auroc_sd"]) for v, _, _, _ in VARIANTS)))

    def kk(x):
        return str(int(float(x))) if x not in ("", None) else "--"

    b = []
    blocks = FAMILIES + [("Reference", ["ihc_2", "ihc_4"])]
    changed = []
    for bi, (fam, ms) in enumerate(blocks):
        if bi:
            b.append("\\addlinespace[2pt]")
        for i, m in enumerate(ms):
            r = ks[m]
            # the columns "against B0" of the sensitivity table are the published values
            assert kk(r["k_star_ivw_b0"]) == kk(pooled[m]["k_star_ivw"]) and kk(r["k_star_both_b0"]) == kk(pooled[m]["k_star_both"]), m
            if (kk(r["k_star_ivw_b0"]), kk(r["k_star_both_b0"])) != (kk(r["k_star_ivw_b0_p5000_wide"]), kk(r["k_star_both_b0_p5000_wide"])):
                changed.append(m)
            b.append("%s & %s & %s & %s & %s & %s \\\\" % (fam if i == 0 else "", NAME[m], kk(r["k_star_ivw_b0"]), kk(r["k_star_both_b0"]),
                                                             kk(r["k_star_ivw_b0_p5000_wide"]), kk(r["k_star_both_b0_p5000_wide"])))
    text = ("% source: B4_tables/b4_b0_variants.csv (auroc_mean, auroc_sd by cohort and variant), B4_tables/b4_kstar_sensitivity.csv (k_star_ivw_b0, k_star_both_b0, k_star_ivw_b0_p5000_wide, k_star_both_b0_p5000_wide), B4_tables/b4_gate_b0.csv (the rerun of B0 equals the published folds, max_abs_diff_auroc = 0), tag survey-v3-dmkd.2\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD200, QD208).  Table of Online Resource 1 since 21/11/2026 (QD240; until then tab_sens_reference.tex, a table of the article).\n"
            "\\begin{table*}[tp]\n\\centering\n\\scriptsize\n\\caption{SA1: the reference model under a wider grid of $C$ and on the pre-filtered genes, and the resulting $k^{*}$}\n\\label{tab:app_sens_reference}\n"
            "\\setlength{\\tabcolsep}{3.5pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n"
            "\\multicolumn{5}{l}{\\textit{(a) AUROC of the reference model without selection}} \\\\\n"
            " & \\textbf{B0} & \\textbf{Wide grid} & \\textbf{Pre-filtered} & \\textbf{Pre-filtered, wide grid} \\\\\n"
            "Genes & all genes of $G$ & all genes of $G$ & 5,000 & 5,000 \\\\\n"
            "Grid of $C$ & $10^{-2}$ to $10$ & $10^{-4}$ to $10$ & $10^{-2}$ to $10$ & $10^{-4}$ to $10$ \\\\\n\\midrule\n"
            + "\n".join(a) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\n\\vspace{6pt}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{llcccc}\n\\toprule\n"
            "\\multicolumn{6}{l}{\\textit{(b) Pooled} $k^{*}$ \\textit{on the three GPL570 cohorts}} \\\\\n"
            " & & \\multicolumn{2}{c}{\\textbf{Against B0}} & \\multicolumn{2}{c}{\\textbf{Against the pre-filtered, wide-grid variant}} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
            "\\textbf{Family} & \\textbf{Method} & $\\boldsymbol{k^{*}_{\\mathrm{IVW}}}$ & $\\boldsymbol{k^{*}_{\\mathrm{both}}}$ & $\\boldsymbol{k^{*}_{\\mathrm{IVW}}}$ & $\\boldsymbol{k^{*}_{\\mathrm{both}}}$ \\\\\n\\midrule\n"
            + "\n".join(b) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{(a) Mean $\\pm$ SD over the 25 folds of E1. All variants are LR-L2 with $C$ chosen by three-fold internal cross-validation; the wide grid adds $10^{-4}$ and $10^{-3}$, and the pre-filtered variants use the 5,000 genes with the largest variance on the training patients. The first column is B0 as used in every other table. (b) $k^{*}$ as in Table~\\artnum{tab:kstar} of the article, with B0 replaced by the variant that has the highest mean AUROC on the three GPL570 cohorts; ``--'' = no tested budget satisfies the bound.}\n"
            "\\end{table*}\n")
    write(out, "tab_app_sens_reference.tex", text)
    NUMBERS["sa1_b0_variants"] = {CNAME[c]: {v: round(float(var[(c, v)]["auroc_mean"]), 4) for v, _, _, _ in VARIANTS} for c in COHORTS}
    NUMBERS["sa1_max_abs_shift_other_cohorts"] = round(max(abs(float(var[(c, v)]["auroc_mean"]) - float(var[(c, "b0")]["auroc_mean"]))
                                                           for c in COHORTS if c != "gse42127" for v, _, _, _ in VARIANTS), 4)
    NUMBERS["sa1_mean_gpl570"] = {v: round(statistics.mean(float(var[(c, v)]["auroc_mean"]) for c in GPL570), 4) for v, _, _, _ in VARIANTS}
    NUMBERS["sa1_C_mode"] = {CNAME[c]: {v: var[(c, v)]["C_mode"] for v, _, _, _ in VARIANTS} for c in COHORTS}
    NUMBERS["sa1_kstar_changed"] = changed
    NUMBERS["sa1_kstar"] = {m: [kk(ks[m][x]) for x in ("k_star_ivw_b0", "k_star_both_b0", "k_star_ivw_b0_p5000_wide", "k_star_both_b0_p5000_wide")]
                            for m in SELECTORS + ["ihc_2", "ihc_4"]}
    by0 = {(r["method"], r["k"]): r for r in rd(os.path.join(official, "e1_kstar_pooled_by_k.csv"))}
    by1 = {(r["method"], r["k"]): r for r in rd(os.path.join(extra, "B4_tables", "b4_kstar_sensitivity_by_k.csv"))}
    NUMBERS["sa1_eq_k10"] = {m: {"lb_eq_b0": round(float(by0[(m, k)]["lb_eq"]), 4), "lb_eq_wide": round(float(by1[(m, k)]["lb_eq"]), 4),
                                 "diff_eq_b0": round(float(by0[(m, k)]["diff_eq"]), 4), "diff_eq_wide": round(float(by1[(m, k)]["diff_eq"]), 4),
                                 "lb_ivw_b0": round(float(by0[(m, k)]["lb_ivw"]), 4), "lb_ivw_wide": round(float(by1[(m, k)]["lb_ivw"]), 4), "k": k}
                             for m, k in (("l1_lr", "10"), ("stability_selection", "10"), ("svm_rfe", "10"), ("ktsp", "10"), ("ihc_4", "4"))}
    NUMBERS["sa1_supported_both_wide"] = [m for m in SELECTORS + ["ihc_2", "ihc_4"]
                                          if ks[m]["k_star_both_b0_p5000_wide"] and float(ks[m]["k_star_both_b0_p5000_wide"]) <= 10]
    NUMBERS["sa1_ivw_le10_wide"] = [m for m in SELECTORS + ["ihc_2", "ihc_4"]
                                    if ks[m]["k_star_ivw_b0_p5000_wide"] and float(ks[m]["k_star_ivw_b0_p5000_wide"]) <= 10]


# ---------------------------------------------------------------------------------------------------------------
#  7. SA2: published signatures cut to ten genes
# ---------------------------------------------------------------------------------------------------------------
SIG = [("sig10_girard2016", "Girard 2016, ten genes"), ("sig10_wilkerson2013", "Wilkerson 2013, ten genes")]


def tables_sa2(extra, out):
    comp = {(r["signature"], r["cohort"], r["reference"]): r for r in rd(os.path.join(extra, "B5_tables", "b5_comparisons.csv"))}
    ni = {(r["signature"], r["scope"]): r for r in rd(os.path.join(extra, "B5_tables", "b5_noninferiority_b0.csv"))}
    lines = ["\\multicolumn{7}{l}{\\textit{(a) AUROC, mean over 25 folds}} \\\\"]
    for s, label in SIG:
        lines.append(" & %s & %s \\\\" % (label, " & ".join(f3(comp[(s, c, "b0")]["auroc_signature"]) for c in COHORTS)))
    s0 = SIG[0][0]
    lines.append(" & best selector at $k = 10$ & %s \\\\" % " & ".join(
        "%s (%s)" % (f3(comp[(s0, c, "best_selector_k10")]["auroc_reference"]), NAME[comp[(s0, c, "best_selector_k10")]["reference_method"]]) for c in COHORTS))
    lines.append(" & IHC-4 & %s \\\\" % " & ".join(f3(comp[(s0, c, "ihc_4")]["auroc_reference"]) for c in COHORTS))
    lines.append(" & B0 & %s \\\\" % " & ".join(f3(comp[(s0, c, "b0")]["auroc_reference"]) for c in COHORTS))
    lines.append("\\midrule")
    lines.append("\\multicolumn{7}{l}{\\textit{(b) Difference, signature minus reference, with its 95\\% interval beneath}} \\\\")
    sig_cells = []
    for s, label in SIG:
        for j, (ref, rl) in enumerate((("best_selector_k10", "best selector at $k = 10$"), ("ihc_4", "IHC-4"), ("b0", "B0"))):
            cells = []
            for c in COHORTS:
                r = comp[(s, c, ref)]
                star = "$^{*}$" if float(r["p_holm"]) < 0.05 else ""
                if star:
                    sig_cells.append((s, CNAME[c], ref, round(float(r["diff"]), 4), round(float(r["p_holm"]), 4)))
                # 21/11/2026 (QD241): the interval stands beneath the difference, so that the table is set at 8 pt (one line: 6.3 pt)
                cells.append("\\begin{tabular}[c]{@{}c@{}}$%+.3f$%s\\\\{}[$%+.3f$; $%+.3f$]\\end{tabular}" % (float(r["diff"]), star, float(r["ci_low"]), float(r["ci_high"])))
            lines.append("%s & minus %s & %s \\\\" % (label if j == 0 else "", rl, " & ".join(cells)))
    text = ("% source: B5_tables/b5_comparisons.csv (auroc_signature, auroc_reference, reference_method, diff, ci_low, ci_high, p_holm), tag survey-v3-dmkd.2\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD208).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{SA2: the two published signatures cut to ten genes, against the best ten-gene selector, IHC-4 and B0}\n\\label{tab:app_sig10}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.15}\n"
            "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
            "\\begin{tabular}{llccccc}\n\\toprule\n"
            " & & \\textbf{TCGA} & \\textbf{GSE10245} & \\textbf{GSE30219} & \\textbf{GSE37745} & \\textbf{GSE42127} \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Inside each training fold, the ten genes of a signature with the largest F statistic are kept and classified with LR-L2. The best selector at $k = 10$ is the data-driven method with the highest mean AUROC on the cohort in E1 (name in parentheses). Intervals and tests are those of the paired Nadeau--Bengio corrected $t$-test over the 25 folds; $^{*}$ = $p < 0.05$ after Holm correction over the five cohorts.}\n"
            "\\end{sidewaystable*}\n")
    write(out, "tab_app_sig10.tex", text)
    NUMBERS["sa2_significant_cells"] = sig_cells
    NUMBERS["sa2_pooled"] = {s: {k: (round(float(v), 4) if k not in ("non_inferior_ivw", "non_inferior_both") else v)
                                 for k, v in ni[(s, "GPL570 pooled")].items() if k not in ("signature", "scope")} for s, _ in SIG}
    NUMBERS["sa2_per_cohort_lb"] = {s: {CNAME[c]: [round(float(ni[(s, c)]["diff_ivw"]), 4), round(float(ni[(s, c)]["lb_ivw"]), 4), ni[(s, c)]["non_inferior_ivw"]] for c in COHORTS} for s, _ in SIG}
    NUMBERS["sa2_auroc"] = {s: {CNAME[c]: round(float(comp[(s, c, "b0")]["auroc_signature"]), 4) for c in COHORTS} for s, _ in SIG}
    NUMBERS["sa2_diff_vs_best"] = {s: {CNAME[c]: round(float(comp[(s, c, "best_selector_k10")]["diff"]), 4) for c in COHORTS} for s, _ in SIG}
    NUMBERS["sa2_diff_vs_ihc4"] = {s: {CNAME[c]: round(float(comp[(s, c, "ihc_4")]["diff"]), 4) for c in COHORTS} for s, _ in SIG}
    NUMBERS["sa2_diff_vs_b0"] = {s: {CNAME[c]: round(float(comp[(s, c, "b0")]["diff"]), 4) for c in COHORTS} for s, _ in SIG}

    fs = {(r["cohort"], r["role"], r["n_t"]): r for r in rd(os.path.join(extra, "B5_tables", "b5_e1b_k10.csv"))}
    lines = []
    roles = [("sig10_girard2016", lambda c: "Girard 2016, ten genes"), ("sig10_wilkerson2013", lambda c: "Wilkerson 2013, ten genes"),
             ("ihc_4", lambda c: "IHC-4"), ("best_selector_k10", lambda c: "%s (best selector of E1 at $k = 10$)" % NAME[fs[(c, "best_selector_k10", "5")]["method"]])]
    for ci, c in enumerate(GEO):
        if ci:
            lines.append("\\midrule")
        for i, (role, lab) in enumerate(roles):
            cells = [f3(fs[(c, role, lv)]["auroc_mean"]) if (c, role, lv) in fs else "--" for lv in LEVELS]
            lines.append("%s & %s & %s \\\\" % (CNAME[c] if i == 0 else "", lab(c), " & ".join(cells)))
    text = ("% source: B5_tables/b5_e1b_k10.csv (auroc_mean by cohort, role and n_t; 20 runs per cell), tag survey-v3-dmkd.2\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD208).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{table*}[tp]\n\\centering\n\\scriptsize\n\\caption{SA2 in the few-shot regime: AUROC of the ten-gene signatures at $k = 10$}\n\\label{tab:app_sig10_fewshot}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{llccccc}\n\\toprule\n"
            "\\textbf{Cohort} & \\textbf{Panel} & $n_T = 5$ & $n_T = 10$ & $n_T = 20$ & $n_T = 40$ & pool \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Mean AUROC over the 20 runs of a cell of E1b. The ten genes of a signature are chosen by the F statistic on the labelled subset $L_T$. The last row of each cohort is the data-driven selector that leads the cohort in E1 at $k = 10$, run on the same subsets; IHC-4 uses its four genes. ``--'' = level not available for the cohort.}\n"
            "\\end{table*}\n")
    write(out, "tab_app_sig10_fewshot.tex", text)
    NUMBERS["sa2_fewshot_nt5"] = {CNAME[c]: {role: round(float(fs[(c, role, "5")]["auroc_mean"]), 4) for role, _ in roles} for c in GEO}


# ---------------------------------------------------------------------------------------------------------------
#  8. SA3: k-TSP pairs with the voting rule
# ---------------------------------------------------------------------------------------------------------------
def tables_sa3(extra, out):
    e1 = {(r["cohort"], r["k"]): r for r in rd(os.path.join(extra, "B2_tables", "b2_e1_delta.csv"))}
    zs = {(r["cohort"], r["k"]): r for r in rd(os.path.join(extra, "B2_tables", "b2_source_only_reversal.csv"))}
    e1b = rd(os.path.join(extra, "B2_tables", "b2_e1b_delta.csv"))
    e7 = rd(os.path.join(extra, "B2_tables", "b2_e7_delta.csv"))
    lines = []
    for ci, c in enumerate(COHORTS):
        if ci:
            lines.append("\\addlinespace[2pt]")
        for i, k in enumerate(("2", "4", "10")):
            r = e1[(c, k)]
            z = zs.get((c, k))
            lines.append("%s & %s & %s & %s & $%+.3f \\pm %.3f$ & %s & %s \\\\" % (
                CNAME[c] if i == 0 else "", k, f3(r["auroc_vote"]), f3(r["auroc_lr"]), float(r["delta_mean"]), float(r["delta_sd"]),
                f3(z["auroc_vote"]) if z else "--", f3(z["auroc_lr"]) if z else "--"))
    text = ("% source: B2_tables/b2_e1_delta.csv (auroc_vote, auroc_lr, delta_mean, delta_sd; 25 folds), B2_tables/b2_source_only_reversal.csv (auroc_vote, auroc_lr; five test sets), tag survey-v3-dmkd.2\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD208).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{table*}[tp]\n\\centering\n\\scriptsize\n\\caption{SA3: the pairs of $k$-TSP classified by their voting rule and by LR-L2}\n\\label{tab:app_ktsp_vote}\n"
            "\\setlength{\\tabcolsep}{5pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{lcccccc}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{\\textbf{E1, pairs selected in the training fold}} & \\multicolumn{2}{c}{\\textbf{Zero-shot, pairs selected on TCGA}} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
            "\\textbf{Cohort} & $\\boldsymbol{k}$ & voting & LR-L2 & voting $-$ LR-L2 & voting & LR-L2 \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{AUROC of the $k/2$ pairs selected by $k$-TSP. Voting: the direction of each pair is learned on the training data and the score of a patient is the fraction of pairs that vote for LUSC, without standardisation. LR-L2: the classifier of the benchmark on the same genes. E1: mean over the 25 folds, difference as mean $\\pm$ SD of the paired fold differences. Zero-shot: pairs and directions (or the classifier, after the within-sample normalisation of E7) learned on TCGA and applied to the five test sets of the target cohort.}\n"
            "\\end{table*}\n")
    write(out, "tab_app_ktsp_vote.tex", text)

    fs = {(r["cohort"], r["n_t"], r["k"]): r for r in e1b}
    so = {(r["cohort"], r["n_t"], r["k"]): r for r in e7 if r["strategy"] == "select_on_source"}
    lines = []
    for ci, c in enumerate(GEO):
        if ci:
            lines.append("\\addlinespace[2pt]")
        first = True
        for lv in LEVELS:
            if not any((c, lv, k) in fs for k in ("2", "4", "10")):
                continue
            cells = []
            for src in (fs, so):
                for k in ("2", "4", "10"):
                    r = src.get((c, lv, k))
                    cells.append("%s ($%+.3f$)" % (f3(r["auroc_vote"]), float(r["delta_mean"])) if r else "--")
            lines.append("%s & %s & %s \\\\" % (CNAME[c] if first else "", lv, " & ".join(cells)))
            first = False
    text = ("% source: B2_tables/b2_e1b_delta.csv and B2_tables/b2_e7_delta.csv (strategy = select_on_source): auroc_vote, delta_mean; 20 runs per cell, tag survey-v3-dmkd.2\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD208).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{table*}[tp]\n\\centering\n\\scriptsize\n\\caption{SA3 in the few-shot regime: AUROC of the voting rule and its difference to LR-L2 on the same pairs}\n\\label{tab:app_ktsp_vote_fewshot}\n"
            "\\setlength{\\tabcolsep}{2.5pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{llcccccc}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{\\textbf{Target-only (E1b)}} & \\multicolumn{3}{c}{\\textbf{Select-on-source (E7)}} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "\\textbf{Cohort} & $\\boldsymbol{n_T}$ & $k = 2$ & $k = 4$ & $k = 10$ & $k = 2$ & $k = 4$ & $k = 10$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Mean AUROC of the voting rule over the 20 runs of a cell, with the mean difference to LR-L2 on the same pairs in parentheses (voting minus LR-L2). Target-only: pairs and directions learned on the labelled subset $L_T$. Select-on-source: pairs selected on TCGA, directions learned on $L_T$. ``--'' = level not part of the experiment (E7 uses $n_T \\in \\{5, 10, 20\\}$) or not available for the cohort.}\n"
            "\\end{table*}\n")
    write(out, "tab_app_ktsp_vote_fewshot.tex", text)
    d1 = [float(r["delta_mean"]) for r in e1.values()]
    d2 = [float(r["delta_mean"]) for r in e1b]
    d3 = [float(r["delta_mean"]) for r in e7]
    NUMBERS["sa3"] = {"e1_mean_delta": round(statistics.mean(d1), 4), "e1_min": round(min(d1), 4), "e1_max": round(max(d1), 4), "e1_cells": len(d1), "e1_cells_positive": sum(v > 0 for v in d1),
                      "e1_mean_delta_k10": round(statistics.mean(float(r["delta_mean"]) for (c, k), r in e1.items() if k == "10"), 4),
                      "e1_mean_delta_k2": round(statistics.mean(float(r["delta_mean"]) for (c, k), r in e1.items() if k == "2"), 4),
                      "e1b_mean_delta": round(statistics.mean(d2), 4), "e1b_min": round(min(d2), 4), "e1b_max": round(max(d2), 4), "e1b_cells": len(d2), "e1b_cells_positive": sum(v > 0 for v in d2),
                      "e7_mean_delta": round(statistics.mean(d3), 4), "e7_min": round(min(d3), 4), "e7_max": round(max(d3), 4), "e7_cells": len(d3), "e7_cells_positive": sum(v > 0 for v in d3),
                      "zero_shot_vote_min": round(min(float(r["auroc_vote"]) for r in zs.values()), 4), "zero_shot_vote_max": round(max(float(r["auroc_vote"]) for r in zs.values()), 4),
                      "zero_shot_lr_min": round(min(float(r["auroc_lr"]) for r in zs.values()), 4), "zero_shot_lr_max": round(max(float(r["auroc_lr"]) for r in zs.values()), 4),
                      "zero_shot_cells": len(zs), "zero_shot_cells_below_0.5": sum(r["vote_below_05"] == "True" or r["lr_below_05"] == "True" for r in zs.values()),
                      "zero_shot_k2": {CNAME[c]: [round(float(zs[(c, "2")]["auroc_vote"]), 4), round(float(zs[(c, "2")]["auroc_lr"]), 4)] for c in GEO},
                      "e1b_nt5_k10": {CNAME[c]: [round(float(fs[(c, "5", "10")]["auroc_vote"]), 4), round(float(fs[(c, "5", "10")]["auroc_lr"]), 4)] for c in GEO}}


# ---------------------------------------------------------------------------------------------------------------
#  9. SA4: two variants of the zero-shot strategy; article table of the two-gene rule
# ---------------------------------------------------------------------------------------------------------------
ZVAR = [("e7_phi", "reference (E7)"), ("consistent_phi", "(i) selection consistent with $\\varphi$"),
        ("target_gene_zscore", "(ii) per-gene standardisation with unlabelled target samples")]


def tables_sa4(extra, k2dir, out):
    zt = {(r["cohort"], r["variant"], r["k"]): r for r in rd(os.path.join(extra, "B3_tables", "b3_zero_shot_table.csv"))}
    gate = rd(os.path.join(extra, "B3_tables", "b3_gate_e7_source_only.csv"))
    assert all(g["pass"] == "True" for g in gate)
    ks = ["2", "3", "4", "5", "10"]
    lines = []
    for ci, c in enumerate(GEO):
        if ci:
            lines.append("\\addlinespace[2pt]")
        for i, (v, lab) in enumerate(ZVAR):
            lines.append("%s & %s & %s \\\\" % (CNAME[c] if i == 0 else "", lab, " & ".join(pm(zt[(c, v, k)]["auroc_mean"], zt[(c, v, k)]["auroc_sd"]) for k in ks)))
    text = ("% source: B3_tables/b3_zero_shot_table.csv (auroc_mean, auroc_sd by cohort, variant and k; five test sets), B3_tables/b3_gate_e7_source_only.csv (the reference rows equal the published E7 values), tag survey-v3-dmkd.2\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD208).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{SA4: AUROC of the zero-shot strategy and of its two variants}\n\\label{tab:app_zeroshot}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
            "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
            "\\begin{tabular}{llccccc}\n\\toprule\n"
            "\\textbf{Cohort} & \\textbf{Variant} & $k = 2$ & $k = 3$ & $k = 4$ & $k = 5$ & $k = 10$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Mean $\\pm$ SD of the AUROC over the five test sets of a target cohort; no target label is used. Reference: genes selected with L1-LR on the source standardised per gene, classifier fitted on the within-sample normalisation $\\varphi$ of the panel. Variant (i): the genes are selected on the source after the within-sample standardisation, so that selection and classification use the same representation. Variant (ii): each panel gene is standardised on the source for training and on all samples of the target cohort, without their labels, for scoring.}\n"
            "\\end{sidewaystable*}\n")
    write(out, "tab_app_zeroshot.tex", text)
    NUMBERS["sa4"] = {v: {CNAME[c]: {k: round(float(zt[(c, v, k)]["auroc_mean"]), 4) for k in ks} for c in GEO} for v, _ in ZVAR}
    NUMBERS["sa4_reversal_cells"] = {v: [(CNAME[c], k) for c in GEO for k in ks if float(zt[(c, v, k)]["auroc_mean"]) < 0.5] for v, _ in ZVAR}
    NUMBERS["sa4_identical_to_reference_i"] = [(CNAME[c], k) for c in GEO for k in ks
                                               if abs(float(zt[(c, "consistent_phi", k)]["auroc_mean"]) - float(zt[(c, "e7_phi", k)]["auroc_mean"])) < 1e-12]

    gd = {(r["cohort"], r["gene"]): r for r in rd(os.path.join(k2dir, "e7_k2_gene_directions.csv"))}
    lines = []
    for i, c in enumerate(COHORTS):
        phi = gd[(c, "phi_DSG3_minus_KRT5")]
        cells = ["$%+.2f$" % float(gd[(c, "DSG3")]["shift"]), "$%+.2f$" % float(gd[(c, "KRT5")]["shift"]), "$%+.2f$" % float(phi["shift"]), f3(phi["auroc_as_lusc_score"])]
        if c in GEO:
            cells += [f3(zt[(c, v, "2")]["auroc_mean"]) for v, _ in ZVAR]
        else:
            cells += ["--", "--", "--"]
        lines.append("%s & %s \\\\" % (CNAME[c] + (" (source)" if i == 0 else ""), " & ".join(cells)))
        if i == 0:
            lines.append("\\addlinespace[2pt]")
    text = ("% source: input/results/_received_2026-11-09/E7_k2_evidence/e7_k2_gene_directions.csv (shift = mean LUSC - mean LUAD of DSG3, KRT5 and of phi = DSG3 - KRT5; auroc_as_lusc_score of phi); B3_tables/b3_zero_shot_table.csv (auroc_mean at k = 2 of e7_phi, consistent_phi, target_gene_zscore), tag survey-v3-dmkd.2\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD174, QD198, QD211).  Table of Online Resource 1 since 21/11/2026 (QD240; until then tab_zeroshot_k2.tex, a table of the article).\n"
            "\\begin{table*}[tp]\n\\centering\n\\scriptsize\n\\caption{The two-gene zero-shot rule: class shifts of DSG3 and KRT5 and AUROC at $k = 2$}\n\\label{tab:app_zeroshot_k2}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{lccccccc}\n\\toprule\n"
            " & \\multicolumn{3}{c}{\\textbf{Shift, LUSC minus LUAD}} & \\textbf{AUROC of} $\\boldsymbol{\\varphi}$ & \\multicolumn{3}{c}{\\textbf{Zero-shot AUROC at} $\\boldsymbol{k = 2}$} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}\n"
            "\\textbf{Cohort} & DSG3 & KRT5 & $\\varphi$ & \\textbf{as a LUSC score} & reference & variant (i) & variant (ii) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Shift = mean log-expression in LUSC minus mean in LUAD over all samples of the cohort; $\\varphi = x_{\\mathrm{DSG3}} - x_{\\mathrm{KRT5}}$. The fourth column is the AUROC obtained when $\\varphi$ itself is read as a score for LUSC, so a value below 0.5 means that $\\varphi$ is lower in LUSC. The source classifier gives $\\varphi$ a negative weight. Reference = zero-shot strategy of E7. Variants (i) and (ii) are those of SA4 (Section~\\ref{sup:design_sensitivity}); variant (i) selects its own two genes on the source, so its rule is not $\\varphi$ of the first columns.}\n"
            "\\end{table*}\n")
    write(out, "tab_app_zeroshot_k2.tex", text)
    NUMBERS["k2_rule"] = {CNAME[c]: {"shift_DSG3": round(float(gd[(c, "DSG3")]["shift"]), 3), "shift_KRT5": round(float(gd[(c, "KRT5")]["shift"]), 3),
                                     "shift_phi": round(float(gd[(c, "phi_DSG3_minus_KRT5")]["shift"]), 3),
                                     "auroc_phi_as_lusc": round(float(gd[(c, "phi_DSG3_minus_KRT5")]["auroc_as_lusc_score"]), 4)} for c in COHORTS}


# ---------------------------------------------------------------------------------------------------------------
#  10. E8: second task, EGFR mutation in lung adenocarcinoma
# ---------------------------------------------------------------------------------------------------------------
E8C = ["tcga_luad_egfr", "gse31210_egfr"]


def tables_e8(extra, official, out):
    summ = {(r["cohort"], r["method"], r["k"]): r for r in rd(os.path.join(extra, "B7_tables", "b7_e1_summary.csv"))}
    tests = {(r["cohort"], r["method"], r["k"]): r for r in rd(os.path.join(extra, "B7_tables", "b7_e1_tests_vs_random_b0.csv"))}
    kst = {(r["cohort"], r["method"]): r for r in rd(os.path.join(extra, "B7_tables", "b7_kstar.csv"))}
    e4 = {(r["fraction"], r["method"], r["k"]): r for r in rd(os.path.join(extra, "B7_tables", "b7_e4_stability.csv"))}
    gate = {r["cohort"]: r for r in rd(os.path.join(extra, "B7_gate", "b7_gate_table.csv"))}
    r1 = {r["cohort"]: r for r in rd(os.path.join(extra, "B7_tables", "b7_R1.csv"))}
    r3 = {r["cohort"]: r for r in rd(os.path.join(extra, "B7_tables", "b7_R3_kendall_auroc.csv"))}
    r4 = rd(os.path.join(extra, "B7_tables", "b7_R4_kendall_stability.csv"))[0]
    b0k = {c: [k for (cc, m, k) in summ if cc == c and m == "b0"][0] for c in E8C}
    sel = NONWRAPPER

    def marks(c, m, k):
        t = tests.get((c, m, k))
        if t is None:
            return ""
        s = ("\\dagger" if t["beats_random_holm"] == "True" else "") + ("\\ddagger" if t["differs_from_b0_holm"] == "True" else "")
        return "$^{%s}$" % s if s else ""

    def kstar(c, m):
        v = kst[(c, m)]["k_star_delta_0.02"]
        return str(int(float(v))) if v else "--"

    # ---- article table at k = 10
    def row10(m):
        cells = []
        for c in E8C:
            t = tests[(c, m, "10")]
            cells += [f3(summ[(c, m, "10")]["auroc"]) + marks(c, m, "10"), f3(summ[(c, m, "10")]["auprc"]),
                      "$%+.3f$ [$%+.3f$; $%+.3f$]" % (float(t["diff_vs_b0"]), float(t["diff_b0_ci_low"]), float(t["diff_b0_ci_high"])), kstar(c, m)]
            if c == "tcga_luad_egfr":
                cells.append(f3(e4[("0.7", m, "10")]["nogueira"]))
        return cells

    lines = family_rows(sel, row10)
    lines.append("\\midrule")
    lines.append("Reference & random-$k$ ($k = 10$) & %s & -- & -- & -- & %s & %s & -- & -- & -- \\\\" % (
        f3(summ[("tcga_luad_egfr", "random_k", "10")]["auroc"]), f3(e4[("0.7", "random_k", "10")]["nogueira"]), f3(summ[("gse31210_egfr", "random_k", "10")]["auroc"])))
    lines.append(" & B0 (all genes) & %s & %s & -- & -- & -- & %s & %s & -- & -- \\\\" % (
        f3(summ[("tcga_luad_egfr", "b0", b0k["tcga_luad_egfr"])]["auroc"]), f3(summ[("tcga_luad_egfr", "b0", b0k["tcga_luad_egfr"])]["auprc"]),
        f3(summ[("gse31210_egfr", "b0", b0k["gse31210_egfr"])]["auroc"]), f3(summ[("gse31210_egfr", "b0", b0k["gse31210_egfr"])]["auprc"])))
    lines.append("\\midrule")
    lines.append("\\multicolumn{2}{l}{Selectors above random-$k$ at $k = 10$ (Holm)} & %s/%s & & & & & %s/%s & & & \\\\" % (
        r1["tcga_luad_egfr"]["n_beat_random_k10_holm"], r1["tcga_luad_egfr"]["n_selectors"], r1["gse31210_egfr"]["n_beat_random_k10_holm"], r1["gse31210_egfr"]["n_selectors"]))
    text = ("%% source: B7_tables/b7_e1_summary.csv (auroc, auprc at k = 10; b0), B7_tables/b7_e1_tests_vs_random_b0.csv (diff_vs_b0, diff_b0_ci_low, diff_b0_ci_high, beats_random_holm, differs_from_b0_holm), B7_tables/b7_kstar.csv (k_star_delta_0.02), B7_tables/b7_e4_stability.csv (nogueira at f = 0.7, k = 10), B7_tables/b7_R1.csv, tag survey-v3-dmkd.2\n"
            "%% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD201, QD208).\n"
            "\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{E8: EGFR mutation in lung adenocarcinoma at $k = 10$}\n\\label{tab:e8}\n"
            "\\setlength{\\tabcolsep}{3.5pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
            "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
            "\\begin{tabular}{llccccccccc}\n\\toprule\n"
            " & & \\multicolumn{5}{c}{\\textbf{TCGA-LUAD} ($n = %s$, %s mutant)} & \\multicolumn{4}{c}{\\textbf{GSE31210} ($n = %s$, %s mutant)} \\\\\n"
            "\\cmidrule(lr){3-7}\\cmidrule(lr){8-11}\n"
            "\\textbf{Family} & \\textbf{Method} & AUROC & AUPRC & AUROC minus B0 [95\\%% CI] & $k^{*}$ & $\\hat{\\Phi}$ & AUROC & AUPRC & AUROC minus B0 [95\\%% CI] & $k^{*}$ \\\\\n\\midrule\n"
            % (gate["tcga_luad_egfr"]["n"], "%d" % round(float(gate["tcga_luad_egfr"]["positive_rate"]) * float(gate["tcga_luad_egfr"]["n"])),
               gate["gse31210_egfr"]["n"], "%d" % round(float(gate["gse31210_egfr"]["positive_rate"]) * float(gate["gse31210_egfr"]["n"]))))
    text += "\n".join(lines) + ("\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
                                "\\tablenote{Mean over the 25 patient-level folds, LR-L2. AUPRC = area under the precision--recall curve (its chance level is the proportion of mutant tumours, %.2f and %.2f). $\\dagger$: above random-$k$ after Holm correction; $\\ddagger$: differs from B0 after Holm correction (family = method $\\times$ its budgets). The interval is the two-sided 95\\%% Nadeau--Bengio interval of the difference to B0 at $k = 10$, without correction. $k^{*}$ = minimum sufficient budget of the cohort at the margin 0.02 (``--'' = no tested budget satisfies the bound). $\\hat{\\Phi}$ = Nogueira index over 30 independent subsamples at $f = 0.7$ (measured on TCGA-LUAD only).}\n\\end{sidewaystable*}\n"
                                % (float(gate["tcga_luad_egfr"]["positive_rate"]), float(gate["gse31210_egfr"]["positive_rate"])))
    write(out, "tab_e8.tex", text)

    # ---- supplement: AUROC and AUPRC by budget, one table per cohort
    for c, short in (("tcga_luad_egfr", "tcga"), ("gse31210_egfr", "gse31210")):
        lines = ["\\multicolumn{10}{l}{\\textit{(a) AUROC}} \\\\"]
        lines += family_rows(sel, lambda m: [f3(summ[(c, m, k)]["auroc"]) + marks(c, m, k) if (c, m, k) in summ else "--" for k in BUDGETS])
        lines.append("\\addlinespace[2pt]")
        lines.append("Reference & random-$k$ & %s \\\\" % " & ".join(f3(summ[(c, "random_k", k)]["auroc"]) for k in BUDGETS))
        lines.append(" & B0 (all genes) & \\multicolumn{8}{c}{%s} \\\\" % f3(summ[(c, "b0", b0k[c])]["auroc"]))
        lines.append("\\midrule")
        lines.append("\\multicolumn{10}{l}{\\textit{(b) AUPRC}} \\\\")
        lines += family_rows(sel, lambda m: [f3(summ[(c, m, k)]["auprc"]) if (c, m, k) in summ else "--" for k in BUDGETS])
        lines.append("\\addlinespace[2pt]")
        lines.append("Reference & B0 (all genes) & \\multicolumn{8}{c}{%s} \\\\" % f3(summ[(c, "b0", b0k[c])]["auprc"]))
        counts = " ".join("%s/%s at $k = %s$%s" % (sum(tests[(c, m, k)]["beats_random_holm"] == "True" for m in sel if (c, m, k) in tests),
                                                   sum(1 for m in sel if (c, m, k) in tests), k, "," if k != "50" else "") for k in BUDGETS)
        text = ("%% source: B7_tables/b7_e1_summary.csv (auroc, auprc; cohort = %s), B7_tables/b7_e1_tests_vs_random_b0.csv (beats_random_holm, differs_from_b0_holm), tag survey-v3-dmkd.2\n"
                "%% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD208).  Table of Online Resource 1 (ESM_1.tex).\n"
                "\\begin{table*}[p]\n\\centering\n\\scriptsize\n\\caption{E8 on %s: AUROC and AUPRC by method and budget $k$}\n\\label{tab:app_e8_%s}\n"
                "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.08}\n"
                "\\begin{adjustbox}{max width=\\textwidth,max totalheight=0.9\\textheight}\n"
                "\\begin{tabular}{llcccccccc}\n\\toprule\n"
                "\\textbf{Family} & \\textbf{Method} & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=5$ & $k=10$ & $k=25$ & $k=50$ \\\\\n\\midrule\n"
                % (c, CNAME[c], short))
        text += "\n".join(lines) + ("\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
                                    "\\tablenote{Mean over the 25 patient-level folds, LR-L2; $k$-TSP exists for even $k$ only. $\\dagger$: above random-$k$ after Holm correction (one-sided Nadeau--Bengio test, family = method $\\times$ its budgets); $\\ddagger$: differs from B0 after Holm correction (two-sided); every $\\ddagger$ lies below B0. Selectors above random-$k$: %s. The AUPRC of the random panels was not stored.}\n\\end{table*}\n" % counts)
        write(out, "tab_app_e8_%s.tex" % short, text)
        assert all(float(t["diff_vs_b0"]) < 0 for (cc, m, k), t in tests.items() if cc == c and t["differs_from_b0_holm"] == "True")

    # ---- supplement: rankings on the two tasks
    e1 = {(r["cohort"], r["method"], r["k"]): r for r in rd(os.path.join(official, "e1_summary.csv")) if r["classifier"] == "lr_l2"}
    e4o = {(r["cohort"], r["fraction"], r["method"], r["k"]): r for r in rd(os.path.join(official, "e4_stability.csv"))}
    T = "tcga_luad_lusc_primary_tumor"
    a_h = [float(e1[(T, m, "10")]["auroc_mean"]) for m in sel]
    a_e = [float(summ[("tcga_luad_egfr", m, "10")]["auroc"]) for m in sel]
    a_g = [float(summ[("gse31210_egfr", m, "10")]["auroc"]) for m in sel]
    s_h = [float(e4o[(T, "0.7", m, "10")]["nogueira"]) for m in sel]
    s_e = [float(e4[("0.7", m, "10")]["nogueira"]) for m in sel]
    tau = {"auroc_tcga": stats.kendalltau(a_h, a_e), "auroc_gse": stats.kendalltau(a_h, a_g), "stab": stats.kendalltau(s_h, s_e)}
    # the stored rank correlations are reproduced from the columns of the table
    assert abs(tau["auroc_tcga"][0] - float(r3["tcga_luad_egfr"]["kendall_tau"])) < 1e-9 and abs(tau["auroc_tcga"][1] - float(r3["tcga_luad_egfr"]["p_value"])) < 1e-9
    assert abs(tau["auroc_gse"][0] - float(r3["gse31210_egfr"]["kendall_tau"])) < 1e-9
    assert abs(tau["stab"][0] - float(r4["kendall_tau"])) < 1e-9 and abs(tau["stab"][1] - float(r4["p_value"])) < 1e-9
    lines = family_rows(sel, lambda m: [f3(e1[(T, m, "10")]["auroc_mean"]), f3(summ[("tcga_luad_egfr", m, "10")]["auroc"]), f3(summ[("gse31210_egfr", m, "10")]["auroc"]),
                                        f3(e4o[(T, "0.7", m, "10")]["nogueira"]),
                                        "%s [%s; %s]" % (f3(e4[("0.7", m, "10")]["nogueira"]), f3(e4[("0.7", m, "10")]["nogueira_ci_low"]), f3(e4[("0.7", m, "10")]["nogueira_ci_high"])),
                                        f3(e4[("0.5", m, "10")]["nogueira"]), f3(e4[("0.9", m, "10")]["nogueira"])])
    lines.append("\\midrule")
    lines.append("\\multicolumn{2}{l}{Kendall $\\tau$ with the histology task ($p$)} & -- & %.2f (%.3f) & %.2f (%.3f) & -- & %.2f (%.3f) & -- & -- \\\\" % (
        tau["auroc_tcga"][0], tau["auroc_tcga"][1], tau["auroc_gse"][0], tau["auroc_gse"][1], tau["stab"][0], tau["stab"][1]))
    text = ("% source: e1_summary.csv (auroc_mean, lr_l2, TCGA, k = 10), e4_stability.csv (nogueira, TCGA, f = 0.7, k = 10) [v03-11]; B7_tables/b7_e1_summary.csv (auroc at k = 10), B7_tables/b7_e4_stability.csv (nogueira and interval at k = 10), B7_tables/b7_R3_kendall_auroc.csv, B7_tables/b7_R4_kendall_stability.csv, tag survey-v3-dmkd.2\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD199, QD208).  The rank correlations recomputed from the columns equal the stored values.  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{E8: AUROC and stability of the eleven non-wrapper selectors at $k = 10$ on the histology task and on the EGFR task}\n\\label{tab:app_e8_rank}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
            "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
            "\\begin{tabular}{llccccccc}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{\\textbf{AUROC at} $\\boldsymbol{k = 10}$} & \\multicolumn{4}{c}{\\textbf{Nogueira index at} $\\boldsymbol{k = 10}$} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\n"
            " & & LUAD/LUSC & EGFR & EGFR & LUAD/LUSC & EGFR, TCGA-LUAD & EGFR, TCGA-LUAD & EGFR, TCGA-LUAD \\\\\n"
            "\\textbf{Family} & \\textbf{Method} & TCGA & TCGA-LUAD & GSE31210 & TCGA, $f = 0.7$ & $f = 0.7$ [95\\% CI] & $f = 0.5$ & $f = 0.9$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{LUAD/LUSC columns: E1 and E4 of the article on TCGA (1,016 patients). EGFR columns: E8. Nogueira index over 30 independent subsamples at fraction $f$. The last row gives the Kendall rank correlation between a column and the LUAD/LUSC column of the same quantity, over the eleven selectors. The correlation of the GSE31210 column compares two cohorts, two platforms and two definitions of the label at once, so it is given for completeness and is not interpreted in the article.}\n"
            "\\end{sidewaystable*}\n")
    write(out, "tab_app_e8_rank.tex", text)

    # ---- supplement: secondary metrics at k = 10
    def sec(c, m, k):
        r = summ[(c, m, k)]
        return [f3(r["bacc"]), f3(r["mcc"]), f3(r["logloss"]), f3(r["brier"])]

    lines = family_rows(sel, lambda m: sec("tcga_luad_egfr", m, "10") + sec("gse31210_egfr", m, "10"))
    lines.append("\\midrule")
    lines.append("Reference & random-$k$ ($k = 10$) & %s \\\\" % " & ".join(sec("tcga_luad_egfr", "random_k", "10") + sec("gse31210_egfr", "random_k", "10")))
    lines.append(" & B0 (all genes) & %s \\\\" % " & ".join(sec("tcga_luad_egfr", "b0", b0k["tcga_luad_egfr"]) + sec("gse31210_egfr", "b0", b0k["gse31210_egfr"])))
    text = ("% source: B7_tables/b7_e1_summary.csv (bacc, mcc, logloss, brier at k = 10; b0), tag survey-v3-dmkd.2\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  18/11/2026 (QD198, QD208).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{table*}[tp]\n\\centering\n\\scriptsize\n\\caption{E8: balanced accuracy, MCC, log-loss and Brier score at $k = 10$}\n\\label{tab:app_e8_secondary}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{llcccccccc}\n\\toprule\n"
            " & & \\multicolumn{4}{c}{\\textbf{TCGA-LUAD}} & \\multicolumn{4}{c}{\\textbf{GSE31210}} \\\\\n"
            "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
            "\\textbf{Family} & \\textbf{Method} & BAcc & MCC & Log-loss & Brier & BAcc & MCC & Log-loss & Brier \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Mean over the 25 patient-level folds, LR-L2. BAcc = balanced accuracy, MCC = Matthews correlation coefficient. Higher is better for BAcc and MCC, lower for the log-loss and the Brier score.}\n"
            "\\end{table*}\n")
    write(out, "tab_app_e8_secondary.tex", text)

    # ---- numbers for the running text
    for c in E8C:
        au = {m: float(summ[(c, m, "10")]["auroc"]) for m in sel}
        ap = {m: float(summ[(c, m, "10")]["auprc"]) for m in sel}
        NUMBERS["e8_%s" % c] = {
            "n": int(gate[c]["n"]), "positive_rate": round(float(gate[c]["positive_rate"]), 4), "p_genes": int(gate[c]["p"]),
            "e0_auroc": round(float(gate[c]["e0_auroc_real"]), 4), "e0_null_mean": round(float(gate[c]["e0_null_mean"]), 4), "p_perm": round(float(gate[c]["p_perm"]), 5),
            "b0_auroc": round(float(gate[c]["b0_auroc_mean"]), 4), "b0_sd": round(float(gate[c]["b0_auroc_sd"]), 4), "random_k10": round(float(gate[c]["random_k_auroc_mean"]), 4),
            "b0_minus_random": round(float(gate[c]["b0_minus_random_k"]), 4), "b0_auprc": round(float(gate[c]["b0_auprc_mean"]), 4),
            "auroc_k10_min": (min(au, key=au.get), round(min(au.values()), 4)), "auroc_k10_max": (max(au, key=au.get), round(max(au.values()), 4)),
            "auroc_k10_range": round(max(au.values()) - min(au.values()), 4), "auroc_k10": {m: round(v, 4) for m, v in au.items()},
            "auprc_k10_min": round(min(ap.values()), 4), "auprc_k10_max": round(max(ap.values()), 4),
            "gain_vs_random_k10_min": round(min(float(tests[(c, m, "10")]["gain_vs_random"]) for m in sel), 4),
            "gain_vs_random_k10_max": round(max(float(tests[(c, m, "10")]["gain_vs_random"]) for m in sel), 4),
            "beats_random_by_k": {k: "%d/%d" % (sum(tests[(c, m, k)]["beats_random_holm"] == "True" for m in sel if (c, m, k) in tests), sum(1 for m in sel if (c, m, k) in tests)) for k in BUDGETS},
            "differs_b0_by_k": {k: [m for m in sel if (c, m, k) in tests and tests[(c, m, k)]["differs_from_b0_holm"] == "True"] for k in BUDGETS},
            "diff_vs_b0_k10": {m: [round(float(tests[(c, m, "10")][x]), 4) for x in ("diff_vs_b0", "diff_b0_ci_low", "diff_b0_ci_high")] for m in sel},
            "kstar_0.02": {m: kstar(c, m) for m in sel}, "kstar_0.01": {m: (kst[(c, m)]["k_star_delta_0.01"] or "--") for m in sel},
            "logloss_b0": round(float(summ[(c, "b0", b0k[c])]["logloss"]), 4),
            "logloss_k10_min": round(min(float(summ[(c, m, "10")]["logloss"]) for m in sel), 4), "logloss_k10_max": round(max(float(summ[(c, m, "10")]["logloss"]) for m in sel), 4)}
    NUMBERS["e8_stability_k10"] = {f: {m: round(float(e4[(f, m, "10")]["nogueira"]), 3) for m in sel} for f in ("0.5", "0.7", "0.9")}
    NUMBERS["e8_kendall"] = {"auroc_tcga_luad": [round(tau["auroc_tcga"][0], 4), round(tau["auroc_tcga"][1], 4)], "auroc_gse31210": [round(tau["auroc_gse"][0], 4), round(tau["auroc_gse"][1], 4)],
                             "stability": [round(tau["stab"][0], 4), round(tau["stab"][1], 4)]}
    NUMBERS["e1_tcga_k10_range_11_selectors"] = round(max(a_h) - min(a_h), 4)
    NUMBERS["e1_k10_range_11_selectors"] = {CNAME[c]: round(max(float(e1[(c, m, "10")]["auroc_mean"]) for m in sel) - min(float(e1[(c, m, "10")]["auroc_mean"]) for m in sel), 4) for c in COHORTS}
    NUMBERS["e4_tcga_f0.7_k10"] = {m: round(v, 3) for m, v in zip(sel, s_h)}


# ---------------------------------------------------------------------------------------------------------------
#  11. E7 over 20 splits (repository folder B1), model-based transfer (B6) and n_T* over 20 splits
#      21/11/2026 (QD226, QD227, QD236-QD238)
# ---------------------------------------------------------------------------------------------------------------
STRATEGIES = [("pooled", "pooled"), ("select_on_source", "select-on-source"), ("source_only", "source-only (zero-shot)")]
MODELS = [("ptlasso", "ptLasso"), ("transglm", "Trans-GLM")]
LABELS4 = ["positive transfer", "negative transfer", "practically equivalent", "inconclusive"]
MARK = {"positive transfer": "+", "negative transfer": "-", "practically equivalent": "=", "inconclusive": ""}
KS = ["2", "3", "4", "5", "10"]
NTS = ["5", "10", "20"]
TOP_LEVEL = {"gse10245": "10", "gse30219": "40", "gse37745": "40", "gse42127": "20"}
MARK_NOTE = ("Superscripts give the label of a cell over the 20 splits: $+$ positive transfer and $-$ negative transfer "
             "(two-sided Wilcoxon signed-rank test over the splits, Holm correction over the 55 cells of the strategy, corrected $p < 0.05$, "
             "and a mean difference beyond $\\pm 0.01$), $=$ practically equivalent (95\\% bootstrap interval of the mean difference within $\\pm 0.01$); "
             "a cell without superscript is inconclusive.")


def dcell(v, label=None):
    """Signed difference in math mode, with the mark of its label as a superscript."""
    s = "%+.3f" % float(v)
    m = MARK[label] if label else ""
    return "$%s^{%s}$" % (s, m) if m else "$%s$" % s


def counts4(rows):
    return [sum(r["label"] == lab for r in rows) for lab in LABELS4]


def tables_transfer20(b1dir, official, out, derived=None):
    lab = rd(os.path.join(b1dir, "B1_tables", "b1_e7_split_labels.csv"))
    b6 = rd(os.path.join(b1dir, "B6_tables", "b6_split_labels_vs_target_only.csv"))
    naive = rd(os.path.join(b1dir, "B6_tables", "b6_delta_vs_naive.csv"))
    degen = rd(os.path.join(b1dir, "B6_tables", "b6_degeneracy.csv"))
    gates = rd(os.path.join(b1dir, "B1_tables", "b1_gates.csv"))
    e7 = rd(os.path.join(official, "e7_transfer_table.csv"))
    assert all(g["pass"] == "True" and float(g["max_abs_diff_auroc"]) == 0.0 for g in gates) and len(gates) == 8
    assert len(lab) == 165 and len(b6) == 110 and len(naive) == 220 and len(degen) == 110 and len(e7) == 165
    assert all(r["n_splits"] == "20" and r["reference"] == "target_only" for r in lab + b6)
    L = {(r["strategy"], r["cohort"], r["n_t"], r["k"]): r for r in lab + b6}
    N = {(r["strategy"], r["reference"], r["cohort"], r["n_t"], r["k"]): r for r in naive}
    E5 = {(r["strategy"], r["cohort"], r["n_t"], r["k"]): r for r in e7}
    cells = [(c, n, k) for c in GEO for n in NTS for k in KS if not (c == "gse10245" and n == "20")]
    assert len(cells) == 55 and all((s, c, n, k) in L for s, _ in STRATEGIES + MODELS for c, n, k in cells)

    # the label of every cell follows from the stored test and interval by the rule fixed on 12/11/2026
    for r in lab + b6:
        p, d, lo, hi = float(r["p_holm"]), float(r["mean_delta"]), float(r["ci_low"]), float(r["ci_high"])
        rule = ("positive transfer" if p < 0.05 and d > 0.01 else "negative transfer" if p < 0.05 and d < -0.01
                else "practically equivalent" if lo > -0.01 and hi < 0.01 else "inconclusive")
        assert rule == r["label"], (r["strategy"], r["cohort"], r["n_t"], r["k"], rule, r["label"])

    # target-only AUROC over 20 splits: mean of the four nested subsets within a split, then mean over the splits
    target = {}
    for c in GEO:
        acc = {}
        for r in rd(os.path.join(b1dir, "B1_E7_20", c, "e7_per_run.csv")):
            if r["strategy"] == "target_only":
                acc.setdefault((r["n_t"], r["k"], r["split"]), []).append(float(r["auroc"]))
        for n in NTS:
            for k in KS:
                per_split = [statistics.mean(v) for (nn, kk, s), v in acc.items() if nn == n and kk == k]
                if per_split:
                    assert len(per_split) == 20 and all(len(v) == 4 for (nn, kk, s), v in acc.items() if nn == n and kk == k)
                    target[(c, n, k)] = statistics.mean(per_split)
    # absolute AUROC of the zero-shot strategy over 20 splits and over the five splits of the few-shot design
    zero = {}
    for c in GEO:
        acc = {}
        for r in rd(os.path.join(b1dir, "B1_E7_20", c, "e7_per_run.csv")):
            if r["strategy"] == "source_only":
                acc.setdefault((r["k"], int(r["split"])), []).append(float(r["auroc"]))
        for k in KS:
            ps = {s: statistics.mean(v) for (kk, s), v in acc.items() if kk == k}
            assert len(ps) == 20
            zero[(c, k)] = {"mean20": statistics.mean(ps.values()), "mean5": statistics.mean(ps[s] for s in range(5)),
                            "min20": min(ps.values()), "below05": sum(v < 0.5 for v in ps.values())}

    # ---- article table: k = 10, five strategies, label counts over all budgets ----
    def block(key, name):
        rows = []
        for i, c in enumerate(GEO):
            cs = []
            for n in NTS:
                r = L.get((key, c, n, "10"))
                cs.append(dcell(r["mean_delta"], r["label"]) if r else "--")
            cnt = counts4([L[(key, cc, n, k)] for cc, n, k in cells if cc == c])
            rows.append("%s & %s & %s & %s \\\\" % (name if i == 0 else "", CNAME[c], " & ".join(cs), " / ".join(map(str, cnt))))
        cnt = counts4([L[(key, c, n, k)] for c, n, k in cells])
        rows.append(" & \\textit{all cohorts} & & & & \\textit{%s} \\\\" % " / ".join(map(str, cnt)))
        return rows

    lines = ["\\multicolumn{6}{l}{\\textit{AUROC of target-only selection (L1-LR on $L_T$)}} \\\\"]
    for i, c in enumerate(GEO):
        lines.append("%s & %s & %s & \\\\" % ("target-only" if i == 0 else "", CNAME[c],
                                             " & ".join(f3(target[(c, n, "10")]) if (c, n, "10") in target else "--" for n in NTS)))
    lines.append("\\midrule")
    lines.append("\\multicolumn{6}{l}{\\textit{Difference to target-only selection, naive strategies}} \\\\")
    for j, (key, name) in enumerate(STRATEGIES):
        if j:
            lines.append("\\addlinespace[2pt]")
        lines += block(key, name)
    lines.append("\\midrule")
    lines.append("\\multicolumn{6}{l}{\\textit{Difference to target-only selection, transfer methods with a model of the source}} \\\\")
    for j, (key, name) in enumerate(MODELS):
        if j:
            lines.append("\\addlinespace[2pt]")
        lines += block(key, name)
    text = ("% source: B1_tables/b1_e7_split_labels.csv and B6_tables/b6_split_labels_vs_target_only.csv (mean_delta, label; 20 splits, Holm over the 55 cells of a strategy); B1_E7_20/<cohort>/e7_per_run.csv (target_only, k = 10: mean of the four subsets of a split, then mean over the 20 splits); branch dmkd-revision, commit 61f6252 (tables written at 3fe6d99)\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  21/11/2026 (QD226, QD236): replaces the table of the five-split flags.\n"
            "\\begin{table*}[tp]\n\\centering\n\\scriptsize\n\\caption{E7: transfer from the TCGA source at $k = 10$ and labels of all cells over 20 splits}\n\\label{tab:e7}\n"
            # arraystretch 1.16: the table is a page of its own.  With 1.1 its height (0.885 of the text height) falls between the
            # limit of a top float (0.87) and that of a float page (0.887) of sn-jnl, and LaTeX defers it to the end of the article.
            "\\setlength{\\tabcolsep}{6pt}\n\\renewcommand{\\arraystretch}{1.16}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{llcccc}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{\\textbf{Budget} $\\boldsymbol{k = 10}$} & \\textbf{Labels over all} $\\boldsymbol{k}$ \\\\\n"
            "\\cmidrule(lr){3-5}\n"
            "\\textbf{Strategy} & \\textbf{Cohort} & $n_T = 5$ & $n_T = 10$ & $n_T = 20$ & $+$ / $-$ / $=$ / none \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Mean over 20 test/pool splits, each split being the mean of its four nested subsets. The first block gives the AUROC of target-only selection, and the other blocks the difference AUROC(strategy) $-$ AUROC(target-only) on identical subsets. " + MARK_NOTE +
            " The last column counts the cells of a cohort over the five budgets and the available target sizes (10 cells for GSE10245, 15 otherwise, 55 in all) by label, in the order positive transfer, negative transfer, practically equivalent, inconclusive. GSE10245 has no $n_T = 20$ level. The differences of ptLasso and Trans-GLM to the naive strategies are given in Online Resource~1 (Table~\\suppnum{tab:app_b6_naive}).}\n"
            "\\end{table*}\n")
    write(out, "tab_e7_20splits.tex", text)

    # ---- supplementary table: every cell over 20 splits, with the five-split difference ----
    lines = []
    for j, (key, name) in enumerate(STRATEGIES):
        if j:
            lines.append("\\midrule")
        first = True
        for c in GEO:
            for i, n in enumerate(NTS):
                if (key, c, n, "2") not in L:
                    continue
                cs = ["%s (%s)" % (dcell(L[(key, c, n, k)]["mean_delta"], L[(key, c, n, k)]["label"]), dcell(E5[(key, c, n, k)]["diff_mean"])) for k in KS]
                lines.append("%s & %s & %s & %s \\\\" % (name if first else "", CNAME[c] if i == 0 else "", n, " & ".join(cs)))
                first = False
    text = ("% source: B1_tables/b1_e7_split_labels.csv (mean_delta, label; 20 splits), commit 61f6252; e7_transfer_table.csv (diff_mean; five splits, label [v03-11]) for the value in parentheses\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  21/11/2026 (QD226, QD236; item 6 of QD173).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{E7: difference to target-only selection in every cell over 20 splits, with the difference over the five splits of the few-shot design}\n\\label{tab:app_e7}\n"
            "\\setlength{\\tabcolsep}{3.5pt}\n\\renewcommand{\\arraystretch}{0.98}\n"
            "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
            "\\begin{tabular}{llrccccc}\n\\toprule\n"
            "\\textbf{Strategy} & \\textbf{Cohort} & $\\boldsymbol{n_T}$ & $k = 2$ & $k = 3$ & $k = 4$ & $k = 5$ & $k = 10$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Cells give AUROC(strategy) $-$ AUROC(target-only) as the mean over 20 splits (seeds 42--61), each split being the mean of its four nested subsets. " + MARK_NOTE +
            " The value in parentheses is the mean difference over the 20 runs of the first five splits, on which E1b and the sensitivity analyses SA3 and SA4 were run. It is given for comparison and carries no label.}\n"
            "\\end{sidewaystable*}\n")
    write(out, "tab_app_e7_20splits.tex", text)

    # ---- supplementary tables of the two model-based methods ----
    lines = []
    for j, (key, name) in enumerate(MODELS):
        if j:
            lines.append("\\midrule")
        first = True
        for c in GEO:
            for i, n in enumerate(NTS):
                if (key, c, n, "2") not in L:
                    continue
                cs = [dcell(L[(key, c, n, k)]["mean_delta"], L[(key, c, n, k)]["label"]) for k in KS]
                lines.append("%s & %s & %s & %s \\\\" % (name if first else "", CNAME[c] if i == 0 else "", n, " & ".join(cs)))
                first = False
    text = ("% source: B6_tables/b6_split_labels_vs_target_only.csv (mean_delta, label; 20 splits, Holm over the 55 cells of a method), commit 61f6252\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  21/11/2026 (QD226, QD236).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{table*}[tp]\n\\centering\n\\scriptsize\n\\caption{E7: ptLasso and Trans-GLM against target-only selection in every cell over 20 splits}\n\\label{tab:app_b6}\n"
            "\\setlength{\\tabcolsep}{5pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{llrccccc}\n\\toprule\n"
            "\\textbf{Method} & \\textbf{Cohort} & $\\boldsymbol{n_T}$ & $k = 2$ & $k = 3$ & $k = 4$ & $k = 5$ & $k = 10$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Cells give AUROC(method) $-$ AUROC(target-only) as the mean over 20 splits, on the subsets of E7. The panel of a method consists of the $k$ genes with the largest absolute coefficient and is classified with LR-L2 fitted on $L_T$, like the panels of the naive strategies. " + MARK_NOTE + "}\n"
            "\\end{table*}\n")
    write(out, "tab_app_b6.tex", text)

    def ncell(key, ref, c, n, k):
        r = N[(key, ref, c, n, k)]
        d, lo, hi = float(r["mean_delta"]), float(r["ci_low"]), float(r["ci_high"])
        m = "\\dagger" if hi < 0 else "\\ddagger" if lo > 0 else ""
        return "$%+.3f^{%s}$" % (d, m) if m else "$%+.3f$" % d

    lines = []
    for j, (key, name) in enumerate(MODELS):
        if j:
            lines.append("\\midrule")
        first = True
        for c in GEO:
            for i, n in enumerate(NTS):
                if (key, "select_on_source", c, n, "2") not in N:
                    continue
                cs = ["%s / %s" % (ncell(key, "select_on_source", c, n, k), ncell(key, "pooled", c, n, k)) for k in KS]
                lines.append("%s & %s & %s & %s \\\\" % (name if first else "", CNAME[c] if i == 0 else "", n, " & ".join(cs)))
                first = False
    text = ("% source: B6_tables/b6_delta_vs_naive.csv (mean_delta, ci_low, ci_high for reference = select_on_source and pooled; 20 splits; no test, no multiplicity correction), commit 61f6252\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  21/11/2026 (QD226, QD236).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{sidewaystable*}\n\\centering\n\\scriptsize\n\\caption{E7: ptLasso and Trans-GLM against the two naive strategies that use target labels (descriptive)}\n\\label{tab:app_b6_naive}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
            "\\begin{adjustbox}{max width=\\linewidth,max totalheight=\\rotheight}\n"
            "\\begin{tabular}{llrccccc}\n\\toprule\n"
            "\\textbf{Method} & \\textbf{Cohort} & $\\boldsymbol{n_T}$ & $k = 2$ & $k = 3$ & $k = 4$ & $k = 5$ & $k = 10$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Each cell gives AUROC(method) $-$ AUROC(select-on-source) and, after the slash, AUROC(method) $-$ AUROC(pooled), as means over 20 splits on identical subsets. $\\dagger$ = the 95\\% bootstrap interval of the mean difference over the splits lies below zero, $\\ddagger$ = above zero. These comparisons are descriptive: no test is applied and the intervals are not corrected for the number of cells.}\n"
            "\\end{sidewaystable*}\n")
    write(out, "tab_app_b6_naive.tex", text)

    # ---- numbers used in the running text ----
    nb = {}
    for key, name in STRATEGIES + MODELS:
        rows = [L[(key, c, n, k)] for c, n, k in cells]
        d = {"labels": dict(zip(LABELS4, counts4(rows))),
             "labels_by_cohort": {CNAME[c]: counts4([L[(key, cc, n, k)] for cc, n, k in cells if cc == c]) for c in GEO},
             "labels_by_n_t": {n: counts4([L[(key, c, nn, k)] for c, nn, k in cells if nn == n]) for n in NTS},
             "cells_mean_below_0": sum(float(r["mean_delta"]) < 0 for r in rows)}
        for tag, f in (("min", min), ("max", max)):
            r = f(rows, key=lambda r: float(r["mean_delta"]))
            d[tag] = {"cohort": CNAME[r["cohort"]], "n_t": r["n_t"], "k": r["k"], "mean_delta": round(float(r["mean_delta"]), 4),
                      "ci_low": round(float(r["ci_low"]), 4), "ci_high": round(float(r["ci_high"]), 4), "label": r["label"]}
        d["k10"] = {CNAME[c]: {n: [round(float(L[(key, c, n, "10")]["mean_delta"]), 4), L[(key, c, n, "10")]["label"]] for n in NTS if (key, c, n, "10") in L} for c in GEO}
        d["negative_cells"] = [(CNAME[r["cohort"]], r["n_t"], r["k"]) for r in rows if r["label"] == "negative transfer"]
        d["positive_cells_n_t"] = sorted(set(r["n_t"] for r in rows if r["label"] == "positive transfer"))
        nb[key] = d
    assert nb["pooled"]["labels"]["negative transfer"] == 0 and nb["select_on_source"]["labels"]["negative transfer"] == 0
    assert all(k in ("2", "3") for _, _, k in nb["source_only"]["negative_cells"])
    nb["target_only_auroc_k10"] = {CNAME[c]: {n: round(target[(c, n, "10")], 4) for n in NTS if (c, n, "10") in target} for c in GEO}
    nb["zero_shot_auroc"] = {CNAME[c]: {k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in zero[(c, k)].items()} for k in KS} for c in GEO}
    nb["zero_shot_k4_range_20"] = [round(min(zero[(c, "4")]["mean20"] for c in GEO), 4), round(max(zero[(c, "4")]["mean20"] for c in GEO), 4)]
    nb["zero_shot_delta"] = {CNAME[c]: {k: {n: [round(float(L[("source_only", c, n, k)]["mean_delta"]), 4), L[("source_only", c, n, k)]["label"]]
                                             for n in NTS if ("source_only", c, n, k) in L} for k in KS} for c in GEO}
    # change against the five-split design (item 6 of QD173)
    ch = {}
    for key, _ in STRATEGIES:
        ch[key] = {(c, n, k): float(L[(key, c, n, k)]["mean_delta"]) - float(E5[(key, c, n, k)]["diff_mean"]) for c, n, k in cells}
    two = [(key, c, n, k, v) for key in ("pooled", "select_on_source") for (c, n, k), v in ch[key].items()]
    big = max(two, key=lambda t: abs(t[4]))
    nb["change_5_to_20_splits"] = {
        "cells_abs_change_gt_0.01_pooled_and_select": sum(abs(t[4]) > 0.01 for t in two), "cells_pooled_and_select": len(two),
        "largest_change": {"strategy": big[0], "cohort": CNAME[big[1]], "n_t": big[2], "k": big[3], "change": round(big[4], 4)},
        "gse42127_n_t10_change_range": [round(min(ch[key][("gse42127", "10", k)] for key in ("pooled", "select_on_source") for k in KS), 4),
                                        round(max(ch[key][("gse42127", "10", k)] for key in ("pooled", "select_on_source") for k in KS), 4)],
        "sign_agreement_of_labelled_cells": {
            key: "%d of %d" % (sum((float(E5[(key, c, n, k)]["diff_mean"]) > 0) == (float(L[(key, c, n, k)]["mean_delta"]) > 0)
                                   for c, n, k in cells if L[(key, c, n, k)]["label"] in ("positive transfer", "negative transfer")),
                               sum(L[(key, c, n, k)]["label"] in ("positive transfer", "negative transfer") for c, n, k in cells))
            for key, _ in STRATEGIES}}
    assert all(v.split(" of ")[0] == v.split(" of ")[1] for v in nb["change_5_to_20_splits"]["sign_agreement_of_labelled_cells"].values())
    flag = lambda r: "positive" if r["positive_transfer"] == "True" else "negative" if r["negative_transfer"] == "True" else "none"
    cross = {}
    for key, _ in STRATEGIES:
        for c, n, k in cells:
            cross.setdefault(key, {}).setdefault(flag(E5[(key, c, n, k)]), dict.fromkeys(LABELS4, 0))[L[(key, c, n, k)]["label"]] += 1
    nb["five_split_flag_vs_label"] = cross
    for key, _ in STRATEGIES:   # no cell changes direction between the two designs
        assert cross[key].get("positive", {}).get("negative transfer", 0) == 0 and cross[key].get("negative", {}).get("positive transfer", 0) == 0
    # the two model-based methods against the naive strategies (descriptive)
    for key, _ in MODELS:
        for ref in ("select_on_source", "pooled"):
            rows = [N[(key, ref, c, n, k)] for c, n, k in cells]
            dm = [float(r["mean_delta"]) for r in rows]
            nb.setdefault("vs_naive", {})["%s_vs_%s" % (key, ref)] = {
                "mean_of_55": round(statistics.mean(dm), 4), "min": round(min(dm), 4), "max": round(max(dm), 4),
                "cells_interval_below_0": sum(float(r["ci_high"]) < 0 for r in rows), "cells_interval_above_0": sum(float(r["ci_low"]) > 0 for r in rows)}
    dg = {}
    for key, _ in MODELS:
        rows = [r for r in degen if r["method"] == key]
        assert len(rows) == 55 and all(r["n_runs"] == "80" for r in rows)
        dg[key] = {"runs": sum(int(r["n_runs"]) for r in rows), "error_rate_max": max(float(r["error_rate"]) for r in rows),
                   "filled_rate_max": max(float(r["filled_rate"]) for r in rows), "cells_with_fill": sum(float(r["filled_rate"]) > 0 for r in rows),
                   "same_as_stage1_min": min(float(r["same_as_stage1_topk_rate"] or "nan") for r in rows) if key == "ptlasso" else None,
                   "same_as_stage1_max": max(float(r["same_as_stage1_topk_rate"] or "nan") for r in rows) if key == "ptlasso" else None,
                   "same_as_stage1_mean": round(statistics.mean(float(r["same_as_stage1_topk_rate"]) for r in rows), 4) if key == "ptlasso" else None}
    nb["degeneracy"] = dg
    NUMBERS["transfer_20_splits"] = nb

    if derived and os.path.isdir(derived):   # cross-check with the advisor's derived numbers of 19/11/2026
        draft = rd(os.path.join(derived, "table19_b1_draft_numbers.csv"))
        for r in draft:
            if r["cohort"] == "all cohorts":
                continue
            for n in NTS:
                v = r["target_only_auroc_k10_nT" + n]
                if v:
                    assert abs(float(v) - target[(r["cohort"], n, "10")]) < 5.1e-4, (r["cohort"], n)
                    cell = r["delta_k10_nT" + n]
                    mine = L[(r["strategy"], r["cohort"], n, "10")]
                    assert cell[:6] == "%+.3f" % float(mine["mean_delta"]) and cell[6:] == (MARK[mine["label"]] or "?"), (cell, mine["mean_delta"], mine["label"])
        vs = {(r["strategy"], r["reference"]): r for r in rd(os.path.join(derived, "b6_vs_naive_descriptive_summary.csv"))}
        for (key, ref), r in vs.items():
            m = nb["vs_naive"]["%s_vs_%s" % (key, ref)]
            assert (abs(m["mean_of_55"] - float(r["mean_of_mean_delta"])) < 1e-4 and m["cells_interval_below_0"] == int(r["cells_ci_below_0"])
                    and m["cells_interval_above_0"] == int(r["cells_ci_above_0"])), (key, ref, m, r)
        zs = rd(os.path.join(derived, "zero_shot_numbers_b1.csv"))
        for r in zs:
            z = zero[(r["cohort"], r["k"])]
            assert abs(z["mean20"] - float(r["auroc_mean_20splits"])) < 5.1e-4 and abs(z["mean5"] - float(r["auroc_mean_splits0to4"])) < 5.1e-4
            assert z["below05"] == int(r["splits_below_0.5_of_20"])
        # 21/11/2026: the result of this optional cross-check is printed and not written to numbers_supplement.json, so that
        # the file is the same with and without the folder --derived (the code repository does not hold that folder).
        print("cross-check with %s: transfer labels, comparisons with the naive strategies and zero-shot values agree" % derived)


def table_nt_star(b1dir, official, out, derived=None):
    """n_T* of the five methods rerun over 20 splits, with the rule of the published table (smallest n_T within 0.02 of the pool value)."""
    order = {"5": 0, "10": 1, "20": 2, "40": 3, "pool": 9}
    b1 = rd(os.path.join(b1dir, "B1_tables", "b1_e1b_split_summary.csv"))
    pub = rd(os.path.join(official, "e1b_learning_curves.csv"))
    methods = ["f_test", "l1_lr", "ktsp", "random_k", "ihc_4"]

    def star(rows):
        rows = sorted((r for r in rows if r["auroc_mean"] not in ("", "nan")), key=lambda r: order[r["n_t"]])
        pool = [r for r in rows if r["n_t"] == "pool"]
        if not pool:
            return None, None
        ref = float(pool[0]["auroc_mean"])
        for r in rows:
            if r["n_t"] != "pool" and float(r["auroc_mean"]) >= ref - 0.02:
                return r["n_t"], ref
        return ">", ref

    res = {}
    for m in methods:
        for c in GEO:
            for k in KS:
                r20 = [r for r in b1 if r["method"] == m and r["cohort"] == c and r["k"] == k]
                r5 = [r for r in pub if r["method"] == m and r["cohort"] == c and r["k"] == k]
                s20, s5 = star(r20)[0], star(r5)[0]
                if s20 is None and s5 is None:
                    continue
                stored = set(r["n_t_star"] for r in r5)
                assert len(stored) == 1
                st = stored.pop()
                assert (st == "" and s5 == ">") or (st != "" and str(int(float(st))) == s5), (m, c, k, st, s5)
                res[(m, c, k)] = (s5, s20)
    assert len(res) == 76
    same = sum(a == b for a, b in res.values())
    fmt = lambda s, c: ("$>$" + TOP_LEVEL[c]) if s == ">" else s
    lines = []
    for mi, m in enumerate(methods):
        if mi:
            lines.append("\\addlinespace[2pt]")
        first = True
        for k in KS:
            if not any((m, c, k) in res for c in GEO):
                continue
            cs = []
            for c in GEO:
                a, b = res[(m, c, k)]
                cs.append("%s / %s" % (fmt(a, c), fmt(b, c)) if a == b else "\\textit{%s / %s}" % (fmt(a, c), fmt(b, c)))
            lines.append("%s & %s & %s \\\\" % (NAME[m] if first else "", k, " & ".join(cs)))
            first = False
    text = ("% source: e1b_learning_curves.csv (auroc_mean, n_t_star; five splits, label [v03-11]); B1_tables/b1_e1b_split_summary.csv (auroc_mean over 20 splits), commit 61f6252; rule of the published table: smallest n_T whose mean AUROC is within 0.02 of the pool value\n"
            "% generated by Scripts/make_supplementary_tables.py; do not edit by hand.  21/11/2026 (QD227, QD237).  Table of Online Resource 1 (ESM_1.tex).\n"
            "\\begin{table*}[tp]\n\\centering\n\\scriptsize\n\\caption{E1b: minimum sufficient target size $n_T^{*}$ over five and over 20 splits}\n\\label{tab:app_nt_star}\n"
            "\\setlength{\\tabcolsep}{6pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{lccccc}\n\\toprule\n"
            "\\textbf{Method} & $\\boldsymbol{k}$ & \\textbf{GSE10245} & \\textbf{GSE30219} & \\textbf{GSE37745} & \\textbf{GSE42127} \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
            "\\tablenote{Each cell gives $n_T^{*}$ over the five splits of the few-shot design and, after the slash, over 20 splits (seeds 42--61), for the three selectors and the two reference panels that were rerun on 20 splits. $n_T^{*}$ is the smallest number of labelled patients per class whose mean AUROC is within 0.02 of the value obtained with the whole pool; $>$ = no tested level below the pool reaches it. Cells in italics differ between the two designs (%d of %d). $k$-TSP exists for even $k$ only, and IHC-4 at $k = 4$.}\n"
            "\\end{table*}\n" % (len(res) - same, len(res)))
    write(out, "tab_app_nt_star.tex", text)
    NUMBERS["nt_star_20_splits"] = {"cells_with_a_value": len(res), "cells_same": same, "cells_different": len(res) - same,
                                    "k10": {"%s %s" % (NAME[m], CNAME[c]): list(v) for (m, c, k), v in res.items() if k == "10"},
                                    "k10_same": sum(a == b for (m, c, k), (a, b) in res.items() if k == "10"),
                                    "k10_cells": sum(k == "10" for (m, c, k) in res),
                                    "different_by_method": {NAME[m]: sum(a != b for (mm, c, k), (a, b) in res.items() if mm == m) for m in methods}}
    if derived and os.path.isdir(derived):
        adv = rd(os.path.join(derived, "e1b_nt_star_5_vs_20_splits.csv"))
        n_adv_same = 0
        for r in adv:
            key = (r["method"], r["cohort"], r["k"])
            conv = lambda s: ">" if s == "" else str(int(float(s)))
            if key in res:
                assert res[key] == (conv(r["n_t_star_5splits"]), conv(r["n_t_star_20splits"])), (key, res[key], r)
            else:
                assert r["auroc_pool_5splits"] == "" and r["auroc_pool_20splits"] == ""   # k-TSP at odd k: no value in either design
            n_adv_same += r["n_t_star_5splits"] == r["n_t_star_20splits"]
        print("cross-check with %s: n_T* agrees in every cell; the file counts %d of %d rows as unchanged, which includes the %d"
              " cells of k-TSP at an odd budget that hold no value in either design" % (derived, n_adv_same, len(adv), len(adv) - len(res)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default=os.path.join(PROJECT, "input", "results", "_received_2026-11-03", "tables_official_2026-11-03"))
    ap.add_argument("--extra", default=os.path.join(PROJECT, "input", "results", "_received_2026-11-16", "tables_survey-v3-dmkd.2"))
    ap.add_argument("--k2", default=os.path.join(PROJECT, "input", "results", "_received_2026-11-09", "E7_k2_evidence"))
    ap.add_argument("--b1", default=os.path.join(PROJECT, "input", "results", "_received_2026-11-19", "tables_dmkd-revision_61f6252"))
    ap.add_argument("--derived", default=os.path.join(PROJECT, "input", "results", "_received_2026-11-19", "derived"),
                    help="optional: the advisor's derived numbers of 19/11/2026, used for cross-checks only")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "Tables"))
    a = ap.parse_args()
    tables_e1_marks(a.official, a.out)
    tables_e1_secondary(a.official, a.out)
    tables_e1b(a.official, a.out)
    table_ablation(a.official, a.out)
    table_kstar(a.official, a.out)
    table_sa1(a.extra, a.official, a.out)
    tables_sa2(a.extra, a.out)
    tables_sa3(a.extra, a.out)
    tables_sa4(a.extra, a.k2, a.out)
    tables_e8(a.extra, a.official, a.out)
    tables_transfer20(a.b1, a.official, a.out, a.derived)
    table_nt_star(a.b1, a.official, a.out, a.derived)
    with open(os.path.join(a.out, "numbers_supplement.json"), "w", encoding="utf-8") as f:
        json.dump(NUMBERS, f, indent=1, ensure_ascii=False, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("written %d table files and numbers_supplement.json in %s" % (len(WRITTEN), a.out))
    for n in WRITTEN:
        print("  " + n)


if __name__ == "__main__":
    main()
