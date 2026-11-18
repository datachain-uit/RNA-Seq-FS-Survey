#!/usr/bin/env python3
"""check_figure4.py -- the exponents written in Figure 4 must equal the measured exponents of E3 (rounded to 2 decimals).

    python3 Scripts/check_figure4.py --tex Images/src/Figure_4_complexity.tex \
                                     --csv <tables_official_2026-11-03>/e3_exponents_vs_table5.csv \
                                     [--log Images/src/Figure_4_complexity.log]

Exit code 1 if a pair differs or a benchmarked method is missing from the figure.  With --log (the log of pdflatex on
the figure), also checks the layout from the node boxes that the figure writes to its log (PAGEBOX, NODEBOX lines):
every box inside the page and no two boxes overlapping (18/11/2026, O53).
"""
import argparse
import re
import sys

import pandas as pd

LABEL = {"F-test": "f_test", "MI": "mutual_info", "ReliefF": "relieff", "mRMR": "mrmr", "L1-LR": "l1_lr",
         "Elastic Net": "elastic_net", "RF importance": "rf_importance", "SVM-RFE": "svm_rfe", "abess": "abess",
         "Stability selection": "stability_selection", "$k$-TSP": "ktsp"}


def layout(log, tight=2.0):
    """Problems of the node boxes logged by the figure (TeX points): outside the page, overlapping."""
    num = r"(-?[\d.]+)pt"
    txt = open(log, encoding="latin-1").read()
    page = re.search(r"PAGEBOX: %s,%s,%s,%s" % (num, num, num, num), txt)
    boxes = [(m.group(1), [float(m.group(k)) for k in range(2, 6)])
             for m in re.finditer(r"NODEBOX (N\d+): %s,%s,%s,%s" % (num, num, num, num), txt)]
    if not page or not boxes:
        return ["no PAGEBOX/NODEBOX lines in %s" % log], []
    px0, py0, px1, py1 = [float(page.group(k)) for k in range(1, 5)]
    cm = 2.54 / 72.27
    probs, notes = [], []
    for name, (x0, y0, x1, y1) in boxes:
        if x0 < px0 - 0.1 or y0 < py0 - 0.1 or x1 > px1 + 0.1 or y1 > py1 + 0.1:
            probs.append("%s outside the page: centre (%.2f, %.2f) cm" % (name, (x0 + x1) / 2 * cm, (y0 + y1) / 2 * cm))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (na, a), (nb, b) = boxes[i], boxes[j]
            w = min(a[2], b[2]) - max(a[0], b[0])
            h = min(a[3], b[3]) - max(a[1], b[1])
            ca = ((a[0] + a[2]) / 2 * cm, (a[1] + a[3]) / 2 * cm)
            cb = ((b[0] + b[2]) / 2 * cm, (b[1] + b[3]) / 2 * cm)
            if w > 0 and h > 0:
                probs.append("%s (%.2f, %.2f) overlaps %s (%.2f, %.2f): %.1f x %.1f pt" % (na, *ca, nb, *cb, w, h))
            elif -tight < w and -tight < h:
                notes.append("%s (%.2f, %.2f) and %s (%.2f, %.2f) closer than %.0f pt" % (na, *ca, nb, *cb, tight))
    print("node boxes checked: %d, page %.2f x %.2f TeX pt" % (len(boxes), px1 - px0, py1 - py0))
    return probs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--log")
    a = ap.parse_args()
    tex = open(a.tex, encoding="utf-8").read()
    csv = pd.read_csv(a.csv, encoding="utf-8-sig").set_index("method")
    found, bad = {}, []
    for body in re.findall(r"\\node\[(?:filter|emb|con)\] at \([^)]*\) \{(.*?)\};", tex):
        name = re.split(r"\\\\| \(", body)[0].strip()
        m = re.search(r"(\d\.\d\d)\\,/\\,(\d\.\d\d)", body)
        if name not in LABEL or not m:
            bad.append("unrecognised box: %s" % body)
            continue
        found[LABEL[name]] = (m.group(1), m.group(2))
    for meth, (an, ap_) in found.items():
        want = ("%.2f" % csv.loc[meth, "alpha_n_measured"], "%.2f" % csv.loc[meth, "alpha_p_measured"])
        status = "OK " if (an, ap_) == want else "DIFF"
        print(f"{status} {meth:20s} figure {an} / {ap_}   table {want[0]} / {want[1]}")
        bad += [] if (an, ap_) == want else [meth]
    missing = sorted(set(csv.index) - set(found))
    if missing:
        bad.append("missing in the figure: %s" % missing)
    print("pairs checked:", len(found))
    if a.log:
        probs, notes = layout(a.log)
        for n in notes:
            print("note:", n)
        bad += probs
    if bad:
        print("PROBLEMS:", bad)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
