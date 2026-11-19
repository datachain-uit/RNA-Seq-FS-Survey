#!/usr/bin/env python3
"""
reanalysis_2026-11-12.py -- derived numbers of Sections 7.2 and 7.3 of the article (Kendall rank correlation between the
orderings of the selectors under LR-L2 and under the secondary classifiers; medians of the eleven data-driven selectors at
n_T = 5; published signatures against the best 50-gene panel) and the counts checked with them.  No experiment is run:
every value is computed from the official result tables (results/survey_v2/tables/).  The output is a Markdown report
(headings in Vietnamese, as in the authors' working notes).

Usage (from the repository root):
    python paper_scripts/reanalysis_2026-11-12.py --tables results/survey_v2/tables > reanalysis_2026-11-12.md

Repository copy of working/tools/reanalysis_2026-11-12.py of the authors (12/11/2026); only the input folder became a
parameter.  With --tables results/survey_v2/tables the report equals the authors' copy except the line that names the
input folder.
"""
import argparse
import collections
import csv
import itertools
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description="Derived numbers of Sections 7.2 and 7.3 from the official result tables.")
ap.add_argument("--tables", default=os.path.join(ROOT, "results", "survey_v2", "tables"),
                help="folder with e1_summary.csv, e1b_learning_curves_table.csv, ... (default: results/survey_v2/tables)")
SRC = ap.parse_args().tables
sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # same bytes on every platform (the report has Vietnamese text)
REF = {"ihc_2", "ihc_4", "panel_girard2016", "panel_wilkerson2013", "random_k", "b0"}


def rd(name):
    with open(os.path.join(SRC, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def tau_a(a, b):
    n, c, d = len(a), 0, 0
    for i, j in itertools.combinations(range(n), 2):
        x = (a[i] - a[j]) * (b[i] - b[j])
        c += x > 0
        d += x < 0
    return (c - d) / (n * (n - 1) / 2)


s = rd("e1_summary.csv")
cohorts = sorted(set(x["cohort"] for x in s))
print("# 10 — Phân tích lại từ kết quả đã lưu (12/11/2026)\n")
print("Sinh bởi `paper_scripts/reanalysis_2026-11-12.py` từ `%s/` "
      "([v03-11]). Không chạy thí nghiệm mới. Các số dưới đây là nguồn của những câu thêm vào bản thảo ngày 12/11/2026 "
      "(QĐ162–QĐ170; `working/03_claims.md` Mục V).\n" % os.path.relpath(os.path.abspath(SRC), ROOT).replace(os.sep, "/"))

print("## 1. Tương quan hạng Kendall giữa thứ tự các bộ chọn dưới LR-L2 và dưới bộ phân loại phụ (Mục 7.2)\n")
print("Nguồn: `e1_summary.csv` (cột classifier, auroc_mean); bộ chọn dữ liệu (không tính bảng tham chiếu, random-k, B0); "
      "k = 5 và 25; tau-a.\n")
print("| Bộ phân loại | Cohort | k | Số bộ chọn | Kendall tau |\n|---|---|---|---|---|")
summary = {}
for cl in ("rf", "svm_linear"):
    vals = []
    for c in cohorts:
        for k in ("5", "25"):
            rows = {x["method"]: float(x["auroc_mean"]) for x in s if x["cohort"] == c and x["k"] == k and x["classifier"] == cl and x["method"] not in REF}
            base = {x["method"]: float(x["auroc_mean"]) for x in s if x["cohort"] == c and x["k"] == k and x["classifier"] == "lr_l2" and x["method"] not in REF}
            ms = sorted(set(rows) & set(base))
            t = tau_a([base[m] for m in ms], [rows[m] for m in ms])
            vals.append(t)
            print("| %s | %s | %s | %d | %.2f |" % (cl, c, k, len(ms), t))
    summary[cl] = vals
for cl, vals in summary.items():
    print("\n%s: %d ô, nhỏ nhất %.2f, trung vị %.2f, lớn nhất %.2f." % (cl, len(vals), min(vals), st.median(vals), max(vals)))

print("\n## 2. Trung vị AUROC của 11 bộ chọn dữ liệu ở n_T = 5, k = 10 (Mục 7.3)\n")
print("Nguồn: `e1b_learning_curves_table.csv` (k = 10, cột 5).\n")
print("| Cohort | Số bộ chọn | Nhỏ nhất | Trung vị | Lớn nhất |\n|---|---|---|---|---|")
t = rd("e1b_learning_curves_table.csv")
for c in sorted(set(x["cohort"] for x in t)):
    v = sorted(float(x["5"].split()[0]) for x in t if x["cohort"] == c and x["k"] == "10" and x["method"] not in REF and x["5"])
    print("| %s | %d | %.3f | %.3f | %.3f |" % (c, len(v), v[0], st.median(v), v[-1]))

print("\n## 3. Chữ ký đã công bố so với bảng 50 gen tốt nhất của các bộ chọn (Mục 7.2)\n")
print("Nguồn: `e1_summary.csv` (lr_l2). Bản thảo dùng hiệu của các giá trị ba chữ số thập phân như in ở Table 18 và Tables S2–S6.\n")
print("| Cohort | Bảng 50 gen tốt nhất | AUROC | Girard (42) | Hiệu | Wilkerson (52) | Hiệu |\n|---|---|---|---|---|---|---|")
for c in cohorts:
    dd = [(float(x["auroc_mean"]), x["method"]) for x in s if x["cohort"] == c and x["classifier"] == "lr_l2" and x["method"] not in REF and x["k"] == "50"]
    b, bm = max(dd)
    g = [float(x["auroc_mean"]) for x in s if x["cohort"] == c and x["method"] == "panel_girard2016"][0]
    w = [float(x["auroc_mean"]) for x in s if x["cohort"] == c and x["method"] == "panel_wilkerson2013"][0]
    print("| %s | %s | %.3f | %.3f | %+.4f | %.3f | %+.4f |" % (c, bm, b, g, g - b, w, w - b))

print("\n## 4. Các số đếm và sự kiện đã kiểm (không phải số mới)\n")
r = rd("e1_tests_vs_random_b0.csv")
cnt = collections.defaultdict(lambda: [0, 0])
for x in r:
    if x["method"] in REF:
        continue
    c = cnt[(x["cohort"], int(x["k"]))]
    c[1] += 1
    c[0] += x["beats_random_holm"] == "True"
print("Số bộ chọn dữ liệu vượt random-k sau Holm, theo cohort và k (`e1_tests_vs_random_b0.csv`, cột beats_random_holm):\n")
ks = sorted(set(k for _, k in cnt))
print("| Cohort | " + " | ".join("k=%d" % k for k in ks) + " |\n|---|" + "---|" * len(ks))
for c in cohorts:
    print("| %s | " % c + " | ".join("%d/%d" % tuple(cnt[(c, k)]) if (c, k) in cnt else "–" for k in ks) + " |")
p = rd("e1_kstar_pooled_by_k.csv")
print("\nDị biệt giữa ba cohort GPL570 (`e1_kstar_pooled_by_k.csv`, cột I2): %d dòng, I2 lớn nhất = %s; trọng số lớn nhất của một cohort từ %.2f đến %.2f."
      % (len(p), max(float(x["I2"]) for x in p), min(float(x["max_weight"]) for x in p), max(float(x["max_weight"]) for x in p)))
print("\nChuỗi kiểm định của từng phương pháp (`e1_kstar_pooled.csv`, cột k_sequence_used):\n")
for x in rd("e1_kstar_pooled.csv"):
    print("- %s: k*_IVW = %s, k*_both = %s, chuỗi %s" % (x["method"], x["k_star_ivw"] or "–", x["k_star_both"] or "–", x["k_sequence_used"] or "–"))
e7 = rd("e7_transfer_table.csv")
print("\nCờ chuyển giao (`e7_transfer_table.csv`):\n")
for sname in sorted(set(x["strategy"] for x in e7)):
    xs = [x for x in e7 if x["strategy"] == sname]
    mn = min(xs, key=lambda x: float(x["diff_mean"]))
    print("- %s: %d ô; dương %d; âm %d; số ô có hiệu trung bình âm %d; hiệu nhỏ nhất %.4f (%s, n_T = %s, k = %s)."
          % (sname, len(xs), sum(x["positive_transfer"] == "True" for x in xs), sum(x["negative_transfer"] == "True" for x in xs),
             sum(float(x["diff_mean"]) < 0 for x in xs), float(mn["diff_mean"]), mn["cohort"], mn["n_t"], mn["k"]))
b = rd("e1_bayes_rope.csv")
dec = [x for x in b if float(x["p_best_better"]) > 0.95 or float(x["p_other_better"]) > 0.95]
print("\nPhân tích Bayes (`e1_bayes_rope.csv`): %d so sánh (k ∈ %s; %d cohort); quyết định (xác suất > 0.95): %s."
      % (len(b), sorted(set(x["k"] for x in b), key=int), len(set(x["cohort"] for x in b)),
         "; ".join("%s vs %s, %s, k = %s" % (x["best"], x["other"], x["cohort"], x["k"]) for x in dec)))
a = rd("e1_ablation_noihc.csv")
print("\nSố gen IHC trung bình trong bảng 10 gen (`e1_ablation_noihc.csv`, cột ihc_genes_per_panel):\n")
for mth in ("f_test", "elastic_net", "relieff", "l1_lr", "svm_rfe"):
    print("- %s: " % mth + ", ".join("%s %.2f" % (x["cohort"], float(x["ihc_genes_per_panel"])) for x in a if x["method"] == mth and x["k"] == "10"))
b0 = {x["cohort"]: float(x["auroc_mean"]) for x in s if x["method"] == "b0" and x["classifier"] == "lr_l2"}
top = sorted(((float(x["auroc_mean"]), x["method"]) for x in s if x["cohort"] == "gse42127" and x["k"] == "10" and x["classifier"] == "lr_l2" and x["method"] not in REF), reverse=True)
print("\nGSE42127, k = 10: B0 = %.3f; các bảng có AUROC trung bình cao hơn B0: %s."
      % (b0["gse42127"], ", ".join("%s %.3f" % (mth, v) for v, mth in top if v > b0["gse42127"])))
