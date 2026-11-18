"""B7 (TCGA-LUAD EGFR): descriptive confounder checks run by the advisor on 16/11/2026 from tag survey-v3-dmkd.2 (9ed315a).
Usage (from the root of a clone of the survey repository at the tag):
    python b7_confounder_check.py results/survey_v2
Prints: AUROC of the tissue source site alone (one-hot TSS, LR, stratified 5x5 CV, seeds 42-46), AUROC of sex alone,
and the share of k = 10 panels that contain a sex-chromosome gene or EGFR itself."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
R = sys.argv[1]
m = pd.read_csv(f"{R}/B7_data/experiment-data-b7/tcga_luad_egfr/metadata.csv")
y = m["label"].astype(int).to_numpy()
X = pd.get_dummies(m["sample_id"].str.split("-").str[1]).to_numpy(float)
aucs = []
for seed in (42, 43, 44, 45, 46):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
print(f"TSS-only AUROC: mean {np.mean(aucs):.3f}, SD {np.std(aucs, ddof=1):.3f} over {len(aucs)} folds")
print(f"sex-only AUROC (female = 1): {roc_auc_score(y, (m['sex'] == 'female').astype(int)):.3f}; mutant female {int(((m.sex=='female')&(y==1)).sum())}/{int(y.sum())}")
SEX = {"XIST", "TSIX", "RPS4Y1", "RPS4Y2", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y", "ZFY", "TXLNGY", "NLGN4Y", "PRKY", "TTTY15"}
for co in ("tcga_luad_egfr", "gse31210_egfr"):
    g = pd.read_csv(f"{R}/E1_feature_selection_b7/{co}/selected_genes.csv")
    g = g[g["k"] == 10]
    panels = g.groupby(["method", "repeat", "fold"])
    n = panels.ngroups
    sex = int(panels["gene"].apply(lambda s: bool(set(s) & SEX)).sum())
    egfr = int(panels["gene"].apply(lambda s: "EGFR" in set(s)).sum())
    print(f"{co}: {n} panels at k = 10; with a sex-chromosome gene {sex}; containing EGFR {egfr} ({egfr / n:.0%})")
