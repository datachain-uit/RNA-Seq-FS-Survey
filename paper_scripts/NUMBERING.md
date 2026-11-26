# Sections, tables and figures of the manuscript

Numbers, titles, source files and LaTeX labels of the manuscript of 26/11/2026: the article (60 pages) and its supplementary
file, Online Resource 1 (96 pages). The files of the manuscript lie in one directory. The last column names the script of
`paper_scripts/` that writes the numbers of a table, with the name of its output file when it differs from the file of the
manuscript (REPRODUCE.md, Sections 4 and 6.3). The figures belong to the manuscript and are not written by this repository.

## Article: sections

| Section | Title | Label |
|---|---|---|
| 1 | Introduction | `sec:intro` |
| 2 | Related Surveys | `sec:surveys` |
| 3 | Gene Selection under a Panel Budget | `sec:impact` |
| 3.1 | Why select genes, and why under a budget | `sec:why` |
| 3.2 | Families of selectors and how they handle the budget | `sec:families` |
| 3.3 | Computational cost: search space, scoring function and budget | `sec:complexity_theory` |
| 3.4 | Stability: concept, factors and index | `sec:stability_theory` |
| 3.5 | Few labels and another platform: transfer, panel design and harmonisation | `sec:transfer_background` |
| 4 | Benchmark Design | `sec:framework` |
| 4.1 | Task, budget and reference points | `sec:problem` |
| 4.2 | Cohorts and gene space | `sec:data` |
| 4.3 | Selectors under a hard budget | `sec:methods` |
| 4.4 | Leak-free protocol and classifier | `sec:protocol` |
| 4.5 | Experiments by research question | `sec:experiments` |
| 4.6 | Statistical analysis | `sec:stats` |
| 5 | Results | `sec:results` |
| 5.1 | Performance under a budget (RQ1) | `sec:results_rq1` |
| 5.1.1 | Signal check | `sec:results_e0` |
| 5.1.2 | Performance at fixed budgets | `sec:results_e1` |
| 5.1.3 | Minimum sufficient budget | `sec:results_kstar` |
| 5.1.4 | Reference panels and robustness | `sec:results_panels` |
| 5.2 | Computational cost (RQ2) | `sec:cost_results` |
| 5.3 | Stability and the three-way trade-off (RQ3) | `sec:stability_results` |
| 5.3.1 | Stability under subsampling, redundancy and randomness | `sec:results_e4` |
| 5.3.2 | Performance, stability and cost together | `sec:results_e6` |
| 5.4 | Few-shot selection and transfer (RQ4) | `sec:results_rq4` |
| 5.4.1 | Few-shot learning curves | `sec:results_e1b` |
| 5.4.2 | Transfer from the RNA-seq source | `sec:results_e7` |
| 5.5 | A second task: EGFR mutation in lung adenocarcinoma | `sec:results_e8` |
| 6 | Discussion | `sec:discussion` |
| 6.1 | What the four answers say together | `sec:answers` |
| 6.2 | A decision framework | `sec:decision_framework` |
| 6.3 | Limitations | `sec:limitations` |
| 7 | Conclusion | `sec:conclusion` |

## Article: tables

| Table | Title | File of the manuscript | Label | Written by |
|---|---|---|---|---|
| Table 1 | Criteria of the comparison covered by the 59 earlier reviews and empirical comparisons found by the systematic search, with the present survey | `tab_table1_counts.tex` | `tab:survey_summary` | record of the systematic search (Online Resource 1, Section S1); not part of this repository |
| Table 2 | Qualitative properties attributed to the three classical families by the literature | `3-Survey.tex` | `tab:fs_extended_summary` | written by hand; no result file |
| Table 3 | Positioning of transfer, panel-design and harmonisation approaches against the requirements of budget-constrained diagnosis | `3-Survey.tex` | `tab:tl_positioning` | written by hand; no result file |
| Table 4 | Experiments of the benchmark by research question | `4-Benchmark.tex` | `tab:experiments` | written by hand; no result file |
| Table 5 | Cohorts after quality control | `tab_datasets.tex` | `tab:datasets` | `make_tables.py` (`tab_datasets.tex`) |
| Table 6 | Gene selectors and reference panels of the benchmark | `4-Benchmark.tex` | `tab:methods` | written by hand; no result file |
| Table 7 | E1 at the central budget k = 10: AUROC of LR-L2 | `tab_e1_k10.tex` | `tab:e1_k10` | `make_tables.py` (`tab_e1_k10.tex`) |
| Table 8 | Minimum sufficient budget k^* | `tab_kstar.tex` | `tab:kstar` | `make_tables.py` (`tab_kstar.tex`) |
| Table 9 | E2: cost of the selection step at a fixed point of the TCGA cohort | `tab_e2.tex` | `tab:e2` | `make_tables.py` (`tab_e2.tex`) |
| Table 10 | E7: transfer from the TCGA source at k = 10 and labels of all cells over 20 splits | `tab_e7_20splits.tex` | `tab:e7` | `make_supplementary_tables.py` |
| Table 11 | E8: EGFR mutation in lung adenocarcinoma at k = 10 | `tab_e8.tex` | `tab:e8` | `make_supplementary_tables.py` |

## Article: figures

| Fig. | Title | File of the manuscript | Label | figure of the manuscript; not part of this repository |
|---|---|---|---|---|
| Fig. 1 | Workflow of the benchmark | `Fig1.pdf` | `fig:pipeline` | figure of the manuscript; not part of this repository |
| Fig. 2 | Structure of the survey | `Fig2.pdf` | `fig:framework` | figure of the manuscript; not part of this repository |
| Fig. 3 | Workflow of the three classical families of selectors | `Fig3.pdf` | `fig:fs_workflow` | figure of the manuscript; not part of this repository |
| Fig. 4 | Cost landscape of the surveyed gene selectors | `Fig4.pdf` | `fig:aspect_time_complex_fs` | figure of the manuscript; not part of this repository |
| Fig. 5 | E1: AUROC of LR-L2 as a function of the budget k on the five cohorts | `Fig5.pdf` | `fig:e1_curves` | figure of the manuscript; not part of this repository |
| Fig. 6 | E4: Nogueira index on TCGA (left) and GSE37745 (right) | `Fig6.pdf` | `fig:e4_stability` | figure of the manuscript; not part of this repository |
| Fig. 7 | E6: AUROC, stability and cost at k = 10 on TCGA and GSE37745 | `Fig7.pdf` | `fig:e6_tradeoff` | figure of the manuscript; not part of this repository |
| Fig. 8 | E1b: AUROC as a function of the labelled target size n_T at k = 10 | `Fig8.pdf` | `fig:e1b_curves` | figure of the manuscript; not part of this repository |
| Fig. 9 | E7: difference in AUROC between each naive transfer strategy and target-only L1-LR over 20 splits | `Fig9.pdf` | `fig:e7_heatmap` | figure of the manuscript; not part of this repository |
| Fig. 10 | Decision framework for selecting a gene panel under a budget | `Fig10.pdf` | `fig:fs-decision-framework` | figure of the manuscript; not part of this repository |

## Online Resource 1: sections

| Section | Title | Label |
|---|---|---|
| S1 | Systematic search for Table 1 of the article | `sup:evidence` |
| S1.1 | Sources, queries and criteria | `sup:search` |
| S1.2 | Selection and agreement | `sup:selection` |
| S1.3 | Rating of the included works | `sup:rating` |
| S1.4 | Passages behind the marks of the new columns | `sup:evidence_list` |
| S1.5 | Studies of the LUAD/LUSC distinction | `sup:blockc` |
| S2 | Survey details: cost, stability and neural selectors | `sup:survey` |
| S2.1 | Cost of the surveyed selectors | `sup:cost_theory` |
| S2.2 | Stability as an evaluation criterion | `sup:stability_theory` |
| S2.3 | Evaluation regimes of the neural selectors | `sup:neural` |
| S3 | Details of the benchmark design | `sup:design` |
| S3.1 | Cohorts: processing and quality control | `sup:design_data` |
| S3.2 | Settings of the selectors | `sup:design_selectors` |
| S3.3 | Classifier, few-shot and transfer protocol | `sup:design_fewshot` |
| S3.4 | Cost and stability measurement | `sup:design_cost_stability` |
| S3.5 | Statistical analysis: pooling of k^* and transfer labels | `sup:design_stats` |
| S3.6 | Designs of the sensitivity analyses and of the second task | `sup:design_sensitivity` |
| S3.7 | Leakage control, sanity check and software | `sup:design_checks` |
| S4 | Performance under a budget (RQ1): complete results | `sup:rq1` |
| S4.1 | Signal check (E0) | `sup:rq1_e0` |
| S4.2 | AUROC at every budget, comparison with the full profile and practical equivalence | `sup:rq1_auroc` |
| S4.3 | Minimum sufficient budget and its sensitivity | `sup:rq1_kstar` |
| S4.4 | Reference panels, signatures at ten genes, ablation and secondary classifiers | `sup:rq1_panels` |
| S5 | Computational cost (RQ2): complete results | `sup:rq2` |
| S6 | Stability and three-way trade-off (RQ3): complete results | `sup:rq3` |
| S6.1 | Stability under subsampling (E4) | `sup:rq3_e4` |
| S6.2 | Redundancy, algorithmic randomness and agreement between methods (E5) | `sup:rq3_e5` |
| S6.3 | Performance, stability and cost together (E6) | `sup:rq3_e6` |
| S7 | Few-shot selection and transfer (RQ4): complete results | `sup:rq4` |
| S7.1 | Few-shot learning curves (E1b) | `sup:rq4_fewshot` |
| S7.2 | Transfer over 20 splits (E7) | `sup:rq4_transfer` |
| S7.3 | The two-gene zero-shot rule and its variants | `sup:rq4_zeroshot` |
| S8 | Second task: EGFR mutation in lung adenocarcinoma (E8), complete results | `sup:e8` |
| S9 | Notation | `sup:notation` |

## Online Resource 1: tables

| Table | Title | File of the manuscript | Label | Written by |
|---|---|---|---|---|
| Table S1 | Marks of the earlier works compared in Table 1 of the article | `tab_app_S1_matrix.tex` | `tab:table1_evidence` | record of the systematic search (Online Resource 1, Section S1); not part of this repository |
| Table S2 | Cost of common gene selectors across filter, wrapper, embedded and budget-constrained approaches | `tab_app_cost_methods.tex` | `tab:app_cost_methods` | written by hand; no result file |
| Table S3 | Classifiers used to score candidate panels in wrapper methods and as evaluation classifiers in this survey | `tab_app_scoring_models.tex` | `tab:app_scoring_models` | written by hand; no result file |
| Table S4 | E0: signal check | `tab_app_e0.tex` | `tab:app_e0` | `make_tables.py` (`tab_e0.tex`) |
| Table S5 | E1 on TCGA: AUROC by method and budget k | `tab_app_e1_tcga.tex` | `tab:app_e1_tcga` | `make_tables.py`, marks by `make_supplementary_tables.py` |
| Table S6 | E1 on GSE10245: AUROC by method and budget k | `tab_app_e1_gse10245.tex` | `tab:app_e1_gse10245` | `make_tables.py`, marks by `make_supplementary_tables.py` |
| Table S7 | E1 on GSE30219: AUROC by method and budget k | `tab_app_e1_gse30219.tex` | `tab:app_e1_gse30219` | `make_tables.py`, marks by `make_supplementary_tables.py` |
| Table S8 | E1 on GSE37745: AUROC by method and budget k | `tab_app_e1_gse37745.tex` | `tab:app_e1_gse37745` | `make_tables.py`, marks by `make_supplementary_tables.py` |
| Table S9 | E1 on GSE42127: AUROC by method and budget k | `tab_app_e1_gse42127.tex` | `tab:app_e1_gse42127` | `make_tables.py`, marks by `make_supplementary_tables.py` |
| Table S10 | E1: balanced accuracy of LR-L2 at k in {2, 5, 10} | `tab_app_e1_bacc.tex` | `tab:app_e1_bacc` | `make_supplementary_tables.py` |
| Table S11 | E1: Matthews correlation coefficient of LR-L2 at k in {2, 5, 10} | `tab_app_e1_mcc.tex` | `tab:app_e1_mcc` | `make_supplementary_tables.py` |
| Table S12 | E1: log-loss of LR-L2 at k in {2, 5, 10} | `tab_app_e1_logloss.tex` | `tab:app_e1_logloss` | `make_supplementary_tables.py` |
| Table S13 | E1: Brier score of LR-L2 at k in {2, 5, 10} | `tab_app_e1_brier.tex` | `tab:app_e1_brier` | `make_supplementary_tables.py` |
| Table S14 | Bayesian correlated t-test between the leading method and each competitor at k in {2, 5, 10} | `tab_app_rope.tex` | `tab:app_rope` | `make_tables.py` (`tab_rope.tex`) |
| Table S15 | SA1: the reference model under a wider grid of C and on the pre-filtered genes, and the resulting k^* | `tab_app_sens_reference.tex` | `tab:app_sens_reference` | `make_supplementary_tables.py` |
| Table S16 | Sensitivity of the pooled minimum sufficient budget k^* to the margin and to the pooling | `tab_app_kstar_sens.tex` | `tab:app_kstar_sens` | `make_supplementary_tables.py` |
| Table S17 | Knowledge panels (IHC-2, IHC-4) and published panels (Girard 2016, Wilkerson 2013) against the no-selection reference B0 | `tab_app_panels.tex` | `tab:app_panels` | `make_tables.py` (`tab_panels.tex`) |
| Table S18 | SA2: the two published signatures cut to ten genes, against the best ten-gene selector, IHC-4 and B0 | `tab_app_sig10.tex` | `tab:app_sig10` | `make_supplementary_tables.py` |
| Table S19 | Ablation without the four IHC marker genes | `tab_app_ablation.tex` | `tab:app_ablation` | `make_tables.py` (`tab_ablation.tex`) |
| Table S20 | Ablation without the four IHC marker genes at every budget | `tab_app_ablation_full.tex` | `tab:app_ablation_full` | `make_supplementary_tables.py` |
| Table S21 | E3: empirical scaling exponents of the selection time on TCGA | `tab_app_e3.tex` | `tab:app_e3` | `make_tables.py` (`tab_e3.tex`) |
| Table S22 | E4: stability at k = 10 | `tab_app_e4.tex` | `tab:app_e4` | `make_tables.py` (`tab_e4.tex`) |
| Table S23 | E5a: redundancy on GSE37745 (k = 10, f = 0.7, M = 30) | `tab_app_e5a.tex` | `tab:app_e5a` | `make_tables.py` (`tab_e5a.tex`) |
| Table S24 | E5b: algorithmic randomness on GSE37745 | `tab_app_e5b.tex` | `tab:app_e5b` | `make_tables.py` (`tab_e5b.tex`) |
| Table S25 | E6: three-way trade-off at k = 10 on the two stability cohorts | `tab_app_e6.tex` | `tab:app_e6` | `make_tables.py` (`tab_e6.tex`) |
| Table S26 | E1b: few-shot learning curves at k = 10 | `tab_app_e1b_k10.tex` | `tab:app_e1b_k10` | `make_tables.py` (`tab_e1b.tex`) |
| Table S27 | E1b: full learning curves at k = 10 | `tab_app_e1b.tex` | `tab:app_e1b` | `make_tables.py` (`tab_app_e1b.tex`) |
| Table S28 | E1b: minimum sufficient target size n_T^* over five and over 20 splits | `tab_app_nt_star.tex` | `tab:app_nt_star` | `make_supplementary_tables.py` |
| Table S29 | SA3: the pairs of k-TSP classified by their voting rule and by LR-L2 | `tab_app_ktsp_vote.tex` | `tab:app_ktsp_vote` | `make_supplementary_tables.py` |
| Table S30 | SA3 in the few-shot regime: AUROC of the voting rule and its difference to LR-L2 on the same pairs | `tab_app_ktsp_vote_fewshot.tex` | `tab:app_ktsp_vote_fewshot` | `make_supplementary_tables.py` |
| Table S31 | SA2 in the few-shot regime: AUROC of the ten-gene signatures at k = 10 | `tab_app_sig10_fewshot.tex` | `tab:app_sig10_fewshot` | `make_supplementary_tables.py` |
| Table S32 | E1b: Nogueira index of the few-shot panels at k = 10 on GSE10245 and GSE30219 | `tab_app_e1b_stability.tex` | `tab:app_e1b_stability` | `make_supplementary_tables.py` |
| Table S33 | E1b: Nogueira index of the few-shot panels at k = 10 on GSE37745 and GSE42127 | `tab_app_e1b_stability.tex` | `tab:app_e1b_stability_b` | `make_supplementary_tables.py` |
| Table S34 | E1b: balanced accuracy, log-loss and Brier score at k = 10 on GSE10245 and GSE30219 | `tab_app_e1b_secondary.tex` | `tab:app_e1b_secondary` | `make_supplementary_tables.py` |
| Table S35 | E1b: balanced accuracy, log-loss and Brier score at k = 10 on GSE37745 and GSE42127 | `tab_app_e1b_secondary.tex` | `tab:app_e1b_secondary_b` | `make_supplementary_tables.py` |
| Table S36 | E7: difference to target-only selection in every cell over 20 splits, with the difference over the five splits of the few-shot design | `tab_app_e7_20splits.tex` | `tab:app_e7` | `make_supplementary_tables.py` |
| Table S37 | E7: ptLasso and Trans-GLM against target-only selection in every cell over 20 splits | `tab_app_b6.tex` | `tab:app_b6` | `make_supplementary_tables.py` |
| Table S38 | E7: ptLasso and Trans-GLM against the two naive strategies that use target labels (descriptive) | `tab_app_b6_naive.tex` | `tab:app_b6_naive` | `make_supplementary_tables.py` |
| Table S39 | The two-gene zero-shot rule: class shifts of DSG3 and KRT5 and AUROC at k = 2 | `tab_app_zeroshot_k2.tex` | `tab:app_zeroshot_k2` | `make_supplementary_tables.py` |
| Table S40 | SA4: AUROC of the zero-shot strategy and of its two variants | `tab_app_zeroshot.tex` | `tab:app_zeroshot` | `make_supplementary_tables.py` |
| Table S41 | E8 on TCGA-LUAD: AUROC and AUPRC by method and budget k | `tab_app_e8_tcga.tex` | `tab:app_e8_tcga` | `make_supplementary_tables.py` |
| Table S42 | E8 on GSE31210: AUROC and AUPRC by method and budget k | `tab_app_e8_gse31210.tex` | `tab:app_e8_gse31210` | `make_supplementary_tables.py` |
| Table S43 | E8: balanced accuracy, MCC, log-loss and Brier score at k = 10 | `tab_app_e8_secondary.tex` | `tab:app_e8_secondary` | `make_supplementary_tables.py` |
| Table S44 | E8: AUROC and stability of the eleven non-wrapper selectors at k = 10 on the histology task and on the EGFR task | `tab_app_e8_rank.tex` | `tab:app_e8_rank` | `make_supplementary_tables.py` |
| Table S45 | Notation of the article | `tab_app_notation.tex` | `tab:app_notation` | written by hand; no result file |

## Online Resource 1: figures

| Fig. | Title | File of the manuscript | Label | figure of the manuscript; not part of this repository |
|---|---|---|---|---|
| Fig. S1 | PRISMA 2020 flow diagram of the search for Table 1 of the article (blocks A and B) | `FigS1.pdf` | `sup:prisma` | figure of the manuscript; not part of this repository |
| Fig. S2 | Pooled lower bound of AUROC(panel_k) - AUROC(B0) over the three GPL570 cohorts | `FigS2.pdf` | `sup:fig_kstar` | figure of the manuscript; not part of this repository |
| Fig. S3 | E3: wall time of the selection step on TCGA | `FigS3.pdf` | `sup:fig_e3` | figure of the manuscript; not part of this repository |
| Fig. S4 | E5: redundancy and overlap between methods on GSE37745 at k = 10 | `FigS4.pdf` | `sup:fig_e5` | figure of the manuscript; not part of this repository |

## Names of the repository folders in the paper

| Repository | Name in the paper |
|---|---|
| B4 | SA1 (reference model) |
| B5 | SA2 (published signatures cut to ten genes) |
| B2 | SA3 (voting rule of k-TSP) |
| B3 | SA4 (two variants of the zero-shot strategy) |
| B7 | E8 (second task: EGFR mutation in lung adenocarcinoma) |
| B1 | no separate name: "E7 (and E1b) over 20 test/pool splits" |
| B6 | no separate name: "two transfer methods with a model of the source (ptLasso, Trans-GLM)" in E7 |
