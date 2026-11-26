# Glossary of the Vietnamese strings in the records

The documentation of this repository is in English. Some files written while the experiments ran contain Vietnamese,
the working language of the authors: notes in `config.json`, a few columns of the result tables, the logs of the
Kaggle sessions, the exclusion lists and the quality-control files. These files are kept byte for byte as produced:
they are the records of the runs, and the `config.json` files and the split manifests record the SHA-256 of many of
them. This glossary translates the strings that occur in them. The same holds for
`survey_v2/panels/published_panels.csv` (Section 5), whose SHA-256 is recorded in the `config.json` of the runs that
read it. The comments of the code also use Vietnamese and the codes of Section 1.

## 1. Codes of the authors' working log

The records cite the working log of the project by short codes. They identify a decision or an issue; they carry no
information that is not stated in the paper, `REPRODUCE.md` or the data cards.

| Code | Meaning |
|---|---|
| `QĐnn` | Decision number nn of the authors' log (QĐ = *quyết định*, decision) |
| `Ynn`, `Onn`, `G-nn` | Issue numbers of the log |
| `Mnn` | Item nn of the experiment plan of the project |
| `KBTN` | The experiment plan of the project (*kế hoạch thực nghiệm*) |
| `Review dd/mm` | Review meeting of the authors on that date |
| `HANDOVER` | Handover note of a batch of runs |
| `YEU-CAU-DOI-CODE-<date>` | Request to the members who ran the code, on that date (*yêu cầu đội code*) |
| `spec ... §B2` | Section B2 of the specification of the sensitivity analyses (12 November 2026) |
| `code J` | The cross-check of the GEO labels against the original series matrices (`scripts/check_label_crosscheck.py`) |
| `v3.1.6`, `v3.1.7` | Versions of the code of the benchmark (October and November 2026) |

Decisions that occur in the records:

| Code | Content |
|---|---|
| QĐ18 | E2 is one reference session for every method; the parts of E3 run in separate sessions and are calibrated against it |
| QĐ21 | Exclusion of the near-duplicate pair GSM1019246/GSM1019247 of GSE37745 |
| QĐ22 | Two-axis sex check of the GEO cohorts (XIST and the mean of five Y genes) |
| QĐ22b | Per-gene vote of five Y genes for TCGA, which has no XIST |
| QĐ27 | Timing without tracemalloc; peak memory of a child process measured by VmHWM |
| QĐ172 to QĐ175 | Specifications of the analyses in the folders `B2_*` (SA3), `B3_*` (SA4), `B4_*` (SA1) and `B5_*` (SA2) |
| QĐ176 | ptLasso and Trans-GLM in E7 (folders `B6_*`) |
| QĐ177 | The second task E8 (EGFR mutation; folders `B7_*` and `*_b7`) |
| QĐ185, QĐ186 | Label rules of `gse31210_egfr` and `tcga_luad_egfr` |
| QĐ188 | The two EGFR data sets are appended at the end of `DATASETS`, so the seeds of the five main cohorts do not change |

## 2. Values in the columns of CSV files

| File | Column | Value | English |
|---|---|---|---|
| `splits/qc/label_crosscheck_*.csv` | `status` | `giu` | kept |
| | | `cach_ly_QC` | excluded by quality control |
| | | `khong_thuoc_bai_toan` | not part of the task (other histology) |
| `splits/qc/label_crosscheck_*.csv` | `qc_decision` | `QĐ22 2026-10-29`, `QĐ21 2026-10-29` | decision QĐ22 (or QĐ21) of 29 October 2026 |
| `splits/*/excluded_samples_*.csv` and the copies in `results/survey_v2/B7_data/` | `decision` | `QĐ22 2026-10-29`, `QĐ21 2026-10-29` | as above |
| | `reason` | `QĐ22 sex check: inconsistent — XIST và gen Y cùng cao (meta_sex = male); không xác minh được danh tính mẫu` | QĐ22 sex check: inconsistent; XIST and the Y genes are both high (meta_sex = male); the identity of the sample cannot be verified |
| | | `... (meta_sex = không có) ...` | ... (meta_sex = not reported) ... |
| | | `QĐ22 sex check: opposite — XIST và gen Y đều trái metadata (meta_sex = male, biểu hiện = F-like); nhãn mô học cùng dòng metadata nên không tin được` | QĐ22 sex check: opposite; XIST and the Y genes both contradict the metadata (meta_sex = male, expression = F-like); the histology label comes from the same metadata row and cannot be trusted |
| | | `... (meta_sex = female, biểu hiện = M-like) ...` | ... (meta_sex = female, expression = M-like) ... |
| | | `QĐ22b sex check: opposite — gen Y (bỏ phiếu từng gen; bộ không có XIST) trái metadata (meta_sex = female, biểu hiện = M-like(Y)); nhãn mô học cùng dòng metadata nên không tin được` | QĐ22b sex check: opposite; the Y genes (vote of each gene; the data set has no XIST) contradict the metadata (meta_sex = female, expression = M-like(Y)); the histology label comes from the same metadata row and cannot be trusted |
| | | `near-duplicate pair, r = 0,9925; ...` | (English; the decimal comma is Vietnamese usage: r = 0.9925) |
| `results/survey_v2/tables/e1_kstar_pooled.csv` (`verdict_vs_b0`) and `results/survey_v2/B4_tables/b4_kstar_sensitivity.csv` (`verdict_vs_b0_b0`, `verdict_vs_b0_b0_p5000_wide`) | verdict columns | `ủng hộ (k*_gộp <= 10 theo cả hai cách gộp)` | supported (pooled k* <= 10 under both pooling methods) |
| | | `ủng hộ một phần (chỉ bản IVW đạt; ghi rõ cohort chi phối)` | partly supported (only the IVW version passes; the dominating cohort is named) |
| | | `không xác định (không đạt ở các k đã thử)` | undetermined (not reached at the values of k tried) |
| | | `bác bỏ (chuỗi không tới được k = 25)` | rejected (the sequence does not reach k = 25) |
| | | `chưa kết luận` | inconclusive |
| `results/survey_v2/B7_tables/b7_kstar.csv` | `reading_delta_0.01`, `reading_delta_0.02` | `không đạt ở k = 50 (mức đầu của chuỗi)` | not reached at k = 50 (the first level of the sequence) |
| `results/survey_v2/B7_gate/b7_gate_table.csv` | `next_step` | `E1 (and E4 for TCGA-LUAD) per QĐ177` | (English) |
| `results/survey_v2/*E1b*/*/e1b_stability.csv` | `note` | `CI tham khảo: các tập con cùng split chồng lấn (lồng nhau theo n_T)` | CI for reference only: the subsets of one split overlap (nested along n_T) |
| `results/survey_v2/tables/e3_exponents_vs_table5.csv` | `complexity_table5` | `O(p^2 n) [cài đặt tham lam k gen: O(k n p) -> kỳ vọng alpha_p = 1]` | O(p^2 n) [greedy implementation for k genes: O(k n p), so alpha_p = 1 is expected] |
| | | `O(n^2 p^2) [loại 10%/vòng: ~O(n^2 p) hoặc O(n p^2) tùy n, p]` | O(n^2 p^2) [10% removed per round: about O(n^2 p) or O(n p^2) depending on n and p] |

## 3. Values in `config.json`

| Key | Value | English |
|---|---|---|
| `note` (in `resumed_from`) | `các dòng/checkpoint không tính lại trong lần chạy này được tạo ở commit này` | the rows or checkpoints not recomputed in this run were created at this commit |
| `note` | `thời gian chỉ gồm bước chọn gen (không gồm huấn luyện bộ phân loại sau chọn)` | the time covers the selection step only (not the training of the classifier after selection) |
| `note` | `E2 = phiên tham chiếu (một phiên, mọi phương pháp); E3 = các phần đo ở phiên riêng, hiệu chuẩn với phiên tham chiếu (QĐ18); cột session ghi phần.` | E2 = the reference session (one session, every method); E3 = the parts measured in separate sessions, calibrated against the reference session (QĐ18); the column `session` names the part |
| `timing_note` | `wall/cpu = trung vị repeats lần sau một lần khởi động; không tracemalloc (v3.1.6, QĐ27)` | wall/cpu = median of `repeats` runs after one warm-up run; no tracemalloc |
| `memory_note` | `không đo bộ nhớ (chạy với --isolate-memory để có maxrss_isolated_mb)` | memory not measured (run with `--isolate-memory` to obtain `maxrss_isolated_mb`) |
| `memory_note` | `maxrss_isolated_mb = đỉnh RSS (VmHWM trên Linux, v3.1.6/QĐ27) của tiến trình con chạy một lần chọn gen trừ mức nền (bộ nhớ đỉnh dùng cho E2)` | `maxrss_isolated_mb` = peak RSS (VmHWM on Linux) of a child process that runs one selection, minus its baseline (the peak memory reported for E2) |
| `decision` | `QĐ172 (Review 12/11), spec YEU-CAU-DOI-CODE-2026-11-12-GVHD-B1-B7 §B2` and the similar values of QĐ173 to QĐ175 | specification of the analysis, decided on 12 November 2026 (Section 1) |
| `evaluation` | `evaluate_panel lr_l2, seed_for(cid, seed, subset, n_t, k) như MVP` | `evaluate_panel` with LR-L2 and the seed `seed_for(cid, seed, subset, n_t, k)`, as in the reference run of B6 (`MVP` names, in the authors' notes, a method that this study does not evaluate) |
| `lt_fold_rule` | `inner_splits lần lặp đầu, seed_for(cid, seed, subset, n_t, 9)` | `inner_splits` of the first repetition, seed `seed_for(cid, seed, subset, n_t, 9)` |
| `panel_rule` | `k gen |β| > 0 lớn nhất, hoà theo thứ tự 5.000 gen, thiếu thì F-test trên L_T; ptLasso β_eff = (1 − α̂)β1 + β2` | the k genes with the largest |β| > 0, ties broken by the order of the 5,000 genes, completed by the F-test on L_T when fewer; for ptLasso β_eff = (1 − α̂)β1 + β2 |
| `source_scaling` | `z-score theo nguồn (ddof = 0)` | z-score on the source (ddof = 0) |
| `target_values` | `log2 gốc; R z-score theo L_T (ddof = 0, sd = 0 -> 1)` | original log2 values; R z-scores on L_T (ddof = 0, sd = 0 → 1) |
| `r_session_info` | `số nhân 4`; `thời điểm 2026-11-15 04:14:35` | 4 cores; time 2026-11-15 04:14:35 |
| `layout` | `experiment-data-b7/<bộ>/ (upload as Kaggle Input)` | `experiment-data-b7/<data set>/` (upload as Kaggle Input) |
| `request` | `YEU-CAU-DOI-CODE-2026-11-09 Việc 1 (Y62 phần 1)` | request of 9 November 2026, task 1 (issue Y62, part 1) |
| `label_rule` | `... (QĐ185)`, `... (QĐ186)`, `... (Y98)` | (English, with the codes of Section 1) |
| `reason` (in `data_card_gse31210_egfr.json`) | as in Section 2 | as in Section 2 |

## 4. Logs of the Kaggle sessions (`results/logs/`)

The runners print their progress in Vietnamese. The frequent phrases are:

| Vietnamese | English |
|---|---|
| `xong: N dòng` | done: N rows |
| `ổn định N dòng`, `EN xong: N dòng ổn định` | stability: N rows; EN done: N stability rows |
| `N lần chạy (split, subset, n_t) cần làm`, `N (fraction, subsample) cần chạy` | N runs (split, subset, n_t) to do; N (fraction, subsample) to run |
| `N lần chạy × N k × N chiến lược (trừ phần đã có)` | N runs × N values of k × N strategies (except those already done) |
| `N mẫu × N gen`, `N bệnh nhân`, `train fold N = N` | N samples × N genes; N patients; size of training fold N |
| `ablation: loại [...] (không có: [])` | ablation: removed [...] (absent: []) |
| `--redo-methods [...]: đã xoá N checkpoint, sẽ tính lại` | `--redo-methods [...]`: deleted N checkpoints, will recompute |
| `đã bỏ N dòng cũ, sẽ đánh giá lại` / `sẽ chạy lại` / `sẽ chọn lại` | dropped N old rows, will evaluate again / run again / select again |
| `vN.N.N: sửa N dòng cũ lệch cột (n_draws/n_features_vs_k)` | version N.N.N: fixed N old rows with shifted columns (n_draws/n_features_vs_k) |
| `[hiệu chuẩn] ...`, `lệch so với phiên tham chiếu: {...} -> ok = True` | [calibration] ...; deviation from the reference session: {...} → ok = True |
| `[bộ nhớ] mức nền của tiến trình con: N MB (VmHWM)` | [memory] baseline of the child process: N MB (VmHWM) |
| `bỏ qua ở p = N (k-TSP: p = N > KTSP_MAX_P = N; lọc sơ bộ trước)` | skipped at p = N (k-TSP: p = N > KTSP_MAX_P = N; pre-filter first) |
| `gộp xong` | merge done |
| `SANITY ĐẠT (ngưỡng AUROC trung bình ≥ N)` | SANITY PASSED (threshold: mean AUROC ≥ N) |
| `ĐẠT`, `KHÔNG ĐẠT`, `DỪNG`, `CẢNH BÁO` | passed, failed, stop, warning |

## 5. Header of `survey_v2/panels/published_panels.csv`

The comment lines at the top of the file read, in English:

- Fixed panels used as references in E1 and E1b. Columns: `panel`, `gene`, `source`. The method name in the results is
  `panel_<panel>` (IHC-2 and IHC-4 are built in: `IHC_2`, `IHC_4` in `survey_v2/config.py`).
- The gene lists of Girard et al. (2016, Clin Cancer Res 22(19):4880, the adenocarcinoma versus squamous part) and
  Wilkerson et al. (2013, J Mol Diagn 15(4):485) are taken from the original articles (tables or supplements), with
  the DOI in the column `source`.
- Genes that are absent from a platform are recorded in the `config.json` (missing), and the panel is evaluated with
  the remaining genes (flag `complete = false`).
- Filled on 2 November 2026 from the original articles. `girard2016` = the 42 genes of the ADC-SCC signature of
  Girard et al. (2016), Fig. 1B (21 genes high in ADC and 21 high in SCC; the same list as Supplementary Table S1),
  without the 20 tumour-versus-non-malignant genes; NAPSA and SOX2 are not part of the signature (note * of Fig. 1A).
  `wilkerson2013` = the 52 marker genes of Table 1 of Wilkerson et al. (2013), the four-class HEP set, without the five
  housekeeping genes CFL1, EEF1A1, RPL10, RPL28 and RPL37A. Old symbols were mapped to HGNC: TITF1 → NKX2-1; TCF2 →
  HNF1B; TUBA1 → TUBA4A (HGNC previous symbol of TUBA4A, HGNC:12407, looked up on 3 November 2026; not checked
  against the PCR primers of the original article).
- Missing genes by platform (written to the `config.json` by the code; checked again on 2 November with
  `scripts/check_panel_coverage.py`, see `splits/qc/panel_gene_coverage.csv`): `girard2016`: GPL570 lacks ABCC6,
  KRT6C and KRT17, GPL6884 lacks PNCK, TCGA is complete; `wilkerson2013`: GPL570 lacks CLEC3B and TCP1, GPL6884 and
  TCGA are complete.

## 6. Working notes named in the code and the notebooks

Some comments and the Kaggle notebooks name working notes of the authors that are not part of the released material:
`docs/survey_v2/HUONG-DAN-DOI-CODE.md` (guide for the members who ran the code), `docs/survey_v2/KAGGLE_RUN_PLAN.md`
(schedule of the Kaggle sessions) and the experiment plan `KBTN`. The protocol that they describe is stated in the paper,
in Online Resource 1 and in `REPRODUCE.md`.
