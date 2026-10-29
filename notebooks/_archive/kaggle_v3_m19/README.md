# Notebook Kaggle v3 chạy 26–28/10/2026 — LƯU TRỮ, KHÔNG DÙNG ĐỂ CHẠY CHÍNH THỨC

Lưu trữ ngày 29/10/2026 theo Review 29/10 (M18, M19, QĐ18, QĐ19). Các notebook này do đội code viết trước khi áp gói v3.1
(commit `6776d71`); chúng chứa logic thí nghiệm ngay trong notebook và gán `traceable: True` bằng tay.

- **M18**: logic thí nghiệm chỉ nằm trong script đã commit; notebook chỉ gọi script. Chạy chính thức bằng các script trong `scripts/`
  (`run_survey_e0_signal_check.py`, `run_e1_feature_selection.py`, `run_e1_classification.py`, `run_survey_e1b_learning_curve.py`,
  `run_survey_e2_e3_timing.py`, `run_survey_e4_e5_stability.py`, ...) qua runner `kaggle_e1_feature_selection_v3_runner.ipynb` /
  `kaggle_e1_separated_5x5_runner.ipynb`.
- **M19**: kết quả do các notebook này tạo ra chỉ dùng để **ước lượng chi phí**, không trích vào bài.

Lệch so với thiết kế v3.1 (để không vô tình dùng lại):

| Notebook | Lệch chính |
|---|---|
| `kaggle_e0_v3_runner` | CV 5×5 (script dùng CV 5 fold), tên tệp đầu ra khác, ghi `results/survey_v3` |
| `kaggle_e1_classification_v3_runner` | chỉ LR-L2; thiếu log-loss/Brier, RF/SVM ở k ∈ {5, 25}; wrapper 5 cohort × 25 fold × 8 k (QĐ19 không dùng); lỗi với ktsp k lẻ |
| `kaggle_e1_reference_v3_runner` | B0 / random-k tách khỏi pha B |
| `kaggle_e1b_fewshot_v3_runner` | 1 tập con mỗi n_T (O23); gom theo (split, n_T) sẽ trộn 4 tập con của bản chia v3.1; không áp M5 |
| `kaggle_e2_e3_v3_runner`, `..._sharded_runner` | không hiệu chuẩn (QĐ18), không `--isolate-memory` (Y19); E3 n = 300 thay vì 400 |
| `kaggle_e4_stability_v3_runner` | 5 cohort (đúng: TCGA, GSE37745) |
| `kaggle_e5a_redundancy_v3_runner` | 5 cohort (đúng: GSE37745) |
| `kaggle_e5b_seed_sensitivity_v3_runner` | TCGA, wrapper k = 50 (đúng: GSE37745; rf_importance + stability_selection mọi k; GA/BPSO k = 10) |
| `kaggle_e5c_overlap_v3_analysis` | k = 50 (đúng: `OVERLAP_K` = 10) |

`tests/test_kaggle_e1_classification_notebooks.py` chỉ kiểm tra cấu trúc các notebook này (đã trỏ sang thư mục lưu trữ).
