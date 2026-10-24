# Manifest notebook và script — cách ly bản cũ (Y8)

Lập ngày 24/10/2026 theo Review 20/10/2026, §7.7 (O3, Y8). Bản đầy đủ, đọc được bằng máy: [`notebook_manifest.csv`](notebook_manifest.csv).

## Tóm tắt

| Trạng thái | Số mục | Ý nghĩa |
|---|---|---|
| **ĐANG DÙNG** | 6 | Sinh ra dữ liệu hoặc kết quả hiện hành; có thể chạy lại. Khi chạy lại phải có ô provenance (mục 4). |
| **ĐÓNG BĂNG** | 48 | Kết quả đã dùng trong báo cáo; giữ nguyên tại chỗ để truy vết, **không chạy thêm** (QĐ4, O13). |
| **CŨ – CÁCH LY** | 57 | Bản thử, bản trùng lặp hoặc đã bị thay thế. Chuyển sang `_archive/`, không dùng cho kết quả mới. |

Tổng: 106 notebook (84 ở `phase_0`, 22 ở `phase_1`), 2 file text đi kèm và 1 script.

Trong kho Git (`datachain-uit/RNA-Seq-FS-Survey`): có toàn bộ 54 notebook ĐANG DÙNG và ĐÓNG BĂNG; trong số notebook CŨ chỉ có 12 notebook đã từng được commit trước đây, nay nằm ở `notebooks/_archive/`. Các notebook cũ còn lại chỉ có ở thư mục dữ liệu và không đưa lên Git.

Tiêu chí phân loại:

- **Cũ** nếu: nằm trong `legacy/`; chạy trước khi có dữ liệu chia 5 fold theo bệnh nhân (27/09/2026); là bản sao chạy thử (`- chay bth`, `- chaynhanh`) hoặc bản CPU đã có bản GPU tương đương; dùng GSE131907 (single-cell, ngoài đề tài); hoặc đã có phiên bản thay thế (perm v1/v2 → v3, notebook tải dữ liệu đã gộp vào `lung_download_label_5fold.ipynb`).
- **Đóng băng** nếu kết quả của nó đã được trích trong báo cáo 15/10, 17/10 hoặc 19/10, hoặc là đầu vào của dữ liệu đang dùng (bộ v1 cho `build_perm_test_data.py`).
- **Đang dùng** nếu là nguồn của dữ liệu v2, cohort GEO, perm v3 hoặc B0 v3.

## 1. Đang dùng

| Đường dẫn | Lý do | Sửa lần cuối |
|---|---|---|
| `notebooks/phase_0/00_data/geo_luad_lusc_target_cohorts_5fold.ipynb` | Dựng 4 cohort GEO đích. Lưu ý: nhãn GSE30219 sai (R5) → dùng scripts/rebuild_gse30219_r5.py; GSE43580 không hợp lệ (QĐ7). | 17/10/2026 |
| `notebooks/phase_0/00_data/lung_datasets_v2_patient_folds.ipynb` | Dựng bộ dữ liệu v2 (TCGA LUAD/LUSC, GSE68465 chỉ mẫu u, sanity). Nguồn của experiment-data-v2. | 12/10/2026 |
| `notebooks/phase_0/04_analysis/b0_full_gene_baseline_v3.ipynb` | Baseline B0 toàn bộ gen trên experiment-dataset-v2. | 24/10/2026 |
| `notebooks/phase_0/04_analysis/perm_test_v3_permutation.ipynb` | Perm v3 (1.000 hoán vị) — nguồn kết quả P0. | 14/10/2026 |
| `notebooks/phase_0/04_analysis/perm_test_v3_random_k.ipynb` | Perm v3 (1.000 random-k) — nguồn kết quả P0. | 14/10/2026 |
| `notebooks/phase_0/04_analysis/perm_test_v3_three_branches.ipynb` | Perm v3 ba nhánh; cùng mã nguồn với hai notebook trên. Dùng cho sanity (Y10) nếu chạy lại. | 14/10/2026 |

## 2. Đóng băng (giữ tại chỗ)

| Đường dẫn | Lý do | Sửa lần cuối |
|---|---|---|
| `notebooks/phase_0/00_data/lung_download_label_5fold.ipynb` | Dựng bộ v1 (experiment-data/h5); v1 vẫn là đầu vào của scripts/build_perm_test_data.py nên giữ lại. | 27/09/2026 |
| `notebooks/phase_0/01_feature_selection/embedded/fs_lasso.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/embedded/fs_random_forest.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/embedded/fs_ridge.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/embedded/fs_scfsnn.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/embedded/fs_svm_rfe.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/embedded/fs_wsbnn.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/extraction/fe_autoencoder.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 01/10/2026 |
| `notebooks/phase_0/01_feature_selection/extraction/fe_chnmf.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 01/10/2026 |
| `notebooks/phase_0/01_feature_selection/extraction/fe_kpca.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 08/10/2026 |
| `notebooks/phase_0/01_feature_selection/extraction/fe_pca.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 01/10/2026 |
| `notebooks/phase_0/01_feature_selection/filter/fs_f_classif.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/filter/fs_mrmr.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/filter/fs_multisurf.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/filter/fs_mutual_info.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/filter/fs_relieff.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 28/09/2026 |
| `notebooks/phase_0/01_feature_selection/wrapper/fs_bpso.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 29/09/2026 |
| `notebooks/phase_0/01_feature_selection/wrapper/fs_ga.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 29/09/2026 |
| `notebooks/phase_0/01_feature_selection/wrapper/fs_gwo_pso.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 29/09/2026 |
| `notebooks/phase_0/01_feature_selection/wrapper/fs_ibabc_cgo.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 29/09/2026 |
| `notebooks/phase_0/01_feature_selection/wrapper/fs_sga.ipynb` | Chọn đặc trưng Phase 0 (bài toán giai đoạn). Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo 15/10, 17/10. | 29/09/2026 |
| `notebooks/phase_0/02_classification/clf_gridsearch_bulkformer.ipynb` | Phân loại Phase 0. Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo. | 03/10/2026 |
| `notebooks/phase_0/02_classification/clf_gridsearch_fe.ipynb` | Phân loại Phase 0. Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo. | 01/10/2026 |
| `notebooks/phase_0/02_classification/clf_gridsearch_fs_group.ipynb` | Phân loại Phase 0. Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo. | 30/09/2026 |
| `notebooks/phase_0/02_classification/clf_gridsearch_fs_resume_500_filter.ipynb` | Phân loại Phase 0. Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo. | 03/10/2026 |
| `notebooks/phase_0/02_classification/clf_gridsearch_no_fs.ipynb` | Phân loại Phase 0. Đóng băng theo QĐ4; kết quả đã dùng trong báo cáo. | 07/10/2026 |
| `notebooks/phase_1/01_feature_selection/embedded/fs_lasso_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/embedded/fs_random_forest_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/embedded/fs_ridge_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/embedded/fs_scfsnn_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/embedded/fs_svm_rfe_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/embedded/fs_wsbnn_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/extraction/fe_autoencoder_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 15/10/2026 |
| `notebooks/phase_1/01_feature_selection/extraction/fe_chnmf_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 15/10/2026 |
| `notebooks/phase_1/01_feature_selection/extraction/fe_kpca_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 15/10/2026 |
| `notebooks/phase_1/01_feature_selection/extraction/fe_pca_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 15/10/2026 |
| `notebooks/phase_1/01_feature_selection/filter/fs_f_classif_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/filter/fs_mrmr_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/filter/fs_multisurf_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/filter/fs_mutual_info_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/filter/fs_relieff_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/wrapper/fs_bpso_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/wrapper/fs_ga_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/wrapper/fs_gwo_pso_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/wrapper/fs_ibabc_cgo_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/01_feature_selection/wrapper/fs_sga_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 11/10/2026 |
| `notebooks/phase_1/02_classification/clf_gridsearch_fe_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 18/10/2026 |
| `notebooks/phase_1/02_classification/clf_gridsearch_fs_group_phase1.ipynb` | Benchmark Phase 1 (LUAD vs LUSC, giao thức cũ một lần chia). Đóng băng theo O13: chỉ dùng mô tả, không chạy thêm. | 18/10/2026 |

## 3. Cũ – cách ly

### Cách chuyển (làm trong File Explorer)

Tạo thư mục `notebooks\_archive\phase_0\` và `scripts\_archive\`, rồi chuyển, **giữ nguyên tên**:

1. Cả thư mục `notebooks\phase_0\legacy\` (17 file) → `notebooks\_archive\phase_0\legacy\`
2. Cả thư mục `notebooks\phase_0\03_fair_comparison\` (27 notebook) → `notebooks\_archive\phase_0\03_fair_comparison\`
3. 13 file lẻ:

| Từ | Đến |
|---|---|
| `notebooks/phase_0/00_data/geo_lung_download.ipynb` | `notebooks/_archive/phase_0/00_data/geo_lung_download.ipynb` |
| `notebooks/phase_0/00_data/inspect_h5_columns.ipynb` | `notebooks/_archive/phase_0/00_data/inspect_h5_columns.ipynb` |
| `notebooks/phase_0/00_data/map_probes_to_gene_symbols.ipynb` | `notebooks/_archive/phase_0/00_data/map_probes_to_gene_symbols.ipynb` |
| `notebooks/phase_0/00_data/tcga_download.ipynb` | `notebooks/_archive/phase_0/00_data/tcga_download.ipynb` |
| `notebooks/phase_0/00_data/tcga_merge_luad_lusc.ipynb` | `notebooks/_archive/phase_0/00_data/tcga_merge_luad_lusc.ipynb` |
| `notebooks/phase_0/01_feature_selection/fs_quicktest_rf.ipynb` | `notebooks/_archive/phase_0/01_feature_selection/fs_quicktest_rf.ipynb` |
| `notebooks/phase_0/02_classification/ml-baseline-no-feature-selection.ipynb` | `notebooks/_archive/phase_0/02_classification/ml-baseline-no-feature-selection.ipynb` |
| `notebooks/phase_0/02_classification/model_tuning_5fold_kaggle.ipynb` | `notebooks/_archive/phase_0/02_classification/model_tuning_5fold_kaggle.ipynb` |
| `notebooks/phase_0/02_classification/rf_tuning_5fold_colab.ipynb` | `notebooks/_archive/phase_0/02_classification/rf_tuning_5fold_colab.ipynb` |
| `notebooks/phase_0/04_analysis/perm_test_stage.ipynb` | `notebooks/_archive/phase_0/04_analysis/perm_test_stage.ipynb` |
| `notebooks/phase_0/04_analysis/perm_test_stage_v2.ipynb` | `notebooks/_archive/phase_0/04_analysis/perm_test_stage_v2.ipynb` |
| `notebooks/phase_0/04_analysis/visualization.ipynb` | `notebooks/_archive/phase_0/04_analysis/visualization.ipynb` |
| `scripts/aggregate_5fold_results.py` | `scripts/_archive/aggregate_5fold_results.py` |

Sau khi chuyển xong, báo lại để kiểm tra: không còn file nào trong danh sách ở vị trí cũ, và `_archive/` có đủ 57 mục.

### Danh sách và lý do

| Đường dẫn | Lý do | Thay thế bởi | Sửa lần cuối |
|---|---|---|---|
| `notebooks/phase_0/00_data/geo_lung_download.ipynb` | Đã gộp vào lung_download_label_5fold.ipynb (theo mô tả của chính notebook đó). | lung_download_label_5fold.ipynb | 10/09/2026 |
| `notebooks/phase_0/00_data/inspect_h5_columns.ipynb` | Tiện ích xem cột GSE131907 (single-cell), không thuộc đề tài hiện tại. | — | 12/09/2026 |
| `notebooks/phase_0/00_data/map_probes_to_gene_symbols.ipynb` | Ánh xạ probe cho selected_features_export GSE68465 cũ (13/09). | — | 13/09/2026 |
| `notebooks/phase_0/00_data/tcga_download.ipynb` | Đã gộp vào lung_download_label_5fold.ipynb (theo mô tả của chính notebook đó). | lung_download_label_5fold.ipynb | 12/09/2026 |
| `notebooks/phase_0/00_data/tcga_merge_luad_lusc.ipynb` | Tạo tcga_expr.h5 cho fair-comparison cũ (12/09). | lung_datasets_v2_patient_folds.ipynb | 13/09/2026 |
| `notebooks/phase_0/01_feature_selection/fs_quicktest_rf.ipynb` | Chạy thử nhanh 1 fold. | — | 28/09/2026 |
| `notebooks/phase_0/02_classification/ml-baseline-no-feature-selection.ipynb` | Baseline đầu tiên (06/2026), trước pipeline 5-fold. | clf_gridsearch_no_fs.ipynb | 27/06/2026 |
| `notebooks/phase_0/02_classification/model_tuning_5fold_kaggle.ipynb` | Bản thử tinh chỉnh 8 mô hình (27/09), thay bằng các notebook clf_gridsearch_*. | clf_gridsearch_fs_group.ipynb | 28/09/2026 |
| `notebooks/phase_0/02_classification/rf_tuning_5fold_colab.ipynb` | Bản thử tinh chỉnh 8 mô hình (27/09), thay bằng các notebook clf_gridsearch_*. | clf_gridsearch_fs_group.ipynb | 28/09/2026 |
| `notebooks/phase_0/03_fair_comparison/baseline_full_features_gpu.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/bmo.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); bản CPU, có bản GPU tương đương. | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/bmo_gpu.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/bmo_gpu_gse131907.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); chạy trên GSE131907 (single-cell), không thuộc đề tài hiện tại. | notebooks/phase_0/01_feature_selection/wrapper/* | 12/09/2026 |
| `notebooks/phase_0/03_fair_comparison/bmo_gpu_tcga.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 13/09/2026 |
| `notebooks/phase_0/03_fair_comparison/bpso.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); bản CPU, có bản GPU tương đương. | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/bpso_gpu - chay bth.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); bản sao chạy thử ('- chay bth' / '- chaynhanh') có output. | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/bpso_gpu_gse131907.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); chạy trên GSE131907 (single-cell), không thuộc đề tài hiện tại. | notebooks/phase_0/01_feature_selection/wrapper/* | 12/09/2026 |
| `notebooks/phase_0/03_fair_comparison/bpso_gpu_tcga.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 13/09/2026 |
| `notebooks/phase_0/03_fair_comparison/deqlspark.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); bản CPU, có bản GPU tương đương. | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/deqlspark_gpu - chaynhanh.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); bản sao chạy thử ('- chay bth' / '- chaynhanh') có output. | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/deqlspark_gpu.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/deqlspark_gpu_threaded.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/deqlspark_gpu_threaded_gse131907.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); chạy trên GSE131907 (single-cell), không thuộc đề tài hiện tại. | notebooks/phase_0/01_feature_selection/wrapper/* | 12/09/2026 |
| `notebooks/phase_0/03_fair_comparison/deqlspark_gpu_threaded_tcga.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 13/09/2026 |
| `notebooks/phase_0/03_fair_comparison/gwopso.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); bản CPU, có bản GPU tương đương. | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/gwopso_gpu.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/gwopso_gpu_gse131907.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); chạy trên GSE131907 (single-cell), không thuộc đề tài hiện tại. | notebooks/phase_0/01_feature_selection/wrapper/* | 12/09/2026 |
| `notebooks/phase_0/03_fair_comparison/gwopso_gpu_tcga.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 13/09/2026 |
| `notebooks/phase_0/03_fair_comparison/ibabccgo.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); bản CPU, có bản GPU tương đương. | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/ibabccgo_gpu - chay bth.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); bản sao chạy thử ('- chay bth' / '- chaynhanh') có output. | notebooks/phase_0/01_feature_selection/wrapper/* | 11/09/2026 |
| `notebooks/phase_0/03_fair_comparison/ibabccgo_gpu_gse131907.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); chạy trên GSE131907 (single-cell), không thuộc đề tài hiện tại. | notebooks/phase_0/01_feature_selection/wrapper/* | 12/09/2026 |
| `notebooks/phase_0/03_fair_comparison/ibabccgo_gpu_tcga.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 13/09/2026 |
| `notebooks/phase_0/03_fair_comparison/sga.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); bản CPU, có bản GPU tương đương. | notebooks/phase_0/01_feature_selection/wrapper/* | 13/09/2026 |
| `notebooks/phase_0/03_fair_comparison/sga_gpu.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 13/09/2026 |
| `notebooks/phase_0/03_fair_comparison/sga_gpu_gse131907.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09); chạy trên GSE131907 (single-cell), không thuộc đề tài hiện tại. | notebooks/phase_0/01_feature_selection/wrapper/* | 13/09/2026 |
| `notebooks/phase_0/03_fair_comparison/sga_gpu_tcga.ipynb` | Fair-comparison cũ (sửa 11–16/09/2026), chạy trước khi có dữ liệu 5-fold theo bệnh nhân (27/09). | notebooks/phase_0/01_feature_selection/wrapper/* | 13/09/2026 |
| `notebooks/phase_0/04_analysis/perm_test_stage.ipynb` | Perm v1, thay bằng v3. | perm_test_v3_*.ipynb | 07/10/2026 |
| `notebooks/phase_0/04_analysis/perm_test_stage_v2.ipynb` | Perm v2 (bỏ trọng số lớp, O11), thay bằng v3. Kịch bản v2 vẫn ở docs/. | perm_test_v3_*.ipynb | 08/10/2026 |
| `notebooks/phase_0/04_analysis/visualization.ipynb` | Vẽ quần thể wrapper từ population_state_history.csv trên Colab (29/09); ô lặp lại, 3,8 MB output. | — | 29/09/2026 |
| `notebooks/phase_0/legacy/_baseline_full.txt` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 13/09/2026 |
| `notebooks/phase_0/legacy/_baseline_full.txt` | Output văn bản của notebook legacy. | — | 13/09/2026 |
| `notebooks/phase_0/legacy/bmo-svm-v1_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 27/06/2026 |
| `notebooks/phase_0/legacy/cell_out.txt` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 14/09/2026 |
| `notebooks/phase_0/legacy/cell_out.txt` | Output văn bản của notebook legacy. | — | 14/09/2026 |
| `notebooks/phase_0/legacy/classification-comparison_gse131907_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 14/09/2026 |
| `notebooks/phase_0/legacy/classification-comparison_gse68465_gridsearch_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 14/09/2026 |
| `notebooks/phase_0/legacy/classification-comparison_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 13/09/2026 |
| `notebooks/phase_0/legacy/de-qlearning-spark_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 27/06/2026 |
| `notebooks/phase_0/legacy/de-qlearning_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 27/06/2026 |
| `notebooks/phase_0/legacy/deql-diversity-no-spark_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 28/06/2026 |
| `notebooks/phase_0/legacy/deql-spark-diversity_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 28/06/2026 |
| `notebooks/phase_0/legacy/deqlspark-models-val-test_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 13/09/2026 |
| `notebooks/phase_0/legacy/gwo-pso-knn_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 27/06/2026 |
| `notebooks/phase_0/legacy/ibabc-cgo_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 27/06/2026 |
| `notebooks/phase_0/legacy/pso-dt_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 27/06/2026 |
| `notebooks/phase_0/legacy/sga-rf-h100_with_metrics.ipynb` | Thư mục legacy: notebook đầu tiên (06–09/2026) trước khi có dữ liệu chia fold theo bệnh nhân. | notebooks/phase_0/01_feature_selection, 02_classification | 27/06/2026 |
| `scripts/aggregate_5fold_results.py` | Script tổng hợp Phase 0 cũ: đọc D:\KLTN\clf-experiment-result (không còn tồn tại) và ghi vào scripts/aggregated_results/. Không thêm provenance. | scripts/aggregate_phase0_v3.py, scripts/aggregate_phase1_results.py | 03/10/2026 |

## 4. Provenance cho mọi lần chạy mới

Mọi script thực nghiệm gọi `scripts/provenance.py` ở đầu `main()`:

- **Dừng** nếu thư mục không phải git clone, hoặc file đã được Git theo dõi còn thay đổi chưa commit.
- Ghi vào `config.json` (hoặc `provenance.json`) cạnh kết quả: commit hash, nhánh, remote, script và tham số, thời điểm, phiên bản Python và thư viện, máy, SHA-256 của dữ liệu đầu vào.
- **Ngoại lệ tạm khi chưa có Git (task 3.1):** đặt `KLTN_ALLOW_UNTRACKED=1`. Script chạy tiếp nhưng ghi `commit_hash: null`, `tracked: false` và lý do. Kết quả loại này không được trích vào survey hay luận văn cho tới khi chạy lại trong git clone.

```powershell
# PowerShell, chạy tạm khi chưa có Git
$env:KLTN_ALLOW_UNTRACKED = "1"; python scripts/run_rq0_prime.py
```

| Script | Provenance ghi ở đâu |
|---|---|
| `run_rq0_prime.py` | `results/phase 1/rq0_prime/config.json` |
| `rebuild_gse30219_r5.py` | `experiment-data-geo-target/gse30219_r5_fixed/config.json` |
| `build_perm_test_data.py` | `experiment-data-perm/manifest.json`; `commit_hash` trong config từng bộ |
| `patch_v2_tcga_stage.py` | `config.json` của 3 bộ v2 |
| `aggregate_phase0_v3.py`, `aggregate_phase1_results.py` | khoá `provenance` trong file JSON đầu ra |
| `stage_split_metrics.py` | `results/stage_split/provenance.json` (kèm SHA-256 đầu ra) |
| `organize_fs_phase1.py` | `results/phase 1/fs/organize_provenance.json` |

`--commit <hash>` nhập tay ở `build_perm_test_data.py` và `patch_v2_tcga_stage.py` đã bỏ, vì không kiểm chứng được hash đó có khớp mã đang chạy hay không.

Không thêm provenance cho các script dựng tài liệu (`build_review_*.py`, `build_rq0_prime_workbook.py`, `*.mjs`) vì chúng không sinh kết quả thực nghiệm.

### Ô provenance cho notebook (khi chạy lại một notebook ĐANG DÙNG)

Notebook trên Kaggle/Colab không thấy thư mục máy, nên phải clone kho ở đúng commit rồi dùng cùng module. Dán ô này **đầu tiên**, thay cho các biến `GIT_COMMIT = ""` hiện có:

```python
# === PROVENANCE (Review 20/10/2026, §7.7) ===
import os, subprocess, sys
REPO_URL = "https://github.com/datachain-uit/RNA-Seq-FS-Survey.git"
COMMIT = "<hash>"                                  # commit sẽ chạy
REPO_DIR = "/kaggle/working/RNA-Seq-FS-Survey"     # Colab: /content/RNA-Seq-FS-Survey
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", "-q", COMMIT], check=True)
sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))
from provenance import provenance, stamp, save_provenance
PROV = provenance(script="notebooks/<đường dẫn notebook>.ipynb", repo_dir=REPO_DIR)
GIT_COMMIT = PROV["commit_hash"]                   # các ô sau dùng biến này như cũ
# Cuối notebook: stamp(config, PROV) trước khi ghi config.json, hoặc save_provenance(RESULT_DIR, PROV)
```

Notebook đóng băng không cần sửa vì không chạy lại.
