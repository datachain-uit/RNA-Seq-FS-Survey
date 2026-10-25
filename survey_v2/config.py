"""Hằng số đặt trước của bộ thực nghiệm survey v2 (Bảng 10–11, Review 24/10/2026; KBTN v2.1 M10–M11).

Mọi thay đổi ở đây phải được ghi lý do và duyệt trước khi chạy (nguyên tắc đặt trước của KBTN).
"""

from __future__ import annotations

# --- Bộ dữ liệu (thư mục con của $KLTN_DATA_ROOT/experiment-data-perm) -------------------
DATASETS = {
    # tên thư mục: vai trò trong survey (QĐ6)
    "gse68465_stage_early_late": "ranking",      # so sánh, kiểm định, xếp hạng; chạy cả wrapper
    "tcga_luad_stage_early_late": "descriptive",  # mô tả: filter + embedded, B0, random-k
    "tcga_lusc_stage_early_late": "descriptive",
}
RANKING_DATASET = "gse68465_stage_early_late"
TIMING_DATASET = "tcga_luad_stage_early_late"    # E2–E3: dải p rộng nhất (19.938 gen)
PERM_DATA_DIR = "experiment-data-perm"
RESULT_DIR = ("results", "survey_v2")           # $KLTN_DATA_ROOT/results/survey_v2/<E>

# --- Giao thức -------------------------------------------------------------------------
REPEAT_SEEDS = (42, 43, 44, 45, 46)             # CV 5×5; mọi phương pháp dùng chung lần chia
N_FOLDS = 5
WRAPPER_REPEATS = 3                             # wrapper (GA, BPSO) chỉ chạy 3 lần lặp đầu (3×5)
K_GRID = (10, 50, 100, 200, 500)                # ngân sách k
PREFILTER_P0 = 5000                             # lọc sơ bộ không dùng nhãn: 5.000 gen phương sai lớn nhất trên train
N_RANDOM_DRAWS = 100                            # random-k: 100 lần rút mỗi fold
INNER_CV_FOLDS = 3                              # CV nội bộ để chọn C và ngưỡng
C_GRID = (0.01, 0.1, 1.0, 10.0)                 # lưới C của LR-L2 và SVM tuyến tính
SECONDARY_K = (50, 200)                         # bộ phân loại phụ chỉ ở k ∈ {50, 200}
RF_N_TREES = 500                                # số cây của RF (bộ phân loại phụ và RF importance)

# --- Phương pháp chọn gen (10) + 2 tham chiếu ----------------------------------------------
FILTER_METHODS = ("f_test", "mutual_info", "mrmr", "relieff")
EMBEDDED_METHODS = ("l1_lr", "elastic_net", "rf_importance", "svm_rfe")
WRAPPER_METHODS = ("ga", "bpso")
REFERENCE_METHODS = ("b0", "random_k")
RANKING_METHODS = ("f_test", "mutual_info", "mrmr", "relieff", "rf_importance", "svm_rfe")  # chạy 1 lần ở k_max, lấy tiền tố
PER_K_METHODS = ("l1_lr", "elastic_net", "ga", "bpso")                                      # chạy riêng theo k
METHOD_FAMILY = {**{m: "filter" for m in FILTER_METHODS}, **{m: "embedded" for m in EMBEDDED_METHODS},
                 **{m: "wrapper" for m in WRAPPER_METHODS}, "b0": "reference", "random_k": "reference"}
STANDARDIZE_BEFORE = ("l1_lr", "elastic_net", "svm_rfe")   # z-score (fit trên train) trước khi chọn (M7)

# --- mRMR (Peng et al., 2005) ----------------------------------------------------------------
MRMR_CRITERION = "MID"           # "MID": I(X;Y) − mean I(X;Xs); "MIQ": I(X;Y) / mean I(X;Xs)
MRMR_DISCRETIZE = "mean_half_sd"  # 3 mức: < μ − σ/2, [μ − σ/2, μ + σ/2], > μ + σ/2 (tính trên train)

# --- ReliefF ------------------------------------------------------------------------------
RELIEFF_NEIGHBORS = 10

# --- Wrapper (QĐ: duyệt có chỉnh sửa M10) ----------------------------------------------------
WRAPPER_POP = 30
WRAPPER_GENERATIONS = 50
WRAPPER_FITNESS_CV = 3            # fitness = AUROC trung bình CV 3 fold của LR-L2 (C = 1) TRONG phần train của fold
WRAPPER_FITNESS_C = 1.0
GA_TOURNAMENT = 3
GA_ELITE = 2
GA_MUTATION_PER_GENE = None       # None -> 1/k (trung bình đổi 1 gen mỗi cá thể)
BPSO_W_START, BPSO_W_END = 0.9, 0.4
BPSO_C1, BPSO_C2 = 2.0, 2.0
BPSO_V_MAX = 6.0

# --- Độ ổn định (E4, E5) ------------------------------------------------------------------------
STABILITY_M = 30                             # số lần lấy mẫu con độc lập (M11)
SUBSAMPLE_FRACTIONS = (0.5, 0.7, 0.9)        # E4: tỉ lệ bệnh nhân
REDUNDANCY_RHO = (0.5, 0.8, 0.95)            # E5a: tương quan của bản sao
REDUNDANCY_N_COPIES = 50                     # E5a: số gen phương sai lớn được tạo bản sao
REDUNDANCY_FRACTION = 0.7                    # E5a: dùng cùng tỉ lệ lấy mẫu con 0,7 của E4 làm nền
SEED_VARIABILITY_SEEDS = (0, 1, 2, 3, 4)     # E5b: 5 seed cho GA, BPSO, RF
OVERLAP_K = 50                               # E5c: chồng lấn Jaccard giữa phương pháp ở k = 50

# --- Thống kê -------------------------------------------------------------------------------------
ALPHA = 0.05
ROPE_AUROC = 0.01                            # Bayesian correlated t-test, ROPE ±0,01 AUROC
NB_TEST_TRAIN_RATIO = 1.0 / 4.0              # Nadeau–Bengio: n_test / n_train với 5 fold

# --- E2–E3: chi phí ---------------------------------------------------------------------------------
TIMING_REPEATS = 3                           # trung vị 3 lần
TIMING_K = 50
E3_N_GRID = (100, 200, 300, None)            # None = toàn bộ phần train
E3_P_GRID = (1000, 2000, 5000, 10000, None)  # None = toàn bộ gen
E3_WRAPPER_BUDGETS = ((10, 20), (30, 50))    # (quần thể, thế hệ)

# --- Sanity (Y10) -------------------------------------------------------------------------------------
SANITY_DATASET = "tcga_normal_vs_tumor_sanity"
SANITY_K = 10
SANITY_AUROC_THRESHOLD = 0.99


def seed_for(*parts: int) -> int:
    """Suy seed tất định từ (cohort_id, split_seed, fold, k, draw, ...) — sửa Y14.

    Dùng np.random.default_rng([...]) trực tiếp cũng được; hàm này trả về một int ổn định
    để ghi vào config và dùng cho các thư viện chỉ nhận int (scikit-learn).
    """
    import numpy as np
    return int(np.random.SeedSequence(list(parts)).generate_state(1)[0] % (2**31 - 1))
