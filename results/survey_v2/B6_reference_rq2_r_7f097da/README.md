# Tham chiếu cổng bằng nhau của B6 (QĐ176)

Trích nguyên văn từ `rq2_r_all_7f097da_results.zip` của bài 2 (kho datachain-uit/CN-23520899-LuanNT, commit 7f097da, phiên Kaggle
STAGE = all), SHA-256 của zip: `10e5e52e53022d65b1e0838764c3ea93f2c7537f7401bb419ae98a96079c62a5`. Chỉ giữ `rq2_r_per_run.csv`, `rq2_r_selected_genes.csv`, `config.json` của từng cohort và
`run_info.json`; không sửa tệp nào. Dùng duy nhất cho cổng B6: trên 5 lần chia đã công bố (seed 42–46), panel và AUROC từng lần chạy
của `scripts/run_e7_r_eval.py` phải trùng các tệp này (sai khác AUROC dưới 1e-9) — `scripts/make_revision_tables.py b6gate`.
