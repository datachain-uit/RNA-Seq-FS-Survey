# Chạy survey v3 trên Kaggle — kế hoạch phiên (sau QĐ21 + QĐ22, 29/10/2026)

Một notebook duy nhất: `notebooks/kaggle_survey_v3_runner.ipynb`. Mỗi phiên Kaggle (Save Version → Run All) chạy **một việc**
(`JOB`) trong `scripts/kaggle_jobs.py`; lệnh, chia shard, đầu vào cần khôi phục và kiểm tra đầu ra nằm trong script đã commit (M18).

```bash
python scripts/kaggle_jobs.py plan          # bảng việc: đợt, hạn, ước lượng giờ, việc cần khôi phục
python scripts/kaggle_jobs.py show e1_phaseA_geo
python scripts/kaggle_jobs.py status        # việc nào đã có đầu ra trong $KLTN_DATA_ROOT/results/survey_v2
```

## 0. Chuẩn bị (một lần)

1. **Kaggle Dataset `experiment-data-v3`** = bản 29/10 sau QC (5 bộ: `expression.h5`, `metadata.csv`, `repeated_splits.csv`,
   `fewshot_splits.csv` (GEO), `excluded_samples_<bộ>.csv`, `DATA_CARD.md`, `rebuild_log.json`; TCGA thêm `sex_gdc*.*`;
   `tcga_normal_vs_tumor_sanity/`; `split_manifests/`; `qc/`). Không đưa `_backup_*` và `experiment-data-v3.zip` cũ.
   Gate 1 của notebook dừng nếu Input là bản cũ.
2. Commit + push code; ghi hash vào `REPO_COMMIT` của notebook và **giữ nguyên cho mọi phiên của một đợt** (cùng commit → gộp được).
3. Tạo notebook trên Kaggle từ `notebooks/kaggle_survey_v3_runner.ipynb` (File → Import Notebook), Accelerator: None (CPU), Internet: On.

## 1. Một phiên

1. Ô cấu hình: `JOB = '<tên việc>'`, `RESTORE_INPUTS = [...]` (nếu `plan` ghi cần khôi phục), `REPO_COMMIT = '<hash>'`.
2. Add Input: `experiment-data-v3` + output của các việc cần khôi phục.
3. Save Version → Save & Run All. Kết quả: `/kaggle/working/<JOB>_results.zip` (toàn bộ `results/` gồm phần đã khôi phục + phần mới,
   `logs/`, `run_info_<JOB>.json`).
4. Lưu output thành Dataset để dùng lại: trang phiên → Output → **New Dataset** (đặt tên `kltn-<job>`), hoặc Add Input → Notebook
   Output Files của đúng version. Trong `RESTORE_INPUTS` ghi thư mục Input đó, vd. `[Path('/kaggle/input/kltn-e1-phasea-geo')]`.

Vì mỗi zip chứa **toàn bộ** `results/` đã có, ở một chuỗi việc chỉ cần khôi phục zip mới nhất; các phiên chạy song song thì khôi phục
tất cả zip của chúng (các script không ghi đè checkpoint của nhau).

**Quá 12 giờ:** notebook tự dừng có kiểm soát sau `MAX_HOURS = 11` (mã 124), vẫn lưu zip. Mở phiên mới **cùng JOB**, khôi phục zip
vừa lưu: mọi script tự tiếp tục từ checkpoint.

**E2/E3 (đo thời gian):** chạy một mình trong phiên (script đặt BLAS 1 luồng); các phiên E3 phải cùng loại máy Kaggle (CPU 4 nhân)
với phiên tham chiếu `e2_ref`; phiên lệch ≥ 10 % sẽ dừng (QĐ18) — chạy lại phiên đó.

## 2. Thứ tự phiên

| Đợt / hạn | Phiên (JOB) | Khôi phục | Ước lượng | Ghi chú |
|---|---|---|---|---|
| 1 · 03/11 | `sanity` | — | ~1 h | AUROC ≥ 0,99 (có thể chạy trên máy) |
| 1 · 03/11 | `e0` | — | ~1 h | `signal_present = True` ở 5 bộ; không đạt → báo GVHD, không sửa ngưỡng |
| 1 · 03/11 | `e1_phaseA_geo` | — | 3–6 h | 4 tiến trình (1/cohort) |
| 1 · 03/11 | `e2_ref` | — | 2–4 h | **một mình**; tạo `E2_E3_ref/calibration.csv` |
| 1 · 03/11 | `e1_phaseB_geo` | `e1_phaseA_geo` | 1–2 h | |
| 1T · 04/11 | `e1_phaseA_tcga_r012` | — | 5–8 h | song song với phiên dưới |
| 1T · 04/11 | `e1_phaseA_tcga_r34` | — | 5–8 h | |
| 1T · 04/11 | `e1_phaseB_tcga` | 2 phiên trên | 3–6 h | gộp pha A TCGA rồi pha B |
| 2 · 06/11 | `fewshot_e1b` | — | 3–5 h | E1b |
| 2 · 06/11 | `e3_p1`, `e3_p2`, `e3_p3` | `e2_ref` | 3–8 h mỗi phiên | **mỗi phiên một mình**; cùng loại máy |
| 2 · 06/11 | `e2e3_merge` | 3 phiên E3 | phút | gộp → `E2_E3` (có thể chạy trên máy) |
| 2 · 06/11 | `e4e5_gse37745` | `e1_phaseA_geo` | 4–8 h | E4 + E5a + E5b + E5c |
| 2 · 06/11 | `e4_tcga` | `e1_phaseB_tcga` | 6–11 h | E4 + E5c TCGA |
| 3 · 08/11 | `wrap_gse10245`, `wrap_gse30219`, `wrap_gse37745` | — | 3–6 h mỗi phiên | 3 tiến trình (1/lần lặp) |
| 3 · 08/11 | `e1_phaseB_wrappers` | `e1_phaseB_geo` + 3 phiên wrap | ~1 h | nối GA/BPSO vào `E1_classification` |
| 3 · 08/11 | `e4e5_gse37745` (lần 2) | lần 1 + `wrap_gse37745` | phút | chỉ bổ sung E5c có GA/BPSO |
| 3 · 08/11 | `e7` | — | 2–4 h | sau E1b (cùng bản chia few-shot) |
| sau mỗi đợt | `tables` | mọi zip | < 0,5 h | hoặc tải các zip về máy rồi `python scripts/make_survey_tables.py` |
| 4 · 10/11 | `ablation_phaseA_geo`, `ablation_phaseA_tcga`, `ablation_phaseB` | | | loại 4 gen IHC (khoá luận, ưu tiên 2) |

Đợt 1 nộp 03/11: E0, E1 bốn cohort GEO (pha A + B), E2 tham chiếu; phần TCGA của E1 chậm nhất 04/11 (QĐ22b). Không kịp phần GEO
→ báo GVHD trước 02/11.

## 3. Nghiệm thu mỗi phiên

- Notebook in `JOB_OK <job>`: mọi `config.json` đầu ra có `traceable = true`, `commit_hash = HEAD`, và (E2/E3) `calibration_ok` không `false`.
- `run_info_<JOB>.json` trong zip: commit, mã thoát, giờ chạy, zip đã khôi phục — chép vào HANDOVER.
- Chỉ kết quả từ phiên có `JOB_OK` mới được trích (kết quả 26–28/10 chỉ để ước lượng chi phí, M19).
