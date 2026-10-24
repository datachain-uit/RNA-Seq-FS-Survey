# RNA-Seq FS Survey – chọn gen có ràng buộc ngân sách cho dữ liệu biểu hiện gen ung thư phổi

Mã nguồn khóa luận: chọn/trích xuất đặc trưng, phân loại, permutation test (bài toán phụ: giai đoạn sớm/muộn) và RQ0′–RQ4 (bài toán chính: LUAD vs LUSC, chuyển giao TCGA → cohort GEO).

**Kho này là nơi duy nhất chứa code.** Dữ liệu và kết quả nằm ở thư mục dữ liệu riêng (vd. `D:\KLTN`), không đưa lên Git. Mọi kết quả phải sinh từ một commit đã push, để commit hash ghi trong `config.json` kiểm chứng được trên GitHub (Review 20/10/2026, O3, §7.7).

## Cấu trúc

```
RNA-Seq-FS-Survey/
├── scripts/                     code chạy trên máy (Python)
│   ├── provenance.py            ghi nguồn gốc mỗi lần chạy; dừng nếu không truy vết được
│   ├── run_rq0_prime.py         RQ0′: trần hiệu năng trên 4 cohort GEO (CV 5×5)
│   ├── build_rq0_prime_workbook.py
│   ├── rebuild_gse30219_r5.py   dựng lại GSE30219 sau khi sửa nhãn (R5): 85 LUAD / 61 LUSC
│   ├── build_perm_test_data.py  dữ liệu cho permutation test v3
│   ├── patch_v2_tcga_stage.py   vá nhãn giai đoạn TCGA v2 từ GDC
│   ├── aggregate_phase0_v3.py, aggregate_phase1_results.py, stage_split_metrics.py, organize_fs_phase1.py
│   ├── build_review_completion_artifacts.py, build_review_documents.py
│   ├── PHASE1_REPRODUCTION.md
│   └── _archive/                script cũ, không dùng
├── tests/                       python -m pytest tests
├── notebooks/
│   ├── phase_0/                 dữ liệu v1/v2/GEO, FS/FE và phân loại Phase 0, permutation v3, B0 v3
│   ├── phase_1/                 benchmark Phase 1 (LUAD vs LUSC) – ĐÓNG BĂNG (O13)
│   └── _archive/                notebook cũ, không dùng
└── docs/notebook_manifest.md    trạng thái từng notebook: đang dùng / đóng băng / cũ
```

## Cài đặt

```powershell
git clone https://github.com/datachain-uit/RNA-Seq-FS-Survey.git
cd RNA-Seq-FS-Survey
pip install -r requirements.txt
$env:KLTN_DATA_ROOT = "D:\KLTN"     # thư mục chứa experiment-data*/, results/ ...
```

`KLTN_DATA_ROOT` cho script biết dữ liệu ở đâu. Thiếu biến này, script tìm dữ liệu ngay trong thư mục kho.

## Quy trình mỗi lần chạy

1. Sửa code trong kho → `git commit` → `git push`.
2. Chạy script từ thư mục kho, ví dụ `python scripts/run_rq0_prime.py`.
3. Script gọi `provenance()` trước khi tính toán:
   - **dừng** nếu thư mục không phải git clone hoặc còn file đã theo dõi bị sửa mà chưa commit;
   - ghi vào `config.json`/`provenance.json` cạnh kết quả: commit hash, script và tham số, phiên bản thư viện, máy, SHA-256 của dữ liệu đầu vào.

Chỉ trong trường hợp bất khả kháng mới chạy với `$env:KLTN_ALLOW_UNTRACKED = "1"`; kết quả khi đó ghi `commit_hash: null` và không được trích vào survey hay luận văn.

Notebook chạy trên Kaggle/Colab: clone kho ở đúng commit rồi dùng `scripts/provenance.py` (ô mẫu ở `docs/notebook_manifest.md`, mục 4).

## Dữ liệu (ở `KLTN_DATA_ROOT`, không có trong kho)

| Thư mục | Nội dung |
|---|---|
| `experiment-data-v2/` | TCGA LUAD/LUSC (1.018 bệnh nhân), GSE68465 chỉ mẫu u, sanity Normal vs u; có `DATA_CARD.md` |
| `experiment-data-geo-target/` | cohort đích GSE10245, GSE30219 (`gse30219_r5_fixed`), GSE37745, GSE42127 |
| `experiment-data-perm/` | dữ liệu permutation test v3 |
| `results/` | kết quả theo phase |
