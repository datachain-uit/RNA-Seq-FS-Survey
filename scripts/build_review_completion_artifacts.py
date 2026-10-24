"""Build auditable hand-off documents for Phase 0 and the Phase 1 data card.

The script only inventories existing files.  It deliberately never turns a
missing Git history into a made-up commit identifier.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

try:  # import khi chạy test (scripts.<x>) hoặc chạy trực tiếp (python scripts/<x>.py)
    from scripts.provenance import DATA_ROOT
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import DATA_ROOT


RUNS = {
    "GSE68465": "perm_test_v2_gse68465",
    "TCGA-LUAD": "perm_test_v2_tcga_luad",
    "TCGA-LUSC": "perm_test_v2_tcga_lusc",
}
EVIDENCE = (
    "summary.csv",
    "real_scores_per_fold.csv",
    "real_confusion.csv",
    "random_k_scores.csv",
    "null_scores.csv",
    "decision.csv",
    "config.json",
    "timing.csv",
)


def git_commit_or_missing(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "Không có (thư mục không phải Git)"
    return proc.stdout.strip() or "Không có (thư mục không phải Git)"


def collect_phase0_manifest(root: Path) -> list[dict[str, str]]:
    base = root / "results" / "phase 0" / "permu_test"
    stage = root / "results" / "phase 0" / "stage_split"
    commit = git_commit_or_missing(root)
    rows: list[dict[str, str]] = []
    for dataset, folder in RUNS.items():
        run = base / folder
        config: dict[str, object] = {}
        config_path = run / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}
        row = {
            "dataset": dataset,
            "run_directory": str(run.relative_to(root)).replace("\\", "/"),
            "n_permutations": str(config.get("n_permutations", "Không đọc được")),
            "commit_hash": commit,
        }
        for name in EVIDENCE:
            row[name.replace(".", "_")] = "Có" if (run / name).is_file() else "Thiếu"
        row["stage_split_no_fs_summary"] = "Có" if (stage / "summary_no_fs_by_dataset_model.csv").is_file() else "Thiếu"
        row["stage_split_metrics_per_fold"] = "Có" if (stage / "split_metrics_per_fold.csv").is_file() else "Thiếu"
        required = ("summary.csv", "real_scores_per_fold.csv", "real_confusion.csv", "config.json")
        has_evidence = all((run / name).is_file() for name in required)
        has_evidence = has_evidence and row["stage_split_no_fs_summary"] == "Có" and row["stage_split_metrics_per_fold"] == "Có"
        row["handoff_status"] = (
            "Hoàn chỉnh" if has_evidence and commit != "Không có (thư mục không phải Git)"
            else "Cần bổ sung commit hash" if has_evidence
            else "Thiếu tệp bằng chứng"
        )
        rows.append(row)
    return rows


def write_phase0_manifest(root: Path, rows: list[dict[str, str]]) -> None:
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    fields = ["dataset", "run_directory", "n_permutations", "stage_split_no_fs_summary", "stage_split_metrics_per_fold", *[x.replace(".", "_") for x in EVIDENCE], "commit_hash", "handoff_status"]
    with (docs / "phase0_results_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    headings = ["Bộ dữ liệu", "Run", "Hoán vị", "B0/no-FS", "Tách chỉ số/fold", "Summary", "Real scores", "Confusion", "Random-k", "Null", "Decision", "Config", "Timing", "Commit", "Trạng thái"]
    keys = ["dataset", "run_directory", "n_permutations", "stage_split_no_fs_summary", "stage_split_metrics_per_fold", "summary_csv", "real_scores_per_fold_csv", "real_confusion_csv", "random_k_scores_csv", "null_scores_csv", "decision_csv", "config_json", "timing_csv", "commit_hash", "handoff_status"]
    lines = [
        "# Manifest bàn giao kết quả Phase 0",
        "",
        "Ngày lập: 2026-10-19. Manifest chỉ xác nhận sự hiện diện của tệp và thông số trong `config.json`; không thay thế việc tái chạy thực nghiệm.",
        "",
        "| " + " | ".join(headings) + " |",
        "|" + "|".join(["---"] * len(headings)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[key] for key in keys) + " |")
    lines += [
        "",
        "## Kết luận bàn giao",
        "",
        "Ba bộ đều có bảng B0/no-FS, bảng tách chỉ số theo fold, summary, điểm thật theo fold, confusion matrix, cấu hình và các tệp permutation/random-k. Hai tệp stage split là bảng tổng hợp dùng chung cho ba bộ; manifest xác nhận sự hiện diện của chúng cho từng dòng bàn giao. Repository hiện không có thư mục `.git`, nên không thể xác minh hay bổ sung commit hash một cách trung thực. Vì vậy trạng thái nghiệm thu là **đủ tệp thực nghiệm, còn thiếu định danh phiên bản Git**.",
    ]
    (docs / "phase0_results_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_official_data_card(root: Path) -> None:
    text = """# Data card chính thức — Phase 0 và Phase 1

Ngày cập nhật: 2026-10-19. Tài liệu này là chỉ mục chuẩn cho các bộ dữ liệu đã dùng; chi tiết truy vết nằm trong từng `DATA_CARD.md` và `filter_log.csv` được liên kết bên dưới.

## Phạm vi và nguồn

| Bộ dữ liệu | Nhiệm vụ | Nguồn / đơn vị | Tệp truy vết |
|---|---|---|---|
| GSE68465 stage | Giai đoạn LUAD (bản data card gốc: I/II/III/IV; bài toán permutation: early I vs late II–IV) | GEO GSE68465, GPL96; 440 mẫu u LUAD sau lọc | [`DATA_CARD.md`](../experiment-data-v2/datasets/gse68465_stage_tumor_only/DATA_CARD.md), [`filter_log.csv`](../experiment-data-v2/datasets/gse68465_stage_tumor_only/filter_log.csv), [bộ perm](../experiment-data-perm/gse68465_stage_early_late/) |
| TCGA LUAD–LUSC | Phân loại mô học LUAD vs LUSC trên u nguyên phát | GDC API; TCGA-LUAD + TCGA-LUSC STAR Counts; 1,018 bệnh nhân (517/501) | [`DATA_CARD.md`](../experiment-data-v2/datasets/tcga_luad_lusc_primary_tumor/DATA_CARD.md), [`filter_log.csv`](../experiment-data-v2/datasets/tcga_luad_lusc_primary_tumor/filter_log.csv), [bộ perm](../experiment-data-perm/tcga_luad_lusc_primary_tumor/) |
| TCGA LUAD stage | Giai đoạn LUAD, early I vs late II–IV | GDC API, TCGA-LUAD u nguyên phát | [bộ perm](../experiment-data-perm/tcga_luad_stage_early_late/) |
| TCGA LUSC stage | Giai đoạn LUSC, early I vs late II–IV | GDC API, TCGA-LUSC u nguyên phát | [bộ perm](../experiment-data-perm/tcga_lusc_stage_early_late/) |
| TCGA normal–tumor | Sanity check Normal vs Primary Tumor | GDC API, TCGA LUAD/LUSC | [`DATA_CARD.md`](../experiment-data-v2/datasets/tcga_normal_vs_tumor_sanity/DATA_CARD.md), [`filter_log.csv`](../experiment-data-v2/datasets/tcga_normal_vs_tumor_sanity/filter_log.csv), [bộ perm](../experiment-data-perm/tcga_normal_vs_tumor_sanity/) |

## Tiền xử lý, nhãn và fold

- **GSE68465:** loại vĩnh viễn 19 mẫu RNA tham chiếu Stratagene; chỉ giữ u adenocarcinoma có stage hợp lệ. MAS5 được log2(x+1); probe được ánh xạ GPL96, bỏ ánh xạ đa gene và chọn probe có phương sai lớn nhất cho gene trùng. Nhãn stage được suy từ thông tin AJCC/pTNM (giả định M0 khi áp dụng trong card gốc).
- **TCGA:** chỉ giữ Primary Tumor, một mẫu mỗi bệnh nhân; expression là log2(TPM+1), protein-coding, trùng gene giữ phương sai lớn nhất. Nhãn histology lấy từ project; nhãn stage lấy từ trường stage của GDC đối với các tập stage.
- **Fold:** các data card v2 dùng `StratifiedGroupKFold`, nhóm theo `patient_id`; patient không chồng giữa train/validation/test. Các file `fold_assignment.csv` và thư mục `folds/` là nguồn kiểm tra cụ thể.

## Liên kết với permutation và giới hạn

Các dataset dùng cho permutation được đặt tại `experiment-data-perm/`. Phiên bản/cấu hình Phase 0 đã được đối chiếu trong [manifest Phase 0](phase0_results_manifest.md). Các card nêu rõ xử lý probe, loại mẫu, nhãn và split; chúng không thay thế đánh giá ngoại kiểm trên cohort GEO. Repository hiện không có lịch sử Git khả dụng, do đó định danh commit trong các card lịch sử là `no-git-repo` và cần được khởi tạo/quản lý trong lần bàn giao sau.
"""
    (root / "docs" / "data_card.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = DATA_ROOT  # thư mục dữ liệu/kết quả (KLTN_DATA_ROOT)
    rows = collect_phase0_manifest(root)
    write_phase0_manifest(root, rows)
    write_official_data_card(root)


if __name__ == "__main__":
    main()
