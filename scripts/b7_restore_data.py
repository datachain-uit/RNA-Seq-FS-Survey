"""B7 (QĐ177): đặt hai bộ EGFR đã dựng ở bước 1 vào ``$KLTN_DATA_ROOT/experiment-data-b7/`` cho các phiên E1, E4 của B7.

Nguồn: ``results/survey_v2/B7_data/experiment-data-b7/<bộ>/`` (có trong zip ``b7_step1_results.zip`` khi khôi phục làm Input).
Mọi tệp được kiểm SHA-256 với ``B7_data/config.json`` (files_sha256) trước khi chép; lệch thì dừng. Đích đã có thì chỉ kiểm.

    python scripts/b7_restore_data.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import sha256  # noqa: E402


def restore(data_root: Path) -> dict:
    src_root = data_root.joinpath(*C.RESULT_DIR, "B7_data")
    cfg_path = src_root / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"thiếu {cfg_path}: khôi phục output của b7_step1 làm Input (RESTORE_INPUTS)")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("traceable") is not True:
        raise SystemExit("B7_data/config.json không traceable")
    expected = cfg["files_sha256"]
    report = {"source_commit": cfg.get("commit_hash"), "files": 0}
    for rel, digest in sorted(expected.items()):
        src = src_root / rel
        dst = data_root / rel
        if sha256(src) != digest:
            raise SystemExit(f"SHA-256 lệch ở {src}")
        if dst.exists():
            if sha256(dst) != digest:
                raise SystemExit(f"{dst} đã có nhưng khác bản B7_data — không ghi đè")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        report["files"] += 1
    for name in C.B7_DATASETS:
        if not (data_root / C.B7_DATA_DIR / name / "expression.h5").exists():
            raise SystemExit(f"thiếu {name} sau khi khôi phục")
    print(f"B7 data: {report['files']} tệp kiểm SHA-256, từ commit {str(report['source_commit'])[:7]} → {data_root / C.B7_DATA_DIR}", flush=True)
    return report


def main(argv=None) -> int:
    from survey_v2.runlog import DATA_ROOT
    restore(Path(DATA_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
