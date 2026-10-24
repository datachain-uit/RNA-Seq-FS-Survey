"""Build a provenance-preserving aggregate payload for Phase 0 permutation v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:  # import khi chạy test (scripts.<x>) hoặc chạy trực tiếp (python scripts/<x>.py)
    from scripts.provenance import require
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import require


SANITY_DATASET = "tcga_normal_vs_tumor_sanity"


def records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return records with missing numeric values represented as JSON null."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def build_payload(permutation_root: Path) -> dict[str, object]:
    """Read completed v3 artifacts only and separate main permutation from sanity."""
    decision_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for run_dir in sorted(path for path in permutation_root.iterdir() if path.is_dir()):
        decision_path = run_dir / "decision.csv"
        summary_path = run_dir / "summary.csv"
        if not decision_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing v3 decision/summary pair in {run_dir}")
        decision = pd.read_csv(decision_path)
        summary = pd.read_csv(summary_path)
        if not decision["complete"].astype(bool).all():
            raise ValueError(f"Incomplete v3 run: {run_dir.name}")
        decision_frames.append(decision)
        summary_frames.append(summary)

    decisions = pd.concat(decision_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    main_permutation = summaries.loc[
        (summaries["dataset"] != SANITY_DATASET) & (summaries["analysis"] == "main")
    ].copy()
    sanity = summaries.loc[
        (summaries["dataset"] == SANITY_DATASET) & (summaries["analysis"] == "main")
    ].copy()

    decisions["complete"] = decisions["complete"].astype(bool)
    return {
        "title": "Phase 0 aggregate — permutation v3",
        "permutation_root": str(permutation_root),
        "decision_columns": decisions.columns.tolist(),
        "decision_rows": records_for_json(decisions),
        "main_permutation_columns": main_permutation.columns.tolist(),
        "main_permutation_rows": records_for_json(main_permutation),
        "sanity_columns": sanity.columns.tolist(),
        "sanity_rows": records_for_json(sanity),
        "b0_status": "Chưa có baseline B0/full-gene rerun trong artifact permutation v3.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutation-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    prov = require(__file__)
    payload = build_payload(args.permutation_root)
    payload["provenance"] = prov
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(payload['decision_rows'])} decisions, "
        f"{len(payload['main_permutation_rows'])} main permutation rows, and "
        f"{len(payload['sanity_rows'])} sanity rows"
    )


if __name__ == "__main__":
    main()
