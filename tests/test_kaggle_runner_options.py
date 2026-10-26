import json
from pathlib import Path


def test_kaggle_feature_selection_runner_exposes_three_method_groups():
    notebook = Path("notebooks/kaggle_e1_feature_selection_v3_runner.ipynb")
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

    for option in ("RUN_FILTER", "RUN_EMBEDDED", "RUN_WRAPPERS"):
        assert option in source
    assert "FILTER_METHODS" in source
    assert "EMBEDDED_METHODS" in source
    assert "WRAPPER_METHODS" in source
