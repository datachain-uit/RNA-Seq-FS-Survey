r"""Ghi nguồn gốc (provenance) cho mỗi lần chạy script thực nghiệm — sửa O3, Y8 (Review 20/10/2026, §7.7).

Cách dùng (đặt ở ĐẦU main(), trước khi tính toán hay ghi file):

    from provenance import provenance, save_provenance      # khi chạy: python scripts/<x>.py
    prov = provenance(script=__file__, data_files=[...])     # dừng ngay nếu không truy vết được
    ...
    config["commit_hash"] = prov["commit_hash"]
    config["provenance"] = prov                              # hoặc save_provenance(out_dir, prov)

Quy tắc:
- Thư mục dự án phải là git clone. Nếu không: DỪNG.
- File đã được Git theo dõi mà còn thay đổi chưa commit: DỪNG (allow_dirty=True để bỏ qua,
  khi đó config ghi dirty = true). File chưa được theo dõi (vd. kết quả mới) không tính.
- Ngoại lệ tạm thời khi chưa có Git (task 3.1): đặt biến môi trường KLTN_ALLOW_UNTRACKED=1.
  Khi đó script chạy tiếp nhưng config ghi commit_hash = null, tracked = false và lý do,
  để kết quả không bao giờ bị nhầm là đã truy vết được.

Code và dữ liệu ở hai nơi:
- CODE_ROOT: thư mục kho Git chứa script này (vd. D:\KLTN-github\RNA-Seq-FS-Survey). Git được kiểm tra ở đây.
- DATA_ROOT: thư mục dữ liệu và kết quả (vd. D:\KLTN), đặt bằng biến môi trường KLTN_DATA_ROOT;
  mặc định trùng CODE_ROOT. Dữ liệu được ghi nhận bằng SHA-256, không cần nằm trong Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO
ENV_ALLOW_UNTRACKED = "KLTN_ALLOW_UNTRACKED"
ENV_DATA_ROOT = "KLTN_DATA_ROOT"


def data_root() -> Path:
    """Thư mục dữ liệu/kết quả: KLTN_DATA_ROOT nếu có, ngược lại là CODE_ROOT."""
    value = os.environ.get(ENV_DATA_ROOT, "").strip()
    return Path(value).expanduser().resolve() if value else CODE_ROOT


DATA_ROOT = data_root()
PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "tables", "openpyxl",
            "requests", "xgboost", "lightgbm", "abess")
SCHEMA = "kltn-provenance/1"


class ProvenanceError(RuntimeError):
    """Không truy vết được lần chạy về mã nguồn."""


def _git(*args: str, cwd: Path) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _rel(path: str | Path, root: Path) -> str:
    """Đường dẫn tương đối với kho code; nếu nằm trong DATA_ROOT thì ghi dạng $KLTN_DATA_ROOT/..."""
    p = Path(path).resolve()
    try:
        return p.relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return f"${ENV_DATA_ROOT}/" + p.relative_to(data_root()).as_posix()
    except ValueError:
        return p.as_posix()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _git_state(repo: Path) -> dict | None:
    """Trả về trạng thái git, hoặc None nếu repo không phải git clone / không có lệnh git."""
    try:
        top = Path(_git("rev-parse", "--show-toplevel", cwd=repo)).resolve()
        commit = _git("rev-parse", "HEAD", cwd=repo)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    changed = _git("status", "--porcelain", "--untracked-files=no", cwd=repo)
    try:
        remote = _git("config", "--get", "remote.origin.url", cwd=repo) or None
    except subprocess.CalledProcessError:
        remote = None
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)
    except subprocess.CalledProcessError:
        branch = None
    return {"toplevel": top, "commit": commit, "changed": changed, "remote": remote, "branch": branch}


def provenance(script: str | Path | None = None, data_files: Iterable[str | Path] = (),
               repo_dir: str | Path = REPO, allow_dirty: bool = False,
               allow_untracked: bool | None = None, params: dict | None = None) -> dict:
    """Thu thập provenance; raise ProvenanceError nếu không truy vết được (xem docstring module)."""
    repo = Path(repo_dir).resolve()
    if allow_untracked is None:
        allow_untracked = _env_flag(ENV_ALLOW_UNTRACKED)

    state = _git_state(repo)
    if state is None:
        if not allow_untracked:
            raise ProvenanceError(
                f"{repo} không phải git clone của kho mã (hoặc không có lệnh git), nên kết quả không "
                f"truy vết được. Clone kho rồi chạy lại; nếu buộc phải chạy tạm, đặt {ENV_ALLOW_UNTRACKED}=1.")
        warnings.warn(f"Chạy KHÔNG truy vết ({ENV_ALLOW_UNTRACKED}=1): commit_hash = null.", stacklevel=2)
        git = {"commit_hash": None, "tracked": False, "dirty": None, "branch": None, "remote": None,
               "untracked_reason": f"không phải git clone; chạy với {ENV_ALLOW_UNTRACKED}=1"}
    else:
        dirty = bool(state["changed"])
        if dirty and not allow_dirty:
            raise ProvenanceError("Còn thay đổi chưa commit ở file đã được Git theo dõi:\n"
                                  + state["changed"] + "\nCommit trước khi chạy để kết quả truy vết được.")
        git = {"commit_hash": state["commit"], "tracked": True, "dirty": dirty,
               "branch": state["branch"], "remote": state["remote"], "untracked_reason": None}
        if dirty:
            git["dirty_files"] = state["changed"].splitlines()

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass

    data = {}
    for f in data_files:
        data[_rel(f, repo)] = sha256(f)

    return {
        "schema": SCHEMA,
        **git,
        "script": _rel(script, repo) if script else None,
        "code_root": repo.as_posix(),
        "data_root": data_root().as_posix(),
        "argv": sys.argv[1:],
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": versions,
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "machine": platform.machine(), "cpu_count": os.cpu_count(),
                    "hostname": platform.node()},
        "data_sha256": data,
        "params": params or {},
    }


def add_data_files(prov: dict, files: Iterable[str | Path], repo_dir: str | Path = REPO) -> dict:
    """Bổ sung SHA-256 cho file đầu vào chỉ biết sau khi script đã bắt đầu."""
    repo = Path(repo_dir).resolve()
    for f in files:
        prov["data_sha256"][_rel(f, repo)] = sha256(f)
    return prov


def stamp(config: dict, prov: dict) -> dict:
    """Gắn provenance vào một config dict (giữ khoá commit_hash ở mức trên cho các script cũ đọc)."""
    config["commit_hash"] = prov["commit_hash"]
    config["provenance"] = prov
    return config


def save_provenance(out_dir: str | Path, prov: dict, filename: str = "provenance.json",
                    outputs: Iterable[str | Path] = ()) -> Path:
    """Ghi provenance.json cạnh kết quả; outputs (nếu có) được ghi kèm SHA-256."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(prov)
    record["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if outputs:
        record["outputs_sha256"] = {_rel(f, REPO): sha256(f) for f in outputs}
    path = out_dir / filename
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def require(script: str | Path, **kw) -> dict:
    """Như provenance() nhưng in lỗi gọn và thoát mã 2 thay vì traceback (dùng trong main())."""
    try:
        return provenance(script=script, **kw)
    except ProvenanceError as e:
        print(f"[provenance] DỪNG: {e}", file=sys.stderr)
        raise SystemExit(2) from None
