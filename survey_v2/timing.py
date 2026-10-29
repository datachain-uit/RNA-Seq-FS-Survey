"""Đo thời gian và bộ nhớ đỉnh của bước chọn gen theo n và p (E2–E3; đoạn code I của Review 24/10/2026).

Chạy trên MỘT máy cố định, giới hạn BLAS về 1 luồng TRƯỚC khi import numpy:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_survey_e2_e3_timing.py
(script tự đặt các biến này khi khởi động, nhưng đặt từ shell là chắc chắn nhất). Trung vị 3 lần.
Bộ nhớ: tracemalloc theo dõi cấp phát của Python/NumPy; thêm ru_maxrss (RSS tối đa của tiến trình,
đơn vị KB trên Linux, byte trên macOS) để đối chiếu. v3.1 (Y19, Review 29/10): tracemalloc KHÔNG thấy cấp phát của
mã biên dịch (cây RF, liblinear, skrebate), và ru_maxrss của một tiến trình dài phụ thuộc thứ tự chạy; vì vậy
`measure_isolated()` chạy từng phương pháp trong một tiến trình con mới và lấy ru_maxrss của tiến trình con trừ
mức nền (tiến trình con chỉ nạp dữ liệu). Đây là số bộ nhớ đỉnh dùng cho bảng E2.
"""

from __future__ import annotations

import os
import platform
import time
import tracemalloc

try:
    import resource
except ImportError:  # Windows does not provide the POSIX resource module.
    resource = None

import numpy as np
import pandas as pd


def machine_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "blas_threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
        info["cpu_freq_mhz"] = getattr(psutil.cpu_freq(), "max", None)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_info
        info["threadpools"] = [{k: v for k, v in d.items() if k in ("internal_api", "num_threads", "version")}
                               for d in threadpool_info()]
    except Exception:
        pass
    try:  # tên CPU trên Linux
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    if platform.system() == "Darwin":
        try:
            import subprocess
            info["cpu_model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return info


def _maxrss_mb() -> float:
    if resource is None:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 2**20
        except Exception:
            return float("nan")
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 2**20 if platform.system() == "Darwin" else v / 1024.0


def measure(select, X, y, k, repeats: int = 3) -> dict:
    """select(X, y, k) -> chỉ số gen. Trả về trung vị wall, CPU (giây), bộ nhớ đỉnh tracemalloc (MB),
    và ru_maxrss sau lần chạy cuối (MB); kèm từng lần đo để báo khoảng dao động."""
    walls, cpus, peaks = [], [], []
    for _ in range(repeats):
        tracemalloc.start()
        t0, c0 = time.perf_counter(), time.process_time()
        select(X, y, k)
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
        peaks.append(tracemalloc.get_traced_memory()[1] / 2**20)
        tracemalloc.stop()
    return {"wall_s": float(np.median(walls)), "cpu_s": float(np.median(cpus)), "peak_mb": float(np.median(peaks)),
            "wall_s_all": walls, "cpu_s_all": cpus, "peak_mb_all": peaks, "maxrss_mb": _maxrss_mb(),
            "repeats": repeats}


def subsample_rows(y: np.ndarray, n: int | None, rng: np.random.Generator) -> np.ndarray:
    """Lấy n mẫu phân tầng theo nhãn (None = tất cả)."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    idx = []
    for c in (0, 1):
        cand = np.flatnonzero(y == c)
        m = int(round(n * np.mean(y == c)))
        idx.append(rng.choice(cand, size=max(2, min(m, len(cand))), replace=False))
    return np.sort(np.concatenate(idx))


def top_variance_columns(X: np.ndarray, p: int | None) -> np.ndarray:
    if p is None or p >= X.shape[1]:
        return np.arange(X.shape[1])
    return np.sort(np.argsort(-X.var(axis=0))[:p])


def empirical_exponent(df: pd.DataFrame, var: str, time_col: str = "wall_s", min_points: int = 3) -> float:
    """Hệ số góc log–log của thời gian theo n hoặc p (biến còn lại cố định). NaN nếu < min_points."""
    d = df[[var, time_col]].dropna()
    d = d[(d[var] > 0) & (d[time_col] > 0)]
    if len(d) < min_points:
        return float("nan")
    slope, _ = np.polyfit(np.log(d[var].to_numpy(float)), np.log(d[time_col].to_numpy(float)), 1)
    return float(slope)


# --- Bộ nhớ đỉnh trong tiến trình con (v3.1, Y19) -----------------------------------------------------------------
def _child_maxrss(npz_path, method, k, seed, pop=None, generations=None, timeout=None) -> dict:
    """Chạy `python -m survey_v2.timing_child` và đọc JSON từ stdout."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "survey_v2.timing_child", "--npz", str(npz_path), "--method", str(method), "--k", str(int(k)),
           "--seed", str(int(seed))]
    if pop:
        cmd += ["--pop", str(int(pop)), "--generations", str(int(generations))]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(repo), timeout=timeout)
    if res.returncode != 0:
        return {"error": (res.stderr or res.stdout).strip()[-500:]}
    line = [l for l in res.stdout.strip().splitlines() if l.startswith("{")][-1]
    return json.loads(line)


def measure_isolated(X, y, k, method, seed, pop=None, generations=None, baseline: dict | None = None, timeout=None) -> dict:
    """Bộ nhớ đỉnh (ru_maxrss, MB) của một tiến trình con chạy đúng một lần chọn gen; trừ mức nền (`baseline`, tiến trình
    con chỉ nạp cùng dữ liệu, lấy bằng method = "none"). Trả về {"maxrss_isolated_mb", "maxrss_child_mb", "maxrss_baseline_mb",
    "wall_s_child"}; NaN kèm "error" nếu tiến trình con lỗi (vd. abess chưa cài, k-TSP p quá lớn)."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix="kltn_timing_") as td:
        npz = Path(td) / "xy.npz"
        np.savez(npz, X=np.ascontiguousarray(X), y=np.asarray(y, dtype=int))
        if baseline is None:
            baseline = _child_maxrss(npz, "none", k, seed, timeout=timeout)
        child = _child_maxrss(npz, method, k, seed, pop, generations, timeout=timeout)
    out = {"maxrss_baseline_mb": float(baseline.get("maxrss_mb", float("nan"))),
           "maxrss_child_mb": float(child.get("maxrss_mb", float("nan"))), "wall_s_child": float(child.get("wall_s", float("nan")))}
    out["maxrss_isolated_mb"] = out["maxrss_child_mb"] - out["maxrss_baseline_mb"] if "error" not in child else float("nan")
    if "error" in child:
        out["error"] = child["error"]
    return out
