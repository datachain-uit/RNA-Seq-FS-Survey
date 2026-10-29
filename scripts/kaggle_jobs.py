"""Danh mục việc chạy trên Kaggle cho survey v3 (E0–E7) — một nơi duy nhất chứa lệnh, chia phiên và kiểm tra (M18).

Notebook `notebooks/kaggle_survey_v3_runner.ipynb` chỉ đặt JOB và gọi script này; mọi lệnh thí nghiệm, cách chia shard,
điều kiện đầu vào và kiểm tra đầu ra nằm ở đây (đã commit). Không có tham số giao thức nào ở đây: k, n_T, seed, lưới… lấy từ
`survey_v2/config.py` qua mặc định của từng script.

    python scripts/kaggle_jobs.py plan                     # lịch: đợt, hạn, thứ tự, đầu vào cần khôi phục, ước lượng giờ
    python scripts/kaggle_jobs.py show e1_phaseA_geo       # lệnh cụ thể của một việc
    python scripts/kaggle_jobs.py run e1_phaseA_geo        # chạy (shard song song tối đa --parallel tiến trình), rồi verify
    python scripts/kaggle_jobs.py verify e1_phaseA_geo     # config.json: traceable = true, commit_hash = HEAD
    python scripts/kaggle_jobs.py status                   # việc nào đã có đầu ra trong $KLTN_DATA_ROOT/results/survey_v2
    python scripts/kaggle_jobs.py merge-e2e3               # gộp E2_E3_p1..p3 -> E2_E3 (kiểm tra hiệu chuẩn, cùng commit)

Quy ước chia shard:
* Pha A (run_e1_feature_selection.py) không song song bên trong -> chạy nhiều tiến trình, mỗi tiến trình một cohort hoặc
  một lần lặp; checkpoint theo (repeat, fold, method) nên các tiến trình không ghi đè nhau. Bước `finalize` chạy lại lệnh
  đầy đủ một lần để gộp (mọi checkpoint đã có -> chỉ gộp).
* Pha B, E1b, E7, E4/E5 ghi nối tiếp (append) theo bộ dữ liệu -> chỉ song song khác bộ; trong một bộ dùng --n-jobs của script.
* E2/E3 đo thời gian: `exclusive` (một tiến trình, không chạy gì khác cùng phiên; BLAS 1 luồng do script tự đặt).
* Mọi script tự tiếp tục khi chạy lại -> phiên Kaggle quá 12 giờ: lưu output, gắn làm Input, chạy lại cùng JOB.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402

TCGA = C.SOURCE_DATASET
GEO = tuple(C.FEWSHOT_COHORTS)
GPL570 = tuple(C.WRAPPER_DATASETS)
IHC_GENES = ("NKX2-1", "TP63", "NAPSA", "KRT5")
E3_PARTS = {  # chia E3 theo chi phí để mỗi phiên < 12 giờ; mỗi phần tự hiệu chuẩn với phiên tham chiếu (QĐ18)
    "p1": ["f_test", "mutual_info", "mrmr", "l1_lr", "ktsp", "abess"],
    "p2": ["elastic_net", "rf_importance", "stability_selection"],
    "p3": ["relieff", "svm_rfe"],
}


@dataclass
class Job:
    name: str
    batch: str                     # đợt / hạn
    what: str
    shards: list                   # mỗi phần tử: list đối số (sau "python")
    outputs: list                  # thư mục kết quả (tương đối results/survey_v2) phải có config.json traceable
    finalize: list = field(default_factory=list)
    requires: list = field(default_factory=list)    # tệp (tương đối results/survey_v2) phải có trước khi chạy
    restore_from: list = field(default_factory=list)  # tên các việc mà output phải gắn làm Input trước
    parallel: int = 4
    exclusive: bool = False
    hours: str = ""


def _fs(*a):
    return ["scripts/run_e1_feature_selection.py", *a]


def _clf(*a):
    return ["scripts/run_e1_classification.py", *a]


def build_jobs() -> dict[str, Job]:
    J: list[Job] = []
    ref = "E2_E3_ref/calibration.csv"
    # ----------------------------------------------------------------- Đợt 1 (03/11): sanity, E0, E1 GEO, E2 tham chiếu
    J.append(Job("sanity", "1 · 03/11", "Sanity Y10 (normal vs tumor, AUROC >= 0,99)", [["scripts/run_sanity_auroc.py"]],
                 ["sanity"], parallel=1, hours="~1"))
    J.append(Job("e0", "1 · 03/11", "E0 kiểm tra tín hiệu, 5 bộ (200 hoán vị + 100 random-k)",
                 [["scripts/run_survey_e0_signal_check.py", "--n-jobs", "4"]], [f"E0/{d}" for d in C.DATASETS], parallel=1, hours="~1"))
    J.append(Job("e1_phaseA_geo", "1 · 03/11", "E1 pha A, 4 cohort GEO, 11 phương pháp + IHC/panel, 5×5 (một tiến trình/cohort)",
                 [_fs("--datasets", d) for d in GEO], [f"E1_feature_selection/{d}" for d in GEO], hours="3–6"))
    J.append(Job("e1_phaseB_geo", "1 · 03/11", "E1 pha B, 4 cohort GEO: LR-L2 (+RF, SVM ở k ∈ {5, 25}), B0, random-k",
                 [_clf("--datasets", d, "--secondary") for d in GEO], [f"E1_classification/{d}" for d in GEO],
                 requires=[f"E1_feature_selection/{d}/selected_genes.csv" for d in GEO], restore_from=["e1_phaseA_geo"], hours="1–2"))
    J.append(Job("e2_ref", "1 · 03/11", "E2 phiên tham chiếu (đo tại một điểm; ghi calibration.csv; không E3)",
                 [["scripts/run_survey_e2_e3_timing.py", "--wrappers", "--isolate-memory", "--skip-e3", "--out", "E2_E3_ref"]],
                 ["E2_E3_ref"], parallel=1, exclusive=True, hours="2–4"))
    # ----------------------------------------------------------------- Đợt 1 phần TCGA (04/11)
    J.append(Job("e1_phaseA_tcga_r012", "1T · 04/11", "E1 pha A TCGA, lần lặp 0, 1, 2 (ba tiến trình song song)",
                 [_fs("--datasets", TCGA, "--repeats", str(r)) for r in (0, 1, 2)], [], hours="5–8"))
    J.append(Job("e1_phaseA_tcga_r34", "1T · 04/11", "E1 pha A TCGA, lần lặp 3, 4 (song song với phiên r012)",
                 [_fs("--datasets", TCGA, "--repeats", str(r)) for r in (3, 4)], [], hours="5–8"))
    J.append(Job("e1_phaseB_tcga", "1T · 04/11", "Gộp pha A TCGA (5 lần lặp) rồi pha B TCGA",
                 [_clf("--datasets", TCGA, "--secondary")], [f"E1_feature_selection/{TCGA}", f"E1_classification/{TCGA}"],
                 finalize=[], parallel=1, restore_from=["e1_phaseA_tcga_r012", "e1_phaseA_tcga_r34"], hours="3–6"))
    # gộp pha A chạy TRƯỚC pha B (cùng tiến trình tuần tự)
    J[-1].shards = [_fs("--datasets", TCGA), J[-1].shards[0]]
    # ----------------------------------------------------------------- Đợt 2 (06/11): E1b, E3, E4 + E5
    J.append(Job("fewshot_e1b", "2 · 06/11", "E1b đường cong học few-shot, 4 cohort (5 split × 4 tập con)",
                 [["scripts/run_survey_e1b_learning_curve.py", "--n-jobs", "4"]], [f"E1b/{d}" for d in GEO], parallel=1, hours="3–5"))
    for part, methods in E3_PARTS.items():
        extra = ["--wrappers"] if part == "p1" else []
        J.append(Job(f"e3_{part}", "2 · 06/11", f"E2+E3 phần {part} ({', '.join(methods)}{', GA, BPSO' if extra else ''}), hiệu chuẩn theo E2_E3_ref",
                     [["scripts/run_survey_e2_e3_timing.py", "--methods", *methods, *extra, "--isolate-memory",
                       "--calibration-reference", "{RESULTS}/" + ref, "--out", f"E2_E3_{part}"]],
                     [f"E2_E3_{part}"], requires=[ref], restore_from=["e2_ref"], parallel=1, exclusive=True, hours="3–8"))
    J.append(Job("e2e3_merge", "2 · 06/11", "Gộp E2_E3_p1..p3 -> E2_E3 (dùng cho make_survey_tables.py)",
                 [["scripts/kaggle_jobs.py", "merge-e2e3"]], ["E2_E3"],
                 requires=[f"E2_E3_{p}/config.json" for p in E3_PARTS], restore_from=[f"e3_{p}" for p in E3_PARTS], parallel=1, hours="<0,1"))
    J.append(Job("e4e5_gse37745", "2 · 06/11", "E4 (M = 30) + E5a + E5b + E5c trên GSE37745 (+GA, BPSO ở k = 10); "
                 "chạy lại sau wrap_gse37745 để E5c có GA/BPSO (phần đã có được bỏ qua)",
                 [["scripts/run_survey_e4_e5_stability.py", "--datasets", "gse37745", "--wrappers", "--n-jobs", "4"]],
                 ["E4_E5/gse37745"], parallel=1, restore_from=["e1_phaseA_geo"],
                 hours="4–8"))
    J.append(Job("e4_tcga", "2 · 06/11", "E4 (M = 30) + E5c trên TCGA (+GA, BPSO ở k = 10); quá 12 giờ thì chạy lại, tự tiếp tục",
                 [["scripts/run_survey_e4_e5_stability.py", "--datasets", TCGA, "--wrappers", "--skip-e5a", "--skip-e5b", "--n-jobs", "4"]],
                 [f"E4_E5/{TCGA}"], parallel=1, restore_from=["e1_phaseB_tcga"], hours="6–11"))
    # ----------------------------------------------------------------- Đợt 3 (08/11): wrapper, E7
    for d in GPL570:
        J.append(Job(f"wrap_{d}", "3 · 08/11", f"E1 pha A wrapper GA + BPSO, {d}, lần lặp 0–2, k ∈ {{5, 10, 50}}",
                     [["scripts/run_e1_wrappers_all_cohorts.py", "--datasets", d, "--repeats", str(r)] for r in range(C.WRAPPER_REPEATS)],
                     [f"E1_wrapper_feature_selection/{d}"], finalize=[["scripts/run_e1_wrappers_all_cohorts.py", "--datasets", d]],
                     parallel=3, hours="3–6"))
    J.append(Job("e1_phaseB_wrappers", "3 · 08/11", "E1 pha B cho panel GA, BPSO (3 cohort GPL570), nối vào E1_classification",
                 [_clf("--datasets", d, "--methods", "ga", "bpso", "--secondary", "--no-references") for d in GPL570],
                 [f"E1_classification/{d}" for d in GPL570],
                 requires=[f"E1_wrapper_feature_selection/{d}/selected_genes.csv" for d in GPL570]
                 + [f"E1_classification/{d}/classification_per_fold.csv" for d in GPL570],
                 restore_from=["e1_phaseB_geo"] + [f"wrap_{d}" for d in GPL570], parallel=3, hours="~1"))
    J.append(Job("e7", "3 · 08/11", "E7 chuyển giao đơn giản (4 chiến lược × 5 k × 20 lần chạy × n_T ∈ {5, 10, 20})",
                 [["scripts/run_survey_e7_transfer_naive.py", "--n-jobs", "4"]], [f"E7/{d}" for d in GEO], parallel=1, hours="2–4"))
    J.append(Job("tables", "sau mỗi đợt", "Bảng, hình, provenance_manifest (E6 ghép tại k = 10)",
                 [["scripts/make_survey_tables.py"]], ["tables"], parallel=1,
                 restore_from=["(mọi việc đã xong)"], hours="<0,5"))
    # ----------------------------------------------------------------- Ablation loại 4 gen IHC (09–10/11, khoá luận)
    abl = ["--exclude-genes", *IHC_GENES, "--out", "E1_feature_selection_noIHC"]
    J.append(Job("ablation_phaseA_geo", "4 · 10/11", "Ablation không IHC: pha A 4 cohort GEO", [_fs("--datasets", d, *abl) for d in GEO],
                 [f"E1_feature_selection_noIHC/{d}" for d in GEO], hours="3–6"))
    J.append(Job("ablation_phaseA_tcga", "4 · 10/11", "Ablation không IHC: pha A TCGA (5 lần lặp, 4 tiến trình)",
                 [_fs("--datasets", TCGA, "--repeats", str(r), *abl) for r in range(len(C.REPEAT_SEEDS))],
                 [f"E1_feature_selection_noIHC/{TCGA}"], finalize=[_fs("--datasets", TCGA, *abl)], hours="8–11"))
    J.append(Job("ablation_phaseB", "4 · 10/11", "Ablation không IHC: pha B 5 bộ",
                 [_clf("--datasets", d, "--selection-roots", "{RESULTS}/E1_feature_selection_noIHC", "--exclude-genes", *IHC_GENES,
                       "--out", "E1_classification_noIHC") for d in C.DATASETS],
                 [f"E1_classification_noIHC/{d}" for d in C.DATASETS],
                 requires=[f"E1_feature_selection_noIHC/{d}/selected_genes.csv" for d in C.DATASETS],
                 restore_from=["ablation_phaseA_geo", "ablation_phaseA_tcga"], hours="3–6"))
    return {j.name: j for j in J}


JOBS = build_jobs()


# ------------------------------------------------------------------------------------------------- helpers
def data_root() -> Path:
    root = os.environ.get("KLTN_DATA_ROOT")
    if not root:
        raise SystemExit("đặt KLTN_DATA_ROOT")
    return Path(root)


def results_root() -> Path:
    return data_root().joinpath(*C.RESULT_DIR)


def head_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def render(argv: list[str]) -> list[str]:
    res = str(results_root())
    return [a.replace("{RESULTS}", res) for a in argv]


def command(argv: list[str]) -> list[str]:
    return [sys.executable, *render(argv)]


def check_requires(job: Job) -> list[str]:
    root = results_root()
    return [r for r in job.requires if not (root / r).exists()]


TIMEOUT_CODE = 124


def run_parallel(cmds: list[list[str]], max_parallel: int, log_dir: Path, tag: str, deadline: float | None = None) -> list[int]:
    """Chạy các lệnh, tối đa max_parallel cùng lúc. Quá deadline (time.time()): dừng các tiến trình đang chạy (checkpoint
    đã ghi được giữ) và trả mã TIMEOUT_CODE cho các lệnh chưa xong — để notebook kịp lưu output trước giới hạn 12 giờ."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(enumerate(cmds))
    running: dict[int, tuple[subprocess.Popen, object, float]] = {}
    codes: dict[int, int] = {}
    env = os.environ.copy()
    while pending or running:
        if deadline is not None and time.time() > deadline:
            for i, (proc, log, _) in running.items():
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()
                codes[i] = TIMEOUT_CODE
                print(f"[stop  {tag}#{i}] hết thời gian của phiên — checkpoint đã ghi được giữ", flush=True)
            for i, _ in pending:
                codes[i] = TIMEOUT_CODE
            break
        while pending and len(running) < max(1, max_parallel):
            i, cmd = pending.pop(0)
            log = open(log_dir / f"{tag}_{i}.log", "w", encoding="utf-8")
            print(f"[start {tag}#{i}] {' '.join(cmd[1:])}  (log: {log.name})", flush=True)
            running[i] = (subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time())
        time.sleep(5)
        for i in list(running):
            proc, log, t0 = running[i]
            if proc.poll() is not None:
                log.close()
                codes[i] = proc.returncode
                print(f"[done  {tag}#{i}] exit {proc.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
                if proc.returncode != 0:
                    tail = (log_dir / f"{tag}_{i}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                    print("    " + "\n    ".join(tail), flush=True)
                del running[i]
    return [codes[i] for i in sorted(codes)]


def verify(job: Job, commit: str | None) -> list[str]:
    problems = []
    root = results_root()
    for rel in job.outputs:
        cfg_path = root / rel / "config.json"
        if not cfg_path.exists():
            problems.append(f"{rel}: thiếu config.json")
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("traceable") is not True:
            problems.append(f"{rel}: traceable = {cfg.get('traceable')} {cfg.get('traceable_errors', '')}")
        if commit and cfg.get("commit_hash") != commit:
            problems.append(f"{rel}: commit_hash {cfg.get('commit_hash')} khác HEAD {commit}")
        if "calibration_ok" in cfg and cfg.get("calibration_ok") is False:
            problems.append(f"{rel}: calibration_ok = false (QĐ18: không được trích)")
    return problems


# ------------------------------------------------------------------------------------------------- E2/E3 merge
E2E3_TABLES = ("e2_cost_point.csv", "e3_scaling_n.csv", "e3_scaling_p.csv", "e3_wrapper_evals.csv", "e3_exponents.csv")


def merge_e2e3(parts=tuple(f"E2_E3_{p}" for p in E3_PARTS), out_name: str = "E2_E3") -> Path:
    root = results_root()
    out = root / out_name
    out.mkdir(parents=True, exist_ok=True)
    cfgs, problems = {}, []
    for p in parts:
        cfg_path = root / p / "config.json"
        if not cfg_path.exists():
            raise SystemExit(f"thiếu {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfgs[p] = cfg
        if cfg.get("traceable") is not True:
            problems.append(f"{p}: traceable = {cfg.get('traceable')}")
        if cfg.get("calibration_ok") is not True:
            problems.append(f"{p}: calibration_ok = {cfg.get('calibration_ok')}")
    commits = {c.get("commit_hash") for c in cfgs.values()}
    if len(commits) != 1:
        problems.append(f"các phần khác commit: {sorted(map(str, commits))}")
    if problems:
        raise SystemExit("KHÔNG gộp:\n  " + "\n  ".join(problems))
    for name in E2E3_TABLES:
        frames = []
        for p in parts:
            f = root / p / name
            if f.exists() and f.stat().st_size > 3:
                frame = pd.read_csv(f)
                frame.insert(0, "session", p)
                frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / name, index=False, encoding="utf-8-sig")
    for p in parts:
        for name in ("calibration.csv", "calibration_check.json", "machine_config.json"):
            if (root / p / name).exists():
                (out / f"{p}_{name}").write_bytes((root / p / name).read_bytes())
    first = cfgs[parts[0]]
    merged = {"experiment": "E2_E3", "merged_from": list(parts), "merged_by": "scripts/kaggle_jobs.py merge-e2e3",
              "commit_hash": first.get("commit_hash"), "traceable": True, "calibration_ok": True,
              "calibration": {p: c.get("calibration") for p, c in cfgs.items()},
              "machine": {p: c.get("machine") for p, c in cfgs.items()},
              "methods": {p: c.get("methods") for p, c in cfgs.items()}, "dataset": first.get("dataset"), "split": first.get("split"),
              "k": first.get("k"), "k_e3": first.get("k_e3"), "p_fixed": first.get("p_fixed"), "n_grid": first.get("n_grid"),
              "p_grid": first.get("p_grid"), "repeats": first.get("repeats"), "isolate_memory": first.get("isolate_memory"),
              "note": "Mỗi phần đo trên một phiên Kaggle riêng, hiệu chuẩn với E2_E3_ref (QĐ18); cột session ghi phần.",
              "merge_commit_hash": head_commit()}
    (out / "config.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print("gộp xong:", out)
    return out


# ------------------------------------------------------------------------------------------------- CLI
def cmd_plan():
    rows = [(j.batch, j.name, j.hours, ", ".join(j.restore_from) or "—", j.what) for j in JOBS.values()]
    width = max(len(r[1]) for r in rows)
    for batch, name, hours, restore, what in rows:
        print(f"{batch:12s} {name:{width}s}  {hours:>10s} h  khôi phục: {restore}\n{'':12s} {'':{width}s}  {what}")


def cmd_show(job: Job):
    print(f"{job.name} — {job.what}\nđợt {job.batch}; song song {job.parallel}; exclusive {job.exclusive}; ước lượng {job.hours} giờ")
    print("khôi phục (gắn output làm Kaggle Input):", ", ".join(job.restore_from) or "—")
    print("cần có:", ", ".join(job.requires) or "—")
    for s in job.shards:
        print("  python", " ".join(render(s)) if os.environ.get("KLTN_DATA_ROOT") else " ".join(s))
    for s in job.finalize:
        print("  [finalize] python", " ".join(render(s)) if os.environ.get("KLTN_DATA_ROOT") else " ".join(s))
    print("đầu ra kiểm tra:", ", ".join(job.outputs) or "—")


def cmd_run(job: Job, parallel: int | None, log_dir: Path, max_hours: float | None = None) -> int:
    missing = check_requires(job)
    if missing:
        print("THIẾU đầu vào (gắn output các việc " + ", ".join(job.restore_from) + " làm Input):\n  " + "\n  ".join(missing))
        return 2
    commit = head_commit()
    deadline = time.time() + max_hours * 3600 if max_hours else None
    print(f"JOB {job.name} | commit {commit} | data {data_root()} | giới hạn {max_hours or '—'} giờ", flush=True)
    n_par = 1 if job.exclusive else (parallel or job.parallel)
    sequential = job.name == "e1_phaseB_tcga"      # gộp pha A rồi mới pha B
    codes = run_parallel([command(s) for s in job.shards], 1 if sequential else n_par, log_dir, job.name, deadline)
    if TIMEOUT_CODE in codes:
        print(f"TIMEOUT {job.name}: chưa xong trong phiên này. Lưu output, gắn nó làm Input rồi chạy lại cùng JOB (tự tiếp tục).")
        return TIMEOUT_CODE
    if any(codes):
        print(f"LỖI: shard thoát khác 0: {codes}. Output đã có vẫn được giữ; chạy lại cùng JOB sẽ tiếp tục.")
        return 1
    if job.finalize:
        codes = run_parallel([command(s) for s in job.finalize], 1, log_dir, job.name + "_finalize", deadline)
        if TIMEOUT_CODE in codes:
            print(f"TIMEOUT {job.name} ở bước gộp: chạy lại cùng JOB.")
            return TIMEOUT_CODE
        if any(codes):
            return 1
    problems = verify(job, commit)
    if problems:
        print("VERIFY KHÔNG ĐẠT:\n  " + "\n  ".join(problems))
        return 3
    print(f"JOB_OK {job.name}: " + ", ".join(job.outputs))
    return 0


def cmd_status():
    root = results_root()
    for j in JOBS.values():
        have = [o for o in j.outputs if (root / o / "config.json").exists()]
        mark = "xong" if j.outputs and len(have) == len(j.outputs) else (f"{len(have)}/{len(j.outputs)}" if j.outputs else "—")
        print(f"{j.batch:12s} {j.name:24s} {mark}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "list", "show", "run", "verify", "status", "merge-e2e3"])
    ap.add_argument("job", nargs="?")
    ap.add_argument("--parallel", type=int, default=None, help="số tiến trình song song tối đa (mặc định theo việc)")
    ap.add_argument("--log-dir", type=Path, default=Path(os.environ.get("KLTN_LOG_DIR", "logs")))
    ap.add_argument("--max-hours", type=float, default=None, help="dừng có kiểm soát sau số giờ này (Kaggle: 11)")
    args = ap.parse_args(argv)
    if args.command in ("plan", "list"):
        cmd_plan()
        return 0
    if args.command == "status":
        cmd_status()
        return 0
    if args.command == "merge-e2e3":
        merge_e2e3()
        return 0
    if args.job not in JOBS:
        ap.error(f"JOB phải là một trong: {', '.join(JOBS)}")
    job = JOBS[args.job]
    if args.command == "show":
        cmd_show(job)
        return 0
    if args.command == "verify":
        problems = verify(job, head_commit())
        print("VERIFY_OK" if not problems else "VERIFY KHÔNG ĐẠT:\n  " + "\n  ".join(problems))
        return 0 if not problems else 3
    return cmd_run(job, args.parallel, args.log_dir, args.max_hours)


if __name__ == "__main__":
    sys.exit(main())
