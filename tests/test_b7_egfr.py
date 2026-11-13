"""B7 bước 1 (QĐ188): quy tắc nhãn EGFR/TP53, đọc trường mô tả theo khoá (Y98), thang giá trị, gom probe, cổng G1/G2,
mã seed của năm bộ cũ; một lần chạy trọn build_b7_egfr_datasets (tcga, gse31210, card) và b7_gate trên dữ liệu giả nhỏ."""

from __future__ import annotations

import gzip
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from survey_v2 import b7_egfr as B  # noqa: E402
from survey_v2 import config as C  # noqa: E402

OLD_FIVE = ("tcga_luad_lusc_primary_tumor", "gse10245", "gse30219", "gse37745", "gse42127")


# ------------------------------------------------------------------------------------------------- registry
def test_old_five_keep_seed_ids_and_cohort_index():
    assert C.MAIN_DATASETS == OLD_FIVE
    assert [C.DATASET_SEED_IDS[n] for n in OLD_FIVE] == [100, 101, 102, 103, 104]
    cohort_index = {name: i for i, name in enumerate(C.DATASETS)}            # COHORT_ID của run_survey_e4_e5_stability.py
    assert [cohort_index[n] for n in OLD_FIVE] == [0, 1, 2, 3, 4]
    assert list(C.DATASETS)[-2:] == list(C.B7_DATASETS) and [C.DATASET_SEED_IDS[n] for n in C.B7_DATASETS] == [105, 106]
    assert C.V3_EXPERIMENT_DATASETS == OLD_FIVE
    for n in C.B7_DATASETS:
        assert n in C.DATASET_DIRS and n in C.DATASET_PLATFORM


def test_old_script_defaults_stay_on_the_five_main_datasets():
    sys.path.insert(0, str(REPO / "scripts"))
    for mod in ("run_survey_e0_signal_check", "run_e1_classification", "run_e1_feature_selection", "build_e1_repeated_splits"):
        sys.modules.pop(mod, None)
        args = importlib.import_module(mod).parse_args([])
        assert tuple(args.datasets) == OLD_FIVE, mod


# ------------------------------------------------------------------------------------------------- TCGA labels
def _row(hg, vc, exon=None, pos=None):
    return {"HGVSp_Short": hg, "Variant_Classification": vc, "Exon_Number": exon, "Protein_position": pos}


@pytest.mark.parametrize("row, expected", [
    (_row("p.L858R", "Missense_Mutation", "21/28"), "L858R"),
    (_row("p.L861Q", "Missense_Mutation", "21/28"), "L861Q"),
    (_row("p.G719S", "Missense_Mutation", "18/28"), "G719X"),
    (_row("p.S768I", "Missense_Mutation", "20/28"), "S768I"),
    (_row("p.E746_A750del", "In_Frame_Del", "19/28", "746-750/1210"), "ex19del"),
    (_row("p.L747_P753delinsS", "In_Frame_Del", None, "747-753/1210"), "ex19del"),
    (_row("p.A767_V769dup", "In_Frame_Ins", "20/28"), "ex20ins"),
    (_row("p.E709_T710delinsD", "In_Frame_Del", "18/28", "709-710/1210"), None),    # exon 18: loại theo quy tắc (QĐ186)
    (_row("p.S752Pfs*3", "Frame_Shift_Del", "19/28"), None),                       # dịch khung, không phải mất đoạn khung đọc
    (_row("p.T790M", "Missense_Mutation", "20/28"), None),
])
def test_activating_class(row, expected):
    assert B.activating_class(row) == expected


def _maf(rows):
    cols = ["Hugo_Symbol", "Variant_Classification", "HGVSp_Short", "Exon_Number", "Protein_position", "Tumor_Sample_Barcode",
            "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2", "FILTER"]
    return pd.DataFrame(rows, columns=cols)


def test_dedup_merges_aliquot_duplicates_and_keeps_file_ids():
    v = _maf([["EGFR", "Missense_Mutation", "p.L858R", "21/28", "858/1210", "S1-01A-11D", "chr7", "1", "T", "G", "PASS"],
              ["EGFR", "Missense_Mutation", "p.L858R", "21/28", "858/1210", "S1-01A-21D", "chr7", "1", "T", "G", "PASS"],
              ["TP53", "Missense_Mutation", "p.R175H", "5/11", "175/393", "S1-01A-11D", "chr17", "2", "C", "T", "PASS"]])
    v["Tumor_Sample_Barcode"] = v["Tumor_Sample_Barcode"].str[:2]
    v["maf_file_id"] = ["f1", "f2", "f1"]
    d = B.dedup_variants(v)
    assert len(d) == 2 and d.loc[d.Hugo_Symbol == "EGFR", "maf_file_id"].iloc[0] == "f1;f2"


def test_label_tcga_luad_rules():
    v = _maf([["EGFR", "Missense_Mutation", "p.L858R", "21/28", "858", "A", "chr7", "1", "T", "G", "PASS"],
              ["EGFR", "Missense_Mutation", "p.T790M", "20/28", "790", "A", "chr7", "3", "C", "T", "PASS"],     # T790M + L858R: đột biến
              ["EGFR", "Missense_Mutation", "p.T790M", "20/28", "790", "B", "chr7", "3", "C", "T", "PASS"],     # chỉ T790M: loại
              ["EGFR", "Silent", "p.Q787=", "20/28", "787", "C", "chr7", "4", "G", "A", "PASS"],               # đồng nghĩa: hoang dại
              ["TP53", "Nonsense_Mutation", "p.R213*", "6/11", "213", "C", "chr17", "5", "C", "T", "PASS"]])
    d = B.dedup_variants(v)
    luad = pd.DataFrame({"sample_id": ["A", "B", "C", "D", "E"], "patient_id": ["pa", "pb", "pc", "pd", "pe"]})
    lab = B.label_tcga_luad(d, luad, maf_samples={"A", "B", "C", "D"}).set_index("sample_id")
    assert lab.label_status.to_dict() == {"A": "mutant", "B": "excluded_other_egfr", "C": "wildtype", "D": "wildtype", "E": "no_maf"}
    assert lab.loc["A", "label"] == 1 and lab.loc["C", "label"] == 0 and np.isnan(lab.loc["B", "label"]) and np.isnan(lab.loc["E", "label"])
    assert lab.tp53_label.to_dict() == {"A": "wildtype", "B": "wildtype", "C": "mutant", "D": "wildtype", "E": "no_maf"}
    assert lab.loc["A", "activating_classes"] == "L858R"
    B.check_counts(lab.label_status.value_counts().to_dict(), {"mutant": 1, "wildtype": 2, "excluded_other_egfr": 1, "no_maf": 1}, "t")
    with pytest.raises(ValueError):
        B.check_counts(lab.label_status.value_counts().to_dict(), {"mutant": 2, "wildtype": 1, "excluded_other_egfr": 1, "no_maf": 1}, "t")


# ------------------------------------------------------------------------------------------------- GSE31210
def test_characteristics_by_key_does_not_shift_values_y98():
    cells = {"G1": ["tissue: primary lung tumor", "gender: male", "gene alteration status: EGFR mutation +", "cluster: A", "relapse: relapsed"],
             "G2": ["tissue: primary lung tumor", "gender: female", "gene alteration status: KRAS mutation +", "relapse: not relapsed", ""],
             "G3": ["tissue: normal lung", "gender: male", "", "", ""]}
    t = B.characteristics_by_key(cells)
    assert t.loc["G2", "relapse"] == "not relapsed" and pd.isna(t.loc["G2", "cluster"]) and t.loc["G1", "cluster"] == "A"
    with pytest.raises(ValueError):
        B.characteristics_by_key({"G1": ["tissue: a", "tissue: b"]})
    with pytest.raises(ValueError):
        B.characteristics_by_key({"G1": ["gene alteration status: EGFR mutation +", "gene alteration status: KRAS mutation +"]})
    d = B.characteristics_by_key({"G1": ["tissue: t", "months before relapse/censor: 8.4", "months before relapse/censor: 14.6"]})
    assert d.loc["G1", "months before relapse/censor"] == "8.4 | 14.6" and d.loc["G1", "tissue"] == "t"
    assert d.attrs["duplicate_keys"] == [{"sample_id": "G1", "key": "months before relapse/censor", "values": ["8.4", "14.6"]}]


def test_label_gse31210_qd185():
    chars = pd.DataFrame({"tissue": ["primary lung tumor"] * 4 + ["normal lung"],
                          "gene alteration status": ["EGFR mutation +", "EGFR/KRAS/ALK -", "KRAS mutation +", "ALK-fusion +", np.nan]},
                         index=list("abcde"))
    lab = B.label_gse31210(chars)
    assert lab.label.tolist()[:4] == [1, 0, 0, 0] and np.isnan(lab.label.iloc[4])
    assert lab.label_status.tolist() == ["positive", "negative", "negative", "negative", "normal_excluded"]
    chars.loc["b", "gene alteration status"] = "something else"
    with pytest.raises(ValueError):
        B.label_gse31210(chars)


def test_geo_urls_follow_the_ftp_stub_rule():
    assert B.geo_stub("GPL570") == "GPLnnn" and B.geo_stub("GPL6884") == "GPL6nnn" and B.geo_stub("GSE31210") == "GSE31nnn"
    assert B.geo_annot_url("GPL570") == "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
    assert B.geo_series_matrix_url("GSE31210").endswith("/geo/series/GSE31nnn/GSE31210/matrix/GSE31210_series_matrix.txt.gz")


def test_decide_scale_rule_a2():
    lin = pd.DataFrame([[0.5, 50.0, 5000.0]] * 10)
    out, info = B.decide_scale(lin)
    assert info["scale"].startswith("linear") and out.iloc[0, 0] == 0.0 and np.isclose(out.iloc[0, 2], np.log2(5000))
    logd = pd.DataFrame([[2.0, 8.0, 14.0]] * 10)
    out, info = B.decide_scale(logd)
    assert info["scale"].startswith("log2") and out.equals(logd)


def test_collapse_probes_keeps_highest_variance_and_drops_missing():
    expr = pd.DataFrame({"s1": [1.0, 5.0, 1.0, np.nan], "s2": [2.0, 1.0, 1.1, 3.0], "s3": [3.0, 9.0, 0.9, 4.0]}, index=["p1", "p2", "p3", "p4"])
    sym = pd.Series({"p1": "GA", "p2": "GA", "p3": "GB", "p4": "GC"})
    X = B.collapse_probes(expr, sym)
    assert list(X.columns) == ["GA", "GB"] and np.allclose(X["GA"].to_numpy(), [5.0, 1.0, 9.0])   # GC: probe thiếu giá trị bị bỏ


# ------------------------------------------------------------------------------------------------- gate
def _gate_inputs(name, b0, rk, p_perm):
    e0 = pd.DataFrame([{"cohort": name, "n": 100, "p": 500, "k": 10, "auroc_real": 0.7, "null_mean": 0.5, "p_perm": p_perm, "n_perm": 200}])
    rows = []
    for r in range(5):
        for f in range(5):
            rows += [{"cohort": name, "repeat": r, "fold": f, "method": "b0", "k": 500, "model": "lr_l2", "auroc": b0, "n_draws": np.nan},
                     {"cohort": name, "repeat": r, "fold": f, "method": "random_k", "k": 10, "model": "lr_l2", "auroc": rk, "n_draws": 100}]
    return e0, pd.DataFrame(rows)


def test_gate_row_pass_and_fail():
    sys.path.insert(0, str(REPO / "scripts"))
    sys.modules.pop("b7_gate", None)
    g = importlib.import_module("b7_gate")
    e0, pf = _gate_inputs("tcga_luad_egfr", 0.80, 0.70, 0.005)
    r = g.gate_row("tcga_luad_egfr", e0, pf, None)
    assert r["G1_pass"] and r["G2_pass"] and r["gate_pass"] and np.isclose(r["b0_minus_random_k"], 0.10)
    e0, pf = _gate_inputs("tcga_luad_egfr", 0.72, 0.70, 0.005)
    r = g.gate_row("tcga_luad_egfr", e0, pf, None)
    assert r["G1_pass"] and not r["G2_pass"] and "TP53" in r["next_step"]
    e0, pf = _gate_inputs("gse31210_egfr", 0.80, 0.70, 0.2)
    r = g.gate_row("gse31210_egfr", e0, pf, None)
    assert not r["G1_pass"] and not r["gate_pass"] and "report" in r["next_step"]
    with pytest.raises(ValueError):
        g.gate_row("tcga_luad_egfr", e0.assign(cohort="tcga_luad_egfr"), pf.iloc[:-2].assign(cohort="tcga_luad_egfr"), None)


def test_step1_driver_order_and_job(capsys):
    sys.path.insert(0, str(REPO / "scripts"))
    for m in ("run_b7_step1", "kaggle_jobs"):
        sys.modules.pop(m, None)
    drv = importlib.import_module("run_b7_step1")
    plan = drv.steps("QC", 4)
    names = [" ".join(s[:2]) for s in plan]
    assert len(plan) == 12 and plan[0][1] == "tcga" and plan[1][1] == "gse31210" and plan[-1][1] == "pack"
    assert names.index("scripts/b7_gate.py --e0-dir") > names.index("scripts/run_e1_classification.py --datasets")
    assert plan[9][plan[9].index("--k") + 1] == "10" and "--save-scores" in plan[9]
    assert drv.main(["--list"]) == 0 and capsys.readouterr().out.count("python scripts/") == 12
    job = importlib.import_module("kaggle_jobs").JOBS["b7_step1"]
    assert job.shards == [["scripts/run_b7_step1.py", "--n-jobs", "4"]] and "B7_gate" in job.outputs and "B7_data" in job.outputs
    nb = json.loads((REPO / "notebooks" / "kaggle" / "kaggle_B7_step1.ipynb").read_text(encoding="utf-8"))
    cfg = "".join(nb["cells"][1]["source"])
    assert "REPO_BRANCH = 'dmkd-revision'" in cfg and "JOB = 'b7_step1'" in cfg


# ------------------------------------------------------------------------------------------------- chạy trọn trên dữ liệu giả
@pytest.fixture(scope="module")
def b7_root(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("kltn_b7")
    for k, v in {"KLTN_DATA_ROOT": str(root), "KLTN_ALLOW_UNPUSHED": "1", "KLTN_ALLOW_UNTRACKED": "1", "KLTN_STRICT_N": "0"}.items():
        mp.setenv(k, v)
    for m in list(sys.modules):
        if m.startswith("scripts.") or m == "scripts" or m.startswith("survey_v2.runlog") or m.startswith("run_") or m in ("provenance", "build_b7_egfr_datasets", "b7_gate", "build_e1_repeated_splits"):
            del sys.modules[m]
    sys.path.insert(0, str(REPO / "scripts"))
    from survey_v2 import synthetic
    synthetic.make_all(root, n=120, p=300, effect=0.9, n_target=60)
    yield root, mp
    mp.undo()


def _write_fake_gdc(root: Path) -> dict:
    from survey_v2.data import load_dataset
    src = load_dataset(root, C.SOURCE_DATASET, gene_space=None)
    meta = pd.read_csv(src.data_dir / "metadata.csv").set_index("sample_id")
    luad = [s for s in src.sample_ids if meta.loc[s, "class"] == "LUAD"]
    gdc = root / "_b7_raw" / "gdc"
    (gdc / "maf").mkdir(parents=True)
    files, plan = [], {}
    rng = np.random.default_rng(7)
    for i, s in enumerate(luad):
        if i % 10 == 9:
            plan[s] = "no_maf"
            continue
        fid = f"f{i:04d}"
        files.append({"file_id": fid, "file_name": f"{fid}.maf.gz", "md5sum": "", "strategy": "WXS", "workflow": "Aliquot Ensemble",
                      "case": s, "sample": s, "sample_type": "Primary Tumor"})
        rows = [["KRAS", "Missense_Mutation", "p.G12C", "2/6", "12", s, "chr12", "9", "C", "A", "PASS"]]
        if i % 5 == 0:
            rows.append(["EGFR", "Missense_Mutation", "p.L858R", "21/28", "858", s, "chr7", "1", "T", "G", "PASS"])
            plan[s] = "mutant"
        elif i % 7 == 0:
            rows.append(["EGFR", "Missense_Mutation", "p.R108K", "3/28", "108", s, "chr7", "2", "G", "A", "PASS"])
            plan[s] = "excluded_other_egfr"
        else:
            plan[s] = "wildtype"
        if rng.random() < 0.5:
            rows.append(["TP53", "Missense_Mutation", "p.R175H", "5/11", "175", s, "chr17", "5", "C", "T", "common_variant"])
        body = "#version 2.4\n" + "\t".join(["Hugo_Symbol", "Variant_Classification", "HGVSp_Short", "Exon_Number", "Protein_position",
                                              "Tumor_Sample_Barcode", "Chromosome", "Start_Position", "Reference_Allele",
                                              "Tumor_Seq_Allele2", "FILTER"]) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"
        (gdc / "maf" / f"{fid}.maf.gz").write_bytes(gzip.compress(body.encode()))
    pd.DataFrame(files).to_csv(gdc / "gdc_files.csv", index=False)
    (root / "status.json").write_text(json.dumps({"data_release": "Data Release 46.0 - August 10, 2026"}))
    return plan


def _write_fake_gse31210(root: Path, genes) -> None:
    rng = np.random.default_rng(3)
    geo = root / "_b7_raw" / "geo"
    geo.mkdir(parents=True)
    n_t, n_n = 50, 4
    gsm = [f"GSM{i}" for i in range(n_t + n_n)]
    status = (["EGFR mutation +"] * 25 + ["EGFR/KRAS/ALK -"] * 15 + ["KRAS mutation +"] * 6 + ["ALK-fusion +"] * 4)
    chars = [[f"tissue: {'primary lung tumor' if i < n_t else 'normal lung'}" for i in range(n_t + n_n)],
             [f"gender: {'male' if i % 2 else 'female'}" for i in range(n_t + n_n)],
             [f"gene alteration status: {status[i]}" if i < n_t else "" for i in range(n_t + n_n)]]
    probes = [f"{g}_at" for g in genes[:120]] + [f"{genes[0]}_x_at"]
    lin = 2 ** rng.normal(7, 1.5, size=(len(probes), n_t + n_n))                       # thang tuyến tính (p99 > 100)
    y = np.array([1 if s == "EGFR mutation +" else 0 for s in status] + [0] * n_n)
    lin[:10] *= np.where(y == 1, 3.0, 1.0)
    lines = ["!Series_title\t\"fake\"", "!Sample_geo_accession\t" + "\t".join(f'"{g}"' for g in gsm),
             "!Sample_platform_id\t" + "\t".join('"GPL570"' for _ in gsm),
             "!Sample_title\t" + "\t".join(f'"{g}"' for g in gsm)]
    lines += ["!Sample_characteristics_ch1\t" + "\t".join(f'"{c}"' for c in row) for row in chars]
    lines += ["!Sample_data_processing\t" + "\t".join('"MAS5"' for _ in gsm), "!series_matrix_table_begin",
              "\"ID_REF\"\t" + "\t".join(f'"{g}"' for g in gsm)]
    lines += [f"\"{p}\"\t" + "\t".join(f"{v:.4f}" for v in lin[i]) for i, p in enumerate(probes)]
    lines += ["!series_matrix_table_end"]
    (geo / "GSE31210_series_matrix.txt.gz").write_bytes(gzip.compress("\n".join(lines).encode()))
    ann = ["^Annotation", "!platform_table_begin", "ID\tGene title\tGene symbol"]
    ann += [f"{p}\tt\t{p.split('_')[0]}" for p in probes] + ["!platform_table_end"]
    (geo / "GPL570.annot.gz").write_bytes(gzip.compress("\n".join(ann).encode()))


@pytest.mark.slow
def test_build_tcga_gse31210_card_and_gate_end_to_end(b7_root, monkeypatch):
    root, _ = b7_root
    build = importlib.import_module("build_b7_egfr_datasets")
    plan = _write_fake_gdc(root)
    counts = pd.Series(plan).value_counts().to_dict()
    monkeypatch.setattr(C, "B7_TCGA_EGFR_COUNTS", counts)
    # TP53: đếm từ chính tệp giả
    maf_dir = root / "_b7_raw" / "gdc" / "maf"
    tp53 = sum("TP53" in gzip.decompress(p.read_bytes()).decode() for p in maf_dir.glob("*.maf.gz"))
    monkeypatch.setattr(C, "B7_TCGA_TP53_COUNTS", {"mutant": tp53, "wildtype": len(list(maf_dir.glob("*.maf.gz"))) - tp53,
                                                   "no_maf": counts["no_maf"]})
    build.main(["tcga", "--status-json", str(root / "status.json")])
    d = root / C.DATASET_DIRS["tcga_luad_egfr"]
    meta = pd.read_csv(d / "metadata.csv")
    assert (meta.label == 1).sum() == counts["mutant"] and (meta.label == 0).sum() == counts["wildtype"]
    assert set(meta["class"]) == {"EGFR_mut", "EGFR_wt"}
    info = json.loads((d / "b7_build_info.json").read_text(encoding="utf-8"))
    assert info["gdc_status"]["data_release"].startswith("Data Release 46.0")
    filt = pd.read_csv(d / "b7_filter_distribution.csv")
    assert set(filt.FILTER) >= {"PASS"}

    genes = pd.read_hdf(root / C.DATASET_DIRS[C.SOURCE_DATASET] / "expression.h5").columns.tolist()
    _write_fake_gse31210(root, genes)
    monkeypatch.setattr(C, "B7_GSE31210_COUNTS", {"positive": 25, "negative": 25, "normal_excluded": 4})
    build.main(["gse31210"])
    g = root / C.DATASET_DIRS["gse31210_egfr"]
    gm = pd.read_csv(g / "metadata.csv")
    assert len(gm) == 50 and gm.label.sum() == 25 and set(gm.sex) == {"male", "female"}
    gi = json.loads((g / "b7_build_info.json").read_text(encoding="utf-8"))
    assert gi["scale_decision"]["scale"].startswith("linear") and gi["n_genes_in_tcga"] == 120

    sys.modules.pop("build_e1_repeated_splits", None)
    importlib.import_module("build_e1_repeated_splits").main(["--datasets", *C.B7_DATASETS])
    build.main(["card"])
    card = json.loads((g / "data_card_gse31210_egfr.json").read_text(encoding="utf-8"))
    assert card["n_per_class_after_qc"] == [25, 25] and "repeated_splits.csv" in card["sha256"]

    sys.modules.pop("run_survey_e0_signal_check", None)
    importlib.import_module("run_survey_e0_signal_check").main(["--datasets", *C.B7_DATASETS, "--n-perm", "8", "--n-random", "4",
                                                                "--prefilter", "100", "--out", "E0_b7"])
    sys.modules.pop("run_e1_classification", None)
    importlib.import_module("run_e1_classification").main(["--datasets", *C.B7_DATASETS, "--k", "10", "--random-draws", "3",
                                                           "--prefilter", "100", "--save-scores", "--out", "E1_classification_b7"])
    sys.modules.pop("b7_gate", None)
    importlib.import_module("b7_gate").main([])
    t = pd.read_csv(root / "results" / "survey_v2" / "B7_gate" / "b7_gate_table.csv")
    assert list(t.cohort) == list(C.B7_DATASETS) and t.n_folds.eq(25).all()
    assert t.b0_auprc_mean.between(0, 1).all() and t.positive_rate.between(0, 1).all()
    assert json.loads((root / "results" / "survey_v2" / "B7_gate" / "config.json").read_text(encoding="utf-8"))["experiment"] == "B7_gate"
    build.main(["pack"])
    packed = root / "results" / "survey_v2" / "B7_data" / C.B7_DATA_DIR
    for name in C.B7_DATASETS:
        for f in ("expression.h5", "metadata.csv", "repeated_splits.csv", f"data_card_{name}.json", "b7_labels_all.csv"):
            assert (packed / name / f).is_file(), (name, f)
    cfg = json.loads((root / "results" / "survey_v2" / "B7_data" / "config.json").read_text(encoding="utf-8"))
    assert cfg["experiment"] == "B7_data" and len(cfg["files_sha256"]) >= 10
