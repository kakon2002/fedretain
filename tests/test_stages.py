"""Every stage runs end to end on the synthetic data through the command line, and resumes."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import yaml

from conftest import small_config
from synthetic import clone_project, make_fake_checkpoints

from fedretain.cli import main
from fedretain.config import Config
from fedretain.stages import stage_check
from fedretain.train import load_json, run_config, run_dirs, save_json


def base_flags(project: str, patches: str) -> list[str]:
    """Legacy-style flags for a two-round CPU run (the author's old notebook cells look like this)."""
    return [
        "--project",
        project,
        "--patches_dir",
        patches,
        "--folds",
        "0",
        "--seeds",
        "1",
        "--rounds",
        "2",
        "--local_epochs",
        "1",
        "--batch_size",
        "8",
        "--eval_batch_size",
        "16",
        "--dct_axis",
        "flat",
        "--warm_start",
        "1",
        "--n_boot",
        "20",
    ]


@pytest.fixture(scope="module")
def site(synthetic: SimpleNamespace, tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """A project with the synthetic data and fake checkpoints, shared by the stage tests in order."""
    project = clone_project(synthetic.project, str(tmp_path_factory.mktemp("st")))
    make_fake_checkpoints(project)
    return SimpleNamespace(
        project=project,
        patches=synthetic.patches,
        flags=base_flags(project, synthetic.patches),
        fix=os.path.join(project, "results", "fix"),
    )


def test_check(site: SimpleNamespace) -> None:
    cfg = small_config(site.project, site.patches)
    assert stage_check(cfg) is True


def test_reeval_and_bootstrap(site: SimpleNamespace) -> None:
    main(["reeval"] + site.flags)
    reeval = load_json(os.path.join(site.fix, "reeval.json"))
    assert reeval["fold0/fedavg"]["done"] and "best/client_val" in reeval["fold0/fedavg"]["heldout"]
    assert reeval["fold0/fedfreq"]["personalized"] is True
    main(["bootstrap", "--ckpt", "best", "--bn_source_heldout", "client_val"] + site.flags)
    boot = load_json(os.path.join(site.fix, "bootstrap_best_client_val.json"))
    assert "fedavg" in boot["pooled"] and boot["config"]["n_boot"] == 20


def test_run_and_resume(site: SimpleNamespace) -> None:
    main(["run", "--method", "fedavg", "--fold", "0", "--seed", "1"] + site.flags)
    run_dir = os.path.join(site.fix, "runs", "fedavg_fold0_seed1_flat")
    for name in (
        "result.json",
        "progress.json",
        "global_last.pt",
        "global_best_by_val.pt",
        "client_low.pt",
        "pred_heldout_last_test_center.npz",
        "pred_heldout_best_by_val_client_val.npz",
    ):
        assert os.path.exists(os.path.join(run_dir, name)), name
    result = load_json(os.path.join(run_dir, "result.json"))
    assert len(result["history"]) == 2 and "local_fedfreq_eval" in result
    # pretend the run was interrupted after round 1 and resume it
    progress = load_json(os.path.join(run_dir, "progress.json"))
    progress["completed_rounds"], progress["history"] = 1, progress["history"][:1]
    save_json(progress, os.path.join(run_dir, "progress.json"))
    os.remove(os.path.join(run_dir, "result.json"))
    main(["run", "--method", "fedavg", "--fold", "0", "--seed", "1"] + site.flags)
    resumed = load_json(os.path.join(run_dir, "progress.json"))
    assert (
        resumed["completed_rounds"] == 2
        and len(load_json(os.path.join(run_dir, "result.json"))["history"]) == 2
    )
    # finished runs are skipped
    main(["run", "--method", "fedavg", "--fold", "0", "--seed", "1"] + site.flags)


def test_sweep_all_methods(site: SimpleNamespace) -> None:
    main(
        ["sweep", "--methods", "fedprox", "fedbn", "fedper", "fedper4", "fedretain", "fedfreq_train"]
        + site.flags
    )
    runs = os.listdir(os.path.join(site.fix, "runs"))
    for tag in (
        "fedprox_fold0_seed1_flat",
        "fedper4_fold0_seed1_flat",
        "fedretain_fold0_seed1_ws1_flat",
        "fedfreq_train_fold0_seed1_ws1_flat",
    ):
        assert tag in runs, tag
    cfg = small_config(site.project, site.patches)
    names = run_dirs(cfg, 0, 1)
    assert {"fedavg", "fedprox", "fedbn", "fedper", "fedper4", "fedretain_ws1", "fedfreq_train_ws1"} <= set(
        names
    )


def test_finetune_localpred_bootstrap_tables_metrics(site: SimpleNamespace) -> None:
    main(["finetune"] + site.flags)
    assert "fold0" in load_json(os.path.join(site.fix, "finetune_e1_runs_seed1.json"))
    main(["localpred"] + site.flags)
    for name in (
        "fedretain_fold0_seed1_ws1_flat",
        "fedfreq_train_fold0_seed1_ws1_flat",
        "fedper4_fold0_seed1_flat",
    ):
        assert os.path.exists(os.path.join(site.fix, "runs", name, "pred_local_C1.npz")), name
    assert os.path.exists(os.path.join(site.fix, "runs", "fedavg_fold0_seed1_flat", "pred_local_ft1_C1.npz"))
    assert os.path.exists(os.path.join(site.fix, "runs", "fedavg_fold0_seed1_flat", "pred_local_fe_C1.npz"))
    main(["bootstrap_runs"] + site.flags)
    boot = load_json(os.path.join(site.fix, "bootstrap_runs.json"))["1"]
    assert {"fedavg_ft1", "fedavg_fedfreq_eval", "fedretain_ws1", "fedfreq_train_ws1", "fedper4"} <= set(
        boot["local"]
    )
    assert "diff_vs_fedavg_ft1" in boot["local"] and "diff_vs_fedavg" in boot["heldout"]
    main(["tables"] + site.flags)
    assert "runs" in load_json(os.path.join(site.fix, "tables.json"))
    main(["metrics"] + site.flags)
    with open(os.path.join(site.fix, "metrics.csv")) as handle:
        header = handle.readline()
    assert header.startswith("tag,method,fold,seed,split,client") and "ece" in header


def test_band_divergence_figures(site: SimpleNamespace) -> None:
    main(["band", "--band", "high"] + site.flags)
    band = os.path.join(site.project, "results", "band")
    assert os.path.exists(os.path.join(band, "runs", "fedretain_fold0_seed1_ws1_rbhigh_flat", "result.json"))
    boot = load_json(os.path.join(band, "bootstrap_runs.json"))["1"]
    assert {"fedavg", "fedretain_ws1", "fedretain_ws1_rbhigh"} <= set(boot["heldout"])
    main(["divergence"] + site.flags)
    div = load_json(os.path.join(site.fix, "divergence_fedavg.json"))
    assert div["fold0/seed1"]["bands"]["high"]["var_ratio_vs_high"] == 1.0
    main(["figures"] + site.flags)
    fig = os.path.join(site.fix, "figures")
    for name in (
        "accuracy_over_rounds.png",
        "heatmap_auc.png",
        "bands_communication.png",
        "divergence_fedavg.png",
    ):
        assert os.path.exists(os.path.join(fig, name)), name
    main(["retention"] + site.flags)
    assert os.path.exists(os.path.join(fig, "retention_fraction.png"))
    assert os.path.exists(os.path.join(fig, "band_ablation.png"))
    main(["datafig"] + site.flags)
    assert os.path.exists(os.path.join(fig, "stain_rgb.png")) and os.path.exists(
        os.path.join(fig, "tsne_fold0.png")
    )


def test_pipeline_with_variants(site: SimpleNamespace, tmp_path: pytest.TempPathFactory) -> None:
    config = {
        "project": site.project,
        "patches_dir": site.patches,
        "out_rel": "results/dev",
        "folds": [0],
        "seeds": [1],
        "rounds": 2,
        "local_epochs": 1,
        "batch_size": 8,
        "eval_batch_size": 16,
        "dct_axis": "flat",
        "n_boot": 20,
        "methods": ["fedavg", "fedretain"],
        "variants": [{"warm_start": 1}, {"warm_start": 2}],
        "stages": ["sweep", "localpred", "bootstrap_runs", "tables"],
    }
    path = os.path.join(str(tmp_path), "dev.yaml")
    with open(path, "w") as handle:
        yaml.safe_dump(config, handle)
    main(["pipeline", "--config", path, "--override", "lr=0.002"])
    dev = os.path.join(site.project, "results", "dev")
    for tag in (
        "fedavg_fold0_seed1_flat",
        "fedretain_fold0_seed1_ws1_flat",
        "fedretain_fold0_seed1_ws2_flat",
    ):
        assert os.path.exists(os.path.join(dev, "runs", tag, "result.json")), tag
    assert load_json(os.path.join(dev, "runs", tag, "result.json"))["config"]["lr"] == 0.002
    assert {"fedretain_ws1", "fedretain_ws2"} <= set(
        load_json(os.path.join(dev, "bootstrap_runs.json"))["1"]["local"]
    )


def test_arch(site: SimpleNamespace) -> None:
    main(
        ["arch", "--arch", "mobilenetv3", "--no_pretrained", "--methods", "fedavg", "fedretain"] + site.flags
    )
    out = os.path.join(site.project, "results", "arch_mobilenetv3")
    result = load_json(os.path.join(out, "runs", "fedretain_fold0_seed1_ws1_flat", "result.json"))
    assert (
        result["config"]["rounds"] == 2
        and result["config"]["optimizer"] == "sgd"
        and result["config"]["lr"] == 0.01
    )
    assert os.path.exists(os.path.join(out, "bootstrap_runs.json"))


def test_legacy_result_config_loads() -> None:
    """A legacy result config (no subset_rule / pointwise keys) loads with the subset taken from the tag."""
    stored = {"t_low": 0.1, "t_high": 0.7, "dct_axis": "cin", "retain_layers": "all", "unknown": 1}
    cfg = run_config(Config(), "fedfreq_train_ws15_tl0.1_rbrandom_raw", stored)
    assert (cfg.t_low, cfg.dct_axis, cfg.subset_rule, cfg.pointwise) == (0.1, "cin", "raw", False)
    cfg = run_config(Config(), "fedfreq_train_ws15_pw", {"t_low": 0.3, "t_high": 0.7, "dct_axis": "flat"})
    assert cfg.pointwise is True and cfg.subset_rule == "low"
    assert Config.from_dict(json.loads(json.dumps(Config().to_dict()))) == Config()
