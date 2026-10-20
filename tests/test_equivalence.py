"""The legacy scripts and the package must produce identical runs, predictions and bootstrap files."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import SMALL, small_config
from synthetic import clone_project

from fedretain.stages import stage_band, stage_finetune, stage_localpred
from fedretain.stats import stage_bootstrap_runs
from fedretain.train import run_tag, stage_run

METHODS = ("fedavg", "fedbn", "fedper", "fedfreq_train")
CHECKPOINTS = (
    "global_last.pt",
    "global_best_by_val.pt",
    "client_low.pt",
    "client_low_best_by_val.pt",
    "server_low.pt",
    "client_bn.pt",
)


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def without_time(history: list) -> list:
    """History entries minus time_s, the only wall-clock field."""
    return [{k: v for k, v in entry.items() if k != "time_s"} for entry in history]


def same_tensors(a: object, b: object) -> bool:
    """torch.equal over nested tensor dicts; key order is ignored (the legacy fedbn keys come from a set)."""
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same_tensors(a[k], b[k]) for k in a)
    return a == b


def assert_run_dirs_identical(legacy_dir: str, new_dir: str) -> None:
    """Same files, same history, same result keys and values, same checkpoint tensors, same predictions."""
    assert sorted(os.listdir(legacy_dir)) == sorted(os.listdir(new_dir))
    legacy_result = load_json(os.path.join(legacy_dir, "result.json"))
    new_result = load_json(os.path.join(new_dir, "result.json"))
    assert list(legacy_result) == list(new_result)
    assert without_time(legacy_result["history"]) == without_time(new_result["history"])
    for key in legacy_result:
        if key not in ("config", "history", "method"):
            assert legacy_result[key] == new_result[key], key
    for name in CHECKPOINTS:
        legacy_path, new_path = os.path.join(legacy_dir, name), os.path.join(new_dir, name)
        assert os.path.exists(legacy_path) == os.path.exists(new_path), name
        if os.path.exists(legacy_path):
            assert same_tensors(
                torch.load(legacy_path, weights_only=True), torch.load(new_path, weights_only=True)
            ), name
    for name in sorted(os.listdir(legacy_dir)):
        if name.endswith(".npz"):
            a, b = np.load(os.path.join(legacy_dir, name)), np.load(os.path.join(new_dir, name))
            assert sorted(a.files) == sorted(b.files), name
            assert all(np.array_equal(a[k], b[k]) for k in a.files), name


def assert_runs_identical(legacy_out: str, new_out: str) -> None:
    """Every run directory under two results folders is identical."""
    legacy_runs, new_runs = os.path.join(legacy_out, "runs"), os.path.join(new_out, "runs")
    assert sorted(os.listdir(legacy_runs)) == sorted(os.listdir(new_runs))
    for tag in sorted(os.listdir(legacy_runs)):
        assert_run_dirs_identical(os.path.join(legacy_runs, tag), os.path.join(new_runs, tag))


@pytest.fixture(scope="module")
def trained(
    synthetic: SimpleNamespace, legacy: SimpleNamespace, tmp_path_factory: pytest.TempPathFactory
) -> SimpleNamespace:
    """fedavg, fedbn, fedper and fedfreq_train (2 rounds, warm_start 1) trained by both code bases."""
    legacy_project = clone_project(synthetic.project, str(tmp_path_factory.mktemp("old")))
    new_project = clone_project(synthetic.project, str(tmp_path_factory.mktemp("new")))
    fields = {k: v for k, v in SMALL.items() if k not in ("seeds", "n_boot")}
    legacy_cfg = legacy.fix.Config(project=legacy_project, patches_dir=synthetic.patches, **fields)
    new_cfg = small_config(new_project, synthetic.patches)
    for method in METHODS:
        legacy.fix.stage_run(legacy_cfg, method, 0, 1)
        stage_run(new_cfg, method, 0, 1)
    return SimpleNamespace(
        legacy_cfg=legacy_cfg,
        new_cfg=new_cfg,
        legacy_project=legacy_project,
        new_project=new_project,
        patches=synthetic.patches,
    )


@pytest.mark.parametrize("method", METHODS)
def test_run_identical(trained: SimpleNamespace, legacy: SimpleNamespace, method: str) -> None:
    """History and saved tensors of each method match bit for bit (torch.equal, not allclose)."""
    tag = legacy.fix.run_tag(trained.legacy_cfg, method, 0, 1)
    assert run_tag(trained.new_cfg, method, 0, 1) == tag
    assert_run_dirs_identical(
        os.path.join(trained.legacy_cfg.out_dir, "runs", tag),
        os.path.join(trained.new_cfg.out_dir, "runs", tag),
    )


def test_fedretain_name_identical(trained: SimpleNamespace, legacy: SimpleNamespace) -> None:
    """The same run under the new method name gives the same history and tensors (only the tag changes)."""
    cfg = trained.new_cfg.updated(out_rel="results/newname")
    stage_run(cfg, "fedretain", 0, 1)
    legacy_tag = legacy.fix.run_tag(trained.legacy_cfg, "fedfreq_train", 0, 1)
    new_tag = run_tag(cfg, "fedretain", 0, 1)
    assert new_tag == legacy_tag.replace("fedfreq_train", "fedretain")
    legacy_dir = os.path.join(trained.legacy_cfg.out_dir, "runs", legacy_tag)
    new_dir = os.path.join(cfg.out_dir, "runs", new_tag)
    assert_run_dirs_identical(legacy_dir, new_dir)


@pytest.fixture(scope="module")
def scored(trained: SimpleNamespace, legacy: SimpleNamespace) -> SimpleNamespace:
    """Per-client predictions and the paired bootstrap (n_boot 20, bootstrap seed 0) from both code bases."""
    legacy.fix.stage_localpred(trained.legacy_cfg, [1])
    legacy.fix.stage_bootstrap_runs(trained.legacy_cfg, [1], n_boot=20, seed=0)
    stage_localpred(trained.new_cfg, [1])
    stage_bootstrap_runs(trained.new_cfg, [1], n_boot=20, seed=0)
    return trained


def test_localpred_identical(scored: SimpleNamespace) -> None:
    """pred_local_*.npz (personalised, fine-tuned and swap-in variants) are identical for every run."""
    assert_runs_identical(scored.legacy_cfg.out_dir, scored.new_cfg.out_dir)


def test_bootstrap_runs_identical(scored: SimpleNamespace) -> None:
    """bootstrap_runs.json is identical for the same n_boot and bootstrap seed."""
    legacy_boot = load_json(os.path.join(scored.legacy_cfg.out_dir, "bootstrap_runs.json"))
    new_boot = load_json(os.path.join(scored.new_cfg.out_dir, "bootstrap_runs.json"))
    assert legacy_boot == new_boot
    assert {"fedavg", "fedbn", "fedper", "fedfreq_train_ws1", "fedavg_ft1", "fedavg_fedfreq_eval"} <= set(
        new_boot["1"]["local"]
    )


def test_finetune_identical(scored: SimpleNamespace, legacy: SimpleNamespace) -> None:
    """The FedAvg + local fine-tuning stage gives identical metrics."""
    legacy_cfg = legacy.fix.Config(**{**legacy.fix.asdict(scored.legacy_cfg), "seed": 1})
    legacy.fix.stage_finetune(legacy_cfg)
    stage_finetune(scored.new_cfg, seeds=[1])
    name = "finetune_e1_runs_seed1.json"
    legacy_ft = load_json(os.path.join(scored.legacy_cfg.out_dir, name))
    new_ft = load_json(os.path.join(scored.new_cfg.out_dir, name))
    for fold in legacy_ft:
        assert {k: v for k, v in legacy_ft[fold].items() if k != "config"} == {
            k: v for k, v in new_ft[fold].items() if k != "config"
        }


def test_band_random_raw_identical(scored: SimpleNamespace, legacy: SimpleNamespace) -> None:
    """`band --band random --raw` (train, score, merged bootstrap) matches the subset_rule=raw band stage."""
    args = [
        "band",
        "--band",
        "random",
        "--raw",
        "--project",
        scored.legacy_project,
        "--patches_dir",
        scored.patches,
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
        "--dct_axis",
        "flat",
        "--warm_start",
        "1",
    ]
    legacy.extra.main(args)
    # the legacy main has no --eval_batch_size flag (so 512) and its band stage always
    # bootstraps with 1000 resamples
    cfg = scored.new_cfg.updated(
        out_rel="results/band",
        subset_rule="raw",
        methods=("fedfreq_train",),
        eval_batch_size=512,
        n_boot=1000,
    )
    stage_band(cfg, [1], reference_rel="results/fix")
    legacy_out = os.path.join(scored.legacy_project, "results", "band")
    new_out = os.path.join(scored.new_project, "results", "band")
    assert_runs_identical(legacy_out, new_out)
    assert os.path.isdir(os.path.join(new_out, "runs", "fedfreq_train_fold0_seed1_ws1_rbrandom_raw_flat"))
    legacy_boot = load_json(os.path.join(legacy_out, "bootstrap_runs.json"))
    new_boot = load_json(os.path.join(new_out, "bootstrap_runs.json"))
    assert legacy_boot == new_boot
    assert {"fedavg", "fedfreq_train_ws1", "fedfreq_train_ws1_rbrandom_raw"} <= set(new_boot["1"]["heldout"])
