"""Synthetic Camelyon17-like data for the tests: the generator of legacy/tests/test_fix.py, unchanged."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fedretain.data import patch_path
from fedretain.model import HistoFreqNet
from fedretain.retention import Retention, conv_layer_names

FAKE_METHODS = ("fedfreq", "fedavg", "fedprox", "fedbn", "freqfed")


def make_synthetic(root: str) -> tuple[str, str]:
    """Build 5 centres x 4 patients x 2 nodes x 6 patches under root; returns (project dir, patches dir)."""
    shutil.rmtree(root, ignore_errors=True)
    project = os.path.join(root, "HistTTA")
    patches = os.path.join(root, "patches")
    os.makedirs(os.path.join(project, "metadata"))
    os.makedirs(os.path.join(project, "results"))
    rng = np.random.RandomState(0)
    rows = []
    for c in range(5):
        for p in range(4):
            pid = c * 10 + p
            for node in range(2):
                for k in range(6):
                    rows.append(
                        dict(
                            center=c,
                            patient=pid,
                            node=node,
                            x_coord=k * 100,
                            y_coord=node * 100,
                            tumor=int(rng.rand() < 0.5),
                            split=int(k >= 4),
                            folder=f"patient_{pid:03d}_node_{node}",
                        )
                    )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(project, "metadata", "metadata_enhanced.csv"), index=False)
    for _, row in df.iterrows():
        os.makedirs(os.path.join(patches, row["folder"]), exist_ok=True)
        base = np.array([150 + 20 * row["center"], 100, 180 - 15 * row["center"]], dtype=np.uint8)
        img = np.clip(base + rng.randint(-30, 30, (96, 96, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(patch_path(row, patches))
    stats = {"mean": [0.6, 0.4, 0.6], "std": [0.2, 0.2, 0.2], "class_weights": [1.0, 1.0]}
    with open(os.path.join(project, "results", "dataset_stats.json"), "w") as handle:
        json.dump(stats, handle)
    return project, patches


def clone_project(source: str, target: str) -> str:
    """Copy the metadata and dataset stats of a project to a fresh project directory (results start empty)."""
    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(os.path.join(target, "results"))
    shutil.copytree(os.path.join(source, "metadata"), os.path.join(target, "metadata"))
    shutil.copy(
        os.path.join(source, "results", "dataset_stats.json"),
        os.path.join(target, "results", "dataset_stats.json"),
    )
    return target


def make_fake_checkpoints(
    project: str, ckpt_rel: str = "checkpoints/camelyon17_histofreqnet", fold: int = 0
) -> str:
    """Write the fold's submitted-checkpoint files from one seeded model (for check/reeval/bootstrap)."""
    fold_dir = os.path.join(project, ckpt_rel, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.manual_seed(1)
    state = {k: v.clone() for k, v in HistoFreqNet().state_dict().items()}
    for method in FAKE_METHODS:
        torch.save(state, os.path.join(fold_dir, f"{method}_best.pt"))
        torch.save(state, os.path.join(fold_dir, f"{method}_global.pt"))
    retention = Retention(0.3, 0.7, "flat", "low")
    convs = conv_layer_names(state)
    low = {c: {k: retention.decompose(state[k] * (1 + 0.1 * c))[0] for k in convs} for c in (1, 2, 3, 4)}
    torch.save(low, os.path.join(fold_dir, "fedfreq_low_freq.pt"))
    return fold_dir
