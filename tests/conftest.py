"""Shared fixtures: the synthetic Camelyon17-like dataset and the archived legacy scripts."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEGACY_DIR = os.path.join(ROOT, "legacy")
sys.path.insert(0, HERE)

from synthetic import clone_project, make_fake_checkpoints, make_synthetic  # noqa: E402

from fedretain.config import Config  # noqa: E402

SMALL = dict(
    folds=(0,),
    rounds=2,
    local_epochs=1,
    batch_size=8,
    eval_batch_size=16,
    ft_epochs=1,
    dct_axis="flat",
    warm_start=1,
    seeds=(1,),
    seed=42,
    n_boot=20,
)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The synthetic dataset (5 centres x 48 patches) with fake submitted checkpoints for fold 0."""
    root = str(tmp_path_factory.mktemp("fr"))
    project, patches = make_synthetic(root)
    make_fake_checkpoints(project)
    return SimpleNamespace(root=root, project=project, patches=patches)


@pytest.fixture(scope="session")
def legacy() -> SimpleNamespace:
    """The archived scripts, imported from legacy/ (never edited)."""
    sys.path.insert(0, LEGACY_DIR)
    import fedfreq_extra as fx
    import fedfreq_fix as ff

    return SimpleNamespace(fix=ff, extra=fx)


def small_config(project: str, patches: str, **changes: object) -> Config:
    """A Config for a two-round CPU run on the synthetic data."""
    values = dict(SMALL, project=project, patches_dir=patches)
    values.update(changes)
    return Config(**values)


@pytest.fixture
def fresh_project(synthetic: SimpleNamespace, tmp_path_factory: pytest.TempPathFactory) -> str:
    """A new project directory with the synthetic metadata and stats but no results."""
    return clone_project(synthetic.project, str(tmp_path_factory.mktemp("proj")))
