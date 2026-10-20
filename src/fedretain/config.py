"""Experiment configuration: legacy Config fields, the new method options, YAML files and overrides."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Sequence

import yaml

SUBSET_RULES = ("low", "high", "random", "raw")
ARCHS = ("histofreqnet", "mobilenetv3", "efficientnetb2")
OPTIMIZERS = ("adamw", "sgd")
DCT_AXES = ("flat", "cin")
TUPLE_FIELDS = {"folds": int, "seeds": int, "methods": str}
FILE_ONLY_KEYS = ("extends", "stages", "variants")


@dataclass
class Config:
    """All settings of one experiment; every legacy field keeps its legacy name and default."""

    project: str = "/content/drive/MyDrive/HistTTA"
    patches_dir: str = "/content/data/camelyon17/data/camelyon17_v1.0/patches"
    metadata_rel: str = "metadata/metadata_enhanced.csv"
    ckpt_rel: str = "checkpoints/camelyon17_histofreqnet"
    results_rel: str = "results"
    out_rel: str = "results/fix"

    folds: tuple = (0, 1, 2)
    t_low: float = 0.3
    t_high: float = 0.7
    dct_axis: str = "flat"

    rounds: int = 25
    local_epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    eval_batch_size: int = 512
    weighting: str = "proportional"
    seed: int = 42

    bn_source_heldout: str = "client_val"
    bn_source_local: str = "client_train"
    bn_max_batches: int = 0
    bn_calib_patches: int = 20000
    bn_max_batches_round: int = 100
    norm_source: str = "train_centers"
    ft_source: str = "runs"
    train_cap: int = 0
    n_bins: int = 10

    ft_epochs: int = 1
    local_only_epochs: int = 0

    warm_start: int = 0
    server_low: str = "freeze"
    retain_layers: str = "all"
    prox_mu: float = 0.01

    # options that the legacy helper script applied by monkeypatching
    subset_rule: str = "low"
    arch: str = "histofreqnet"
    pretrained: bool = True
    pointwise: bool = False
    optimizer: str = "adamw"

    # stage-level settings that were command-line only in the legacy scripts
    seeds: tuple = (42, 43)
    methods: tuple = ("fedretain", "fedavg")
    n_boot: int = 1000
    boot_seed: int = 0
    ckpt: str = "best"
    div_source: str = "fedavg"
    div_epochs: int = 1
    n_tsne: int = 600

    @property
    def metadata(self) -> str:
        """Path of the patch metadata CSV."""
        return os.path.join(self.project, self.metadata_rel)

    @property
    def ckpt_base(self) -> str:
        """Directory holding the original submitted checkpoints."""
        return os.path.join(self.project, self.ckpt_rel)

    @property
    def results_dir(self) -> str:
        """Top-level results directory (holds dataset_stats.json)."""
        return os.path.join(self.project, self.results_rel)

    @property
    def out_dir(self) -> str:
        """Output directory of the current experiment; created on first access like the legacy code."""
        path = os.path.join(self.project, self.out_rel)
        os.makedirs(path, exist_ok=True)
        return path

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON (tuples become lists exactly as the legacy asdict did)."""
        return asdict(self)

    def validate(self) -> "Config":
        """Raise ValueError on an impossible combination of settings."""
        checks = (
            ("dct_axis", DCT_AXES),
            ("subset_rule", SUBSET_RULES),
            ("arch", ARCHS),
            ("optimizer", OPTIMIZERS),
            ("weighting", ("proportional", "equal")),
            ("server_low", ("freeze", "avg")),
            ("retain_layers", ("all", "early", "late")),
            ("norm_source", ("train_centers", "dataset_stats")),
            ("ft_source", ("runs", "checkpoints")),
            ("bn_source_heldout", ("checkpoint", "client_val", "test_center", "notebook")),
            ("bn_source_local", ("client_train", "client_val")),
        )
        for name, allowed in checks:
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r} is not one of {allowed}")
        if not 0.0 <= self.t_low <= self.t_high <= 1.0:
            raise ValueError(f"need 0 <= t_low <= t_high <= 1, got {self.t_low}, {self.t_high}")
        return self

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Config":
        """Build a Config from a dict, ignoring unknown keys (legacy result configs lack the new fields)."""
        known = {f.name for f in fields(cls)}
        kept = {k: coerce(k, v) for k, v in values.items() if k in known}
        return cls(**kept)

    def updated(self, **changes: Any) -> "Config":
        """Copy with the given fields replaced (values are coerced to the field types)."""
        return replace(self, **{k: coerce(k, v) for k, v in changes.items()})


FIELD_TYPES = {f.name: f.type for f in fields(Config)}


def coerce(name: str, value: Any) -> Any:
    """Convert a value from YAML or the command line to the type of Config field `name`."""
    if name not in FIELD_TYPES:
        raise KeyError(f"unknown config field {name!r}")
    if name in TUPLE_FIELDS:
        element = TUPLE_FIELDS[name]
        if isinstance(value, str):
            value = [v for v in value.replace(",", " ").split() if v]
        if not isinstance(value, (list, tuple)):
            value = [value]
        return tuple(element(v) for v in value)
    kind = FIELD_TYPES[name]
    if kind in ("bool", bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind in ("int", int):
        return int(value)
    if kind in ("float", float):
        return float(value)
    return str(value)


def parse_override(text: str) -> tuple[str, Any]:
    """Split a `key=value` command-line override into a field name and its typed value."""
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like key=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    return key, coerce(key, raw.strip())


def read_yaml(path: str) -> dict[str, Any]:
    """Read a YAML config file and merge in the file named by its `extends` key (relative to the file)."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    merged = read_yaml(os.path.join(os.path.dirname(os.path.abspath(path)), parent))
    merged.update(raw)
    return merged


@dataclass
class Experiment:
    """A config file after expansion: one Config per variant, plus the stage list for `pipeline`."""

    configs: list
    stages: list


def load_experiment(path: str | None, overrides: Sequence[str] = ()) -> Experiment:
    """Configs described by a YAML file (or the defaults) with `key=value` overrides applied last."""
    raw = read_yaml(path) if path else {}
    stages = list(raw.get("stages", []) or [])
    variants = list(raw.get("variants", []) or [{}])
    base_values = {k: v for k, v in raw.items() if k not in FILE_ONLY_KEYS}
    override_values = dict(parse_override(text) for text in overrides)
    configs = []
    for variant in variants:
        values = dict(base_values)
        values.update(variant or {})
        values.update(override_values)
        configs.append(Config.from_dict(values).validate())
    return Experiment(configs=configs, stages=stages)
