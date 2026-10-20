"""Command line: `fedretain <stage> --config configs/x.yaml [--override key=value ...]` plus legacy flags."""

from __future__ import annotations

import argparse
from typing import Sequence

from fedretain.config import (
    ARCHS,
    DCT_AXES,
    OPTIMIZERS,
    SUBSET_RULES,
    Config,
    Experiment,
    load_experiment,
    read_yaml,
)
from fedretain.figures import stage_datafig, stage_figures, stage_retention
from fedretain.model import ARCH_DEFAULTS
from fedretain.stages import (
    stage_arch,
    stage_band,
    stage_check,
    stage_divergence,
    stage_finetune,
    stage_localpred,
    stage_metrics,
    stage_reeval,
    stage_sweep,
    stage_tables,
)
from fedretain.stats import stage_bootstrap, stage_bootstrap_runs
from fedretain.train import stage_run

STAGES = (
    "check",
    "reeval",
    "bootstrap",
    "finetune",
    "run",
    "sweep",
    "tables",
    "metrics",
    "localpred",
    "bootstrap_runs",
    "band",
    "divergence",
    "figures",
    "datafig",
    "arch",
    "retention",
    "pipeline",
)

# legacy flags that set a Config field of the same name: (flag, type, nargs)
FIELD_FLAGS = (
    ("project", str, None),
    ("patches_dir", str, None),
    ("out_rel", str, None),
    ("folds", int, "+"),
    ("seeds", int, "+"),
    ("methods", str, "+"),
    ("seed", int, None),
    ("rounds", int, None),
    ("local_epochs", int, None),
    ("lr", float, None),
    ("weight_decay", float, None),
    ("batch_size", int, None),
    ("eval_batch_size", int, None),
    ("weighting", str, None),
    ("dct_axis", str, None),
    ("t_low", float, None),
    ("t_high", float, None),
    ("server_low", str, None),
    ("retain_layers", str, None),
    ("bn_source_heldout", str, None),
    ("bn_source_local", str, None),
    ("bn_calib_patches", int, None),
    ("warm_start", int, None),
    ("ft_epochs", int, None),
    ("local_only_epochs", int, None),
    ("norm_source", str, None),
    ("ft_source", str, None),
    ("train_cap", int, None),
    ("n_boot", int, None),
    ("boot_seed", int, None),
    ("ckpt", str, None),
    ("prox_mu", float, None),
    ("subset_rule", str, None),
    ("arch", str, None),
    ("optimizer", str, None),
    ("n_tsne", int, None),
    ("div_epochs", int, None),
)
CHOICES = {
    "dct_axis": DCT_AXES,
    "subset_rule": SUBSET_RULES,
    "arch": ARCHS,
    "optimizer": OPTIMIZERS,
    "weighting": ("proportional", "equal"),
    "server_low": ("freeze", "avg"),
    "retain_layers": ("all", "early", "late"),
    "norm_source": ("train_centers", "dataset_stats"),
    "ft_source": ("runs", "checkpoints"),
    "bn_source_local": ("client_train", "client_val"),
    "bn_source_heldout": ("checkpoint", "client_val", "test_center", "notebook"),
}


def build_parser() -> argparse.ArgumentParser:
    """The argument parser: new-style --config/--override and every legacy flag as an alias."""
    ap = argparse.ArgumentParser(prog="fedretain", description=__doc__)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", help="YAML config (configs/*.yaml)")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    for name, kind, nargs in FIELD_FLAGS:
        ap.add_argument(f"--{name}", type=kind, nargs=nargs, choices=CHOICES.get(name), default=None)
    ap.add_argument("--fold", type=int, default=0, help="run: fold to train")
    ap.add_argument("--method", default="fedavg", help="run: method to train")
    ap.add_argument("--redo", nargs="*", default=[], help="reeval: held-out BN sources to recompute")
    ap.add_argument(
        "--band", choices=("low", "high", "random"), default=None, help="legacy alias of subset_rule"
    )
    ap.add_argument("--raw", action="store_true", help="legacy: retain raw random weights (subset_rule raw)")
    ap.add_argument("--no_pretrained", action="store_true", help="arch: random init instead of ImageNet")
    ap.add_argument("--pointwise", action="store_true", help="arch: also decompose 1x1 convolutions")
    ap.add_argument("--source", default=None, help="divergence: run whose final model the clients start from")
    ap.add_argument("--reference_rel", default="results/fix", help="band: results dir of the reference runs")
    return ap


def legacy_values(args: argparse.Namespace) -> dict:
    """Config fields given through legacy flags on this command line."""
    values = {name: getattr(args, name) for name, _, _ in FIELD_FLAGS if getattr(args, name) is not None}
    if args.band is not None:
        values["subset_rule"] = args.band
    if args.raw:
        if values.get("subset_rule", "random") != "random":
            raise SystemExit("--raw is only supported together with --band random (subset_rule raw)")
        values["subset_rule"] = "raw"
    if args.no_pretrained:
        values["pretrained"] = False
    if args.pointwise:
        values["pointwise"] = True
    if args.source is not None:
        values["div_source"] = args.source
    return values


def apply_arch_defaults(cfg: Config, explicit: set) -> Config:
    """Backbone-specific rounds / epochs / lr / optimizer for every field the user did not set."""
    changes = {k: v for k, v in ARCH_DEFAULTS[cfg.arch].items() if k not in explicit}
    return cfg.updated(**changes) if changes else cfg


def parse(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, Experiment]:
    """Parse the command line into the stage arguments and the expanded experiment."""
    args = build_parser().parse_args(argv)
    experiment = load_experiment(args.config, args.override)
    values = legacy_values(args)
    explicit = set(values) | {text.split("=", 1)[0].strip() for text in args.override}
    if args.config:
        explicit |= set(read_yaml(args.config))
    configs = []
    for cfg in experiment.configs:
        cfg = cfg.updated(**values) if values else cfg
        if args.stage == "band" and "out_rel" not in explicit:
            cfg = cfg.updated(out_rel="results/band")
        if args.stage == "arch" and "out_rel" not in explicit:
            cfg = cfg.updated(out_rel=f"results/arch_{cfg.arch}")
        configs.append(apply_arch_defaults(cfg, explicit).validate())
    return args, Experiment(configs=configs, stages=experiment.stages)


def run_stage(stage: str, cfg: Config, args: argparse.Namespace, stages: Sequence[str] = ()) -> None:
    """Dispatch one stage on one Config."""
    seeds = list(cfg.seeds)
    if stage == "check":
        stage_check(cfg)
    elif stage == "reeval":
        stage_reeval(cfg.updated(norm_source="dataset_stats"), redo=tuple(args.redo))
    elif stage == "bootstrap":
        sources = (
            [args.bn_source_heldout] if args.bn_source_heldout else ["client_val", "test_center", "notebook"]
        )
        for src in sources:
            print(f"\n=== bootstrap ck={cfg.ckpt} bn={src} ===")
            stage_bootstrap(cfg, ck=cfg.ckpt, src=src, n_boot=cfg.n_boot, seed=cfg.boot_seed)
    elif stage == "finetune":
        stage_finetune(cfg, seeds=[cfg.seed] if args.seed is not None else seeds)
    elif stage == "run":
        stage_run(cfg, args.method, args.fold, cfg.seed)
    elif stage == "sweep":
        stage_sweep(cfg, list(cfg.methods), seeds)
    elif stage == "tables":
        stage_tables(cfg)
    elif stage == "metrics":
        stage_metrics(cfg)
    elif stage == "localpred":
        stage_localpred(cfg, seeds)
    elif stage == "bootstrap_runs":
        stage_bootstrap_runs(cfg, seeds, n_boot=cfg.n_boot, seed=cfg.boot_seed)
    elif stage == "band":
        stage_band(cfg, seeds, reference_rel=args.reference_rel)
    elif stage == "divergence":
        stage_divergence(cfg, seeds, cfg.div_source, cfg.div_epochs)
    elif stage == "figures":
        stage_figures(cfg, seeds)
    elif stage == "datafig":
        stage_datafig(cfg, seeds, cfg.n_tsne)
    elif stage == "retention":
        stage_retention(cfg)
    elif stage == "arch":
        stage_arch(cfg, list(cfg.methods), seeds)
    elif stage == "pipeline":
        if not stages:
            raise SystemExit("pipeline needs a `stages:` list in the config file")
        for inner in stages:
            print(f"\n##### {inner} #####")
            run_stage(inner, cfg, args)
    else:
        raise SystemExit(f"unknown stage {stage}")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point: run the stage on every variant of the config."""
    args, experiment = parse(argv)
    for i, cfg in enumerate(experiment.configs):
        if len(experiment.configs) > 1:
            print(f"\n===== variant {i + 1}/{len(experiment.configs)} =====")
        run_stage(args.stage, cfg, args, experiment.stages)


if __name__ == "__main__":
    main()
