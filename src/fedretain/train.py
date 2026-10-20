"""Local training, the federated round loop with per-round resumable checkpoints, run tags and run dirs."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from fedretain.adapt import autocast_ctx, evaluate
from fedretain.config import Config
from fedretain.data import FoldData, Loader, get_device, maybe_mount_drive
from fedretain.methods import Method, is_retain_method, make_method
from fedretain.model import build_model
from fedretain.retention import Retention, StateDict, conv_layer_names, select_layers

HISTORY_HELDOUT_KEYS = ("accuracy", "auc", "specificity", "sensitivity", "ece", "mce", "brier")


def set_seed(seed: int) -> None:
    """Seed torch (CPU and CUDA) and numpy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pt(path: str) -> dict | None:
    """Load a checkpoint to CPU, or None when the file does not exist."""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def load_json(path: str) -> dict:
    """Read a JSON file."""
    with open(path) as handle:
        return json.load(handle)


def save_json(obj: dict, path: str) -> None:
    """Write a JSON file with the legacy indentation."""
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=1)


def to_device(state: StateDict, device: torch.device) -> dict:
    """Copy of a state dict on `device`, ready for load_state_dict."""
    return {k: v.to(device) for k, v in state.items()}


def local_train(
    model: nn.Module,
    loader: Loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: Sequence[float],
    device: torch.device,
    prox_mu: float = 0.0,
    optimizer: str = "adamw",
) -> StateDict:
    """Train in place for `epochs` passes; prox_mu > 0 adds FedProx's (mu/2)||w - w_entry||^2 term."""
    model.train()
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    anchor = [p.detach().clone() for p in model.parameters()] if prox_mu > 0 else None
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            if bx.dim() == 4:
                bx = bx.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                loss = criterion(model(bx), by)
            if anchor is not None:
                proximal = sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
                loss = loss + 0.5 * prox_mu * proximal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_tag(cfg: Config, method: str, fold: int, seed: int) -> str:
    """Run directory name; the FedRetain settings are encoded exactly as the legacy scripts did."""
    tag = f"{method}_fold{fold}_seed{seed}"
    if is_retain_method(method):
        tag += f"_ws{cfg.warm_start}"
        if cfg.t_low != 0.3:
            tag += f"_tl{cfg.t_low}"
        if cfg.server_low != "freeze":
            tag += f"_sl{cfg.server_low}"
        if cfg.retain_layers != "all":
            tag += f"_rl{cfg.retain_layers}"
        if cfg.subset_rule in ("high", "random"):
            tag += f"_rb{cfg.subset_rule}"
        elif cfg.subset_rule == "raw":
            tag += "_rbrandom_raw"
        if cfg.pointwise:
            tag += "_pw"
    return tag + f"_{cfg.dct_axis}"


VARIANT_PATTERN = re.compile(r"(?:fedfreq_train|fedretain)_ws(\d+)(?:_tl([\d.]+))?(?:_rb([a-z]+))?(_raw)?$")


def parse_variant(name: str, t_low_default: float) -> tuple[int, float, str] | None:
    """'fedretain_ws15_tl0.1_rbrandom' -> (warm_start, t_low, subset label); None for other run names."""
    match = VARIANT_PATTERN.match(name)
    if not match:
        return None
    label = (match.group(3) or "low") + (" (raw weights)" if match.group(4) else "")
    return int(match.group(1)), float(match.group(2) or t_low_default), label


def subset_from_name(name: str) -> str:
    """Subset rule encoded in a run name or tag (legacy result files do not store it in their config)."""
    if "_raw" in name:
        return "raw"
    if "_rbhigh" in name:
        return "high"
    if "_rbrandom" in name:
        return "random"
    return "low"


def run_config(cfg: Config, name: str, stored: dict) -> Config:
    """cfg with the retention settings of a finished run, taken from its stored config or its name."""
    return cfg.updated(
        t_low=stored["t_low"],
        t_high=stored["t_high"],
        dct_axis=stored["dct_axis"],
        retain_layers=stored.get("retain_layers", "all"),
        subset_rule=stored.get("subset_rule", subset_from_name(name)),
        pointwise=stored.get("pointwise", name.endswith("_pw") or "_pw_" in name),
    )


def run_dirs(cfg: Config, fold: int, seed: int, reference_rel: str | None = None) -> dict[str, str]:
    """{short name: run dir} of every finished run of this fold/seed on cfg.dct_axis, in sorted tag order."""
    runs = os.path.join(cfg.out_dir, "runs")
    out: dict[str, str] = {}
    if os.path.isdir(runs):
        mid, end = f"_fold{fold}_seed{seed}", f"_{cfg.dct_axis}"
        for tag in sorted(os.listdir(runs)):
            if mid in tag and tag.endswith(end) and os.path.exists(os.path.join(runs, tag, "result.json")):
                out[tag.replace(mid, "")[: -len(end)]] = os.path.join(runs, tag)
    if reference_rel:
        wanted = ("fedavg", f"fedfreq_train_ws{cfg.warm_start}", f"fedretain_ws{cfg.warm_start}")
        for name, path in run_dirs(cfg.updated(out_rel=reference_rel), fold, seed).items():
            if name in wanted and name not in out:
                out[name] = path
    return out


def retained_conv_layers(state: StateDict, cfg: Config) -> list[str]:
    """Conv layers whose kernels take part in retention under this config."""
    return select_layers(conv_layer_names(state, cfg.pointwise), cfg.retain_layers)


def _round_evaluation(
    model: nn.Module, fd: FoldData, cfg: Config, device: torch.device, method: Method, global_state: StateDict
) -> tuple[dict, dict]:
    """Held-out metrics under the test-centre protocol and per-client own-centre metrics (capped BN pass)."""
    cap = cfg.bn_max_batches_round
    model.load_state_dict(to_device(global_state, device))
    heldout, _, _ = evaluate(
        model, fd.test_loader, device, fd.bn_heldout("test_center"), cfg, max_batches=cap
    )
    local = {}
    for c in fd.train_centers:
        model.load_state_dict(to_device(method.eval_state(c, global_state), device))
        bn = fd.bn_client(c, cfg.bn_source_local)
        metrics, _, _ = evaluate(model, fd.client_val_loader[c], device, bn, cfg, max_batches=cap)
        local[str(c)] = {"accuracy": metrics["accuracy"], "auc": metrics["auc"]}
    return heldout, local


def _final_result(
    model: nn.Module,
    fd: FoldData,
    cfg: Config,
    device: torch.device,
    method: Method,
    method_name: str,
    fold: int,
    seed: int,
    global_state: StateDict,
    history: list,
    best_sel: float,
    run_dir: str,
) -> dict:
    """Full held-out metrics for the last and val-selected checkpoints, plus the own-centre summaries."""
    final: dict = {
        "method": method_name,
        "fold": fold,
        "seed": seed,
        "config": cfg.to_dict(),
        "history": history,
    }
    best_state = load_pt(os.path.join(run_dir, "global_best_by_val.pt"))
    for name, state in (("last", global_state), ("best_by_val", best_state)):
        if state is None:
            continue
        for src in ("test_center", "client_val"):
            model.load_state_dict(to_device(state, device))
            metrics, probs, labels = evaluate(model, fd.test_loader, device, fd.bn_heldout(src), cfg)
            final[f"heldout_{name}_{src}"] = metrics
            if src == "test_center":
                final[f"heldout_{name}"] = metrics
            np.savez_compressed(
                os.path.join(run_dir, f"pred_heldout_{name}_{src}.npz"),
                probs=probs,
                labels=labels,
                patient=fd.test_ds.patient,
                node=fd.test_ds.node,
            )
    final["local_last"] = history[-1]["local"]
    final["local_mean_acc_last"] = history[-1]["local_mean_acc"]
    final["local_mean_acc_best"] = best_sel
    if method_name == "fedavg":
        personalised = {}
        for c in fd.train_centers:
            state = method.retention.personalize(global_state, method.client_retained[c], method.conv_layers)
            model.load_state_dict(to_device(state, device))
            bn = fd.bn_client(c, cfg.bn_source_local)
            metrics, _, _ = evaluate(model, fd.client_val_loader[c], device, bn, cfg)
            personalised[str(c)] = {"accuracy": metrics["accuracy"], "auc": metrics["auc"]}
        final["local_fedfreq_eval"] = personalised
        final["local_mean_acc_fedfreq_eval"] = float(np.mean([v["accuracy"] for v in personalised.values()]))
    return final


def stage_run(cfg: Config, method_name: str, fold: int, seed: int, fd: FoldData | None = None) -> dict:
    """Train one (method, fold, seed) run for cfg.rounds rounds, resuming from the last completed round."""
    maybe_mount_drive(cfg)
    device = get_device()
    tag = run_tag(cfg, method_name, fold, seed)
    run_dir = os.path.join(cfg.out_dir, "runs", tag)
    os.makedirs(run_dir, exist_ok=True)
    result_path = os.path.join(run_dir, "result.json")
    progress_path = os.path.join(run_dir, "progress.json")
    if os.path.exists(result_path):
        print(f"  skip {tag} (result.json exists)")
        return load_json(result_path)
    if fd is None:
        fd = FoldData(cfg, fold, device)
    clients = fd.train_centers
    set_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_model(cfg.arch, cfg.pretrained).to(device).to(memory_format=torch.channels_last)
    global_state: StateDict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    conv_layers = retained_conv_layers(global_state, cfg)
    method = make_method(method_name, cfg, Retention.from_config(cfg), conv_layers, model)
    history: list = []
    start = 0
    best_sel = -1.0
    if os.path.exists(progress_path):
        progress = load_json(progress_path)
        history, start, best_sel = progress["history"], progress["completed_rounds"], progress["best_sel"]
        global_state = load_pt(os.path.join(run_dir, "global_last.pt")) or global_state
        method.load(run_dir)
        print(f"  resuming {tag} at round {start}")
    for rnd in range(start, cfg.rounds):
        t0 = time.time()
        states, weights = [], []
        retaining = method.retaining(rnd)
        for c in clients:
            model.load_state_dict(to_device(method.client_start(c, rnd, global_state), device))
            states.append(
                local_train(
                    model,
                    fd.client_train_loader[c],
                    cfg.local_epochs,
                    cfg.lr,
                    cfg.weight_decay,
                    fd.class_weights,
                    device,
                    prox_mu=method.prox_mu,
                    optimizer=cfg.optimizer,
                )
            )
            weights.append(len(fd.client_train_ds[c]) if cfg.weighting == "proportional" else 1.0)
        for c, state in zip(clients, states):
            method.on_client_done(c, state)
        global_state = method.aggregate(global_state, states, weights, rnd)
        heldout, local = _round_evaluation(model, fd, cfg, device, method, global_state)
        sel = float(np.mean([v["accuracy"] for v in local.values()]))
        history.append(
            {
                "round": rnd + 1,
                "heldout": {k: heldout[k] for k in HISTORY_HELDOUT_KEYS},
                "local": local,
                "local_mean_acc": sel,
                "time_s": time.time() - t0,
                "retaining": retaining,
            }
        )
        improved = sel > best_sel
        if improved:
            best_sel = sel
            torch.save(global_state, os.path.join(run_dir, "global_best_by_val.pt"))
            method.save(run_dir, best=True)
        torch.save(global_state, os.path.join(run_dir, "global_last.pt"))
        method.save(run_dir)
        save_json(
            {
                "completed_rounds": rnd + 1,
                "history": history,
                "best_sel": best_sel,
                "config": cfg.to_dict(),
                "method": method_name,
                "fold": fold,
                "seed": seed,
            },
            progress_path,
        )
        per_client = " ".join(f"C{c}:{local[str(c)]['accuracy']:.2f}" for c in clients)
        print(
            f"  {tag} R{rnd + 1:2d}/{cfg.rounds} heldout acc={heldout['accuracy']:.4f} "
            f"auc={heldout['auc']:.4f} | local {sel:.4f} {per_client} | {time.time() - t0:.0f}s"
            + (" *" if improved else "")
            + (" [retain]" if retaining else ""),
            flush=True,
        )
    final = _final_result(
        model, fd, cfg, device, method, method_name, fold, seed, global_state, history, best_sel, run_dir
    )
    save_json(final, result_path)
    print(
        f"  done {tag}: heldout(last) acc={final['heldout_last']['accuracy']:.4f} "
        f"auc={final['heldout_last']['auc']:.4f} local(last)={final['local_mean_acc_last']:.4f}"
    )
    return final
