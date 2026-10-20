"""Stages: check, reeval, finetune, sweep, localpred, divergence, tables, metrics dump, band and arch."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch

from fedretain.adapt import evaluate
from fedretain.config import Config
from fedretain.data import FoldData, get_device, load_metadata, load_stats, maybe_mount_drive, patch_path
from fedretain.methods import PERSONAL_KEY_METHODS, is_retain_method, is_retain_run, with_personal
from fedretain.model import build_model
from fedretain.retention import Retention, band_masks, conv_layer_names, retained_count, upload_size_mb
from fedretain.stats import stage_bootstrap_runs
from fedretain.train import (
    load_json,
    load_pt,
    local_train,
    retained_conv_layers,
    run_config,
    run_dirs,
    run_tag,
    save_json,
    set_seed,
    stage_run,
    to_device,
)

SUBMITTED_METHODS = ("fedfreq", "fedavg", "fedprox", "fedbn", "freqfed")
REEVAL_PROTOCOL = 3
HELDOUT_SOURCES = ("checkpoint", "client_val", "test_center", "notebook")
TABLE_METRICS = ("accuracy", "auc", "specificity", "sensitivity", "brier", "ece", "mce", "wga")


def _free_cuda() -> None:
    """Release cached GPU memory between folds."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_pred(path: str, fd: FoldData, c: int, probs: np.ndarray, labels: np.ndarray) -> None:
    """Write a client's validation predictions with the metadata the bootstrap needs."""
    ds = fd.client_val_ds[c]
    np.savez_compressed(path, probs=probs, labels=labels, patient=ds.patient, node=ds.node, center=ds.center)


# ----------------------------------------------------------------------------------------------------------
#  check
# ----------------------------------------------------------------------------------------------------------


def stage_check(cfg: Config) -> bool:
    """Report what is on disk (patches, metadata, stats, submitted checkpoints) and whether it is usable."""
    maybe_mount_drive(cfg)
    print("=" * 70 + "\nCHECK\n" + "=" * 70)
    print(f"  project:  {cfg.project}  exists={os.path.isdir(cfg.project)}")
    ok = os.path.isdir(cfg.patches_dir)
    print(
        f"  patches:  {cfg.patches_dir}  exists={ok}"
        + ("" if ok else "   <- extract the patch archive first")
    )
    try:
        df = load_metadata(cfg)
        print(f"  metadata: {len(df):,} rows; columns ok (patient, node present)")
        for c in range(5):
            d = df[df.center == c]
            print(
                f"    C{c}: train={int((d.split == 0).sum()):,} val={int((d.split == 1).sum()):,} "
                f"patients={d.patient.nunique()} nodes={d.groupby(['patient', 'node']).ngroups} "
                f"tumor={d.tumor.mean():.3f}"
            )
        sample = patch_path(df.iloc[0], cfg.patches_dir)
        print(f"  sample patch exists: {os.path.exists(sample)}  ({sample})")
    except Exception as exc:
        ok = False
        print(f"  metadata problem: {exc}")
    try:
        mean, _, class_weights = load_stats(cfg)
        print(f"  dataset_stats.json: mean={np.round(mean, 3).tolist()} class_weights={class_weights}")
    except Exception as exc:
        ok = False
        print(f"  dataset_stats.json problem: {exc}")
    print("\n  checkpoints:")
    for fold in cfg.folds:
        d = os.path.join(cfg.ckpt_base, f"fold_{fold}")
        if not os.path.isdir(d):
            print(f"    fold_{fold}: MISSING")
            ok = False
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
        sizes = ", ".join(f"{f} ({os.path.getsize(os.path.join(d, f)) / 2 ** 20:.0f}MB)" for f in files)
        print(f"    fold_{fold}: {sizes}")
        for m in SUBMITTED_METHODS:
            for suffix in ("_best.pt", "_global.pt"):
                if m + suffix not in files:
                    print(f"      missing {m}{suffix}")
        if "fedfreq_low_freq.pt" not in files:
            print("      missing fedfreq_low_freq.pt (needed for personalized eval)")
    ok = _check_checkpoint_model(cfg) and ok
    print(f"\n  out dir: {cfg.out_dir}")
    print("\n  RESULT:", "ready" if ok else "fix the items above first")
    return ok


def _check_checkpoint_model(cfg: Config) -> bool:
    """Load one submitted checkpoint into the model and print the band sizes for both DCT axes."""
    try:
        path = os.path.join(cfg.ckpt_base, f"fold_{cfg.folds[0]}", "fedavg_best.pt")
        if not os.path.exists(path):
            return True
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(cfg.arch, cfg.pretrained)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"\n  checkpoint/model key match: missing={len(missing)} unexpected={len(unexpected)}")
        ok = not (missing or unexpected)
        if not ok:
            print("   ", missing[:5], unexpected[:5])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
        convs = conv_layer_names(state, cfg.pointwise)
        for axis in ("flat", "cin"):
            retention = Retention(cfg.t_low, cfg.t_high, axis, cfg.subset_rule)
            kept = retained_count(state, convs, retention)
            total = sum(state[k].numel() for k in convs)
            print(
                f"  dct_axis={axis}: {len(convs)} conv layers, retained coeffs={kept:,} of conv params "
                f"{total:,} ({kept / total * 100:.1f}%), of all params {kept / n_params * 100:.1f}%"
            )
        return ok
    except Exception as exc:
        print(f"  checkpoint load problem: {exc}")
        return False


# ----------------------------------------------------------------------------------------------------------
#  reeval of the submitted checkpoints (no training)
# ----------------------------------------------------------------------------------------------------------


def _reeval_one(
    cfg: Config,
    fd: FoldData,
    device: torch.device,
    fold: int,
    method: str,
    prev: dict | None,
    heldout_sources: Sequence[str],
    redo: Sequence[str],
) -> dict | None:
    """Re-score one submitted checkpoint pair (best/last) under every held-out BN source and per client."""
    d = os.path.join(cfg.ckpt_base, f"fold_{fold}")
    best = load_pt(os.path.join(d, f"{method}_best.pt"))
    last = load_pt(os.path.join(d, f"{method}_global.pt"))
    key = f"fold{fold}/{method}"
    if best is None and last is None:
        print(f"  {key}: no checkpoint")
        return None
    todo = [
        (ck, src)
        for ck in ("best", "last")
        for src in heldout_sources
        if prev is None or src in redo or f"{ck}/{src}" not in prev["heldout"]
    ]
    todo = [(ck, src) for ck, src in todo if (best if ck == "best" else last) is not None]
    if prev is not None and not todo:
        print(f"  skip {key}")
        return prev
    model = build_model(cfg.arch, cfg.pretrained).to(device)
    res = prev if prev is not None else {"heldout": {}, "local": {}}
    retention = Retention.from_config(cfg)
    convs = conv_layer_names(best if best is not None else last, cfg.pointwise)
    for ck_name, ck in (("best", best), ("last", last)):
        if ck is None:
            continue
        for src in heldout_sources:
            if (ck_name, src) not in todo:
                continue
            model.load_state_dict(to_device(ck, device))
            metrics, probs, labels = evaluate(model, fd.test_loader, device, fd.bn_heldout(src), cfg)
            res["heldout"][f"{ck_name}/{src}"] = metrics
            np.savez_compressed(
                os.path.join(cfg.out_dir, f"pred_fold{fold}_{method}_{ck_name}_{src}.npz"),
                probs=probs,
                labels=labels,
                patient=fd.test_ds.patient,
                node=fd.test_ds.node,
                center=fd.test_ds.center,
            )
            print(
                f"  {key} {ck_name:4s} bn={src:11s} acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f} "
                f"spec={metrics['specificity']:.4f} ece={metrics['ece']:.4f} mce={metrics['mce']:.4f}",
                flush=True,
            )
    base = last if last is not None else best
    low, low_file = None, None
    if method.startswith("fedfreq"):
        for candidate in (f"{method}_low_freq.pt", "fedfreq_low_freq.pt"):
            if os.path.exists(os.path.join(d, candidate)):
                low, low_file = load_pt(os.path.join(d, candidate)), candidate
                break
    res["low_band_file"] = low_file
    for c in fd.train_centers:
        if prev is not None and str(c) in prev["local"]:
            continue
        state = base
        if low is not None and c in low:
            state = retention.personalize(base, low[c], convs)
        model.load_state_dict(to_device(state, device))
        metrics, probs, labels = evaluate(
            model, fd.client_val_loader[c], device, fd.bn_client(c, cfg.bn_source_local), cfg
        )
        res["local"][str(c)] = metrics
        np.savez_compressed(
            os.path.join(cfg.out_dir, f"pred_fold{fold}_{method}_local_C{c}.npz"),
            probs=probs,
            labels=labels,
            patient=fd.client_val_ds[c].patient,
            node=fd.client_val_ds[c].node,
            center=fd.client_val_ds[c].center,
        )
    res["local_mean_acc"] = float(np.mean([v["accuracy"] for v in res["local"].values()]))
    res["personalized"] = low is not None
    res["done"] = True
    res["protocol"] = REEVAL_PROTOCOL
    res["config"] = cfg.to_dict()
    print(f"  {key} local mean acc={res['local_mean_acc']:.4f} personalized={res['personalized']}")
    return res


def stage_reeval(
    cfg: Config,
    methods: Sequence[str] = SUBMITTED_METHODS,
    heldout_sources: Sequence[str] = HELDOUT_SOURCES,
    redo: Sequence[str] = (),
) -> dict:
    """Re-score the submitted checkpoints under the corrected BN protocols; resumable per (fold, method)."""
    maybe_mount_drive(cfg)
    device = get_device()
    out_path = os.path.join(cfg.out_dir, "reeval.json")
    results = load_json(out_path) if os.path.exists(out_path) else {}
    results = {k: v for k, v in results.items() if v.get("protocol") == REEVAL_PROTOCOL}
    for fold in cfg.folds:
        need_train = "calib" if cfg.bn_source_local == "client_train" else False
        fd = FoldData(cfg, fold, device, need_train=need_train)
        d = os.path.join(cfg.ckpt_base, f"fold_{fold}")
        extra = [m for m in ("fedfreq_v3", "centralized") if os.path.exists(os.path.join(d, f"{m}_best.pt"))]
        for method in list(methods) + extra:
            key = f"fold{fold}/{method}"
            prev = results.get(key) if results.get(key, {}).get("done") else None
            res = _reeval_one(cfg, fd, device, fold, method, prev, heldout_sources, redo)
            if res is None or res is prev:
                continue
            results[key] = res
            save_json(results, out_path)
        del fd
        _free_cuda()
    print(f"\n  saved {out_path}")
    return results


# ----------------------------------------------------------------------------------------------------------
#  finetune baseline (FedAvg + local fine-tuning; optional local-only)
# ----------------------------------------------------------------------------------------------------------


def stage_finetune(cfg: Config, seeds: Sequence[int] | None = None) -> dict:
    """FedAvg weights fine-tuned ft_epochs on each client's training split, scored on its validation split."""
    results = {}
    for seed in (seeds if seeds is not None else (cfg.seed,)):
        results.update(_finetune_seed(cfg.updated(seed=seed)))
    return results


def _finetune_seed(cfg: Config) -> dict:
    """The finetune stage for cfg.seed; writes finetune_e{E}_runs_seed{S}.json (or finetune_e{E}.json)."""
    maybe_mount_drive(cfg)
    device = get_device()
    tag = f"finetune_e{cfg.ft_epochs}" + ("" if cfg.ft_source == "checkpoints" else f"_runs_seed{cfg.seed}")
    out_path = os.path.join(cfg.out_dir, f"{tag}.json")
    results = load_json(out_path) if os.path.exists(out_path) else {}
    for fold in cfg.folds:
        key = f"fold{fold}"
        if key in results and results[key].get("done"):
            print(f"  skip {key}")
            continue
        if cfg.ft_source == "checkpoints":
            d = os.path.join(cfg.ckpt_base, f"fold_{fold}")
            base = load_pt(os.path.join(d, "fedavg_global.pt")) or load_pt(os.path.join(d, "fedavg_best.pt"))
        else:
            run_dir = os.path.join(cfg.out_dir, "runs", run_tag(cfg, "fedavg", fold, cfg.seed))
            base = load_pt(os.path.join(run_dir, "global_last.pt"))
        if base is None:
            print(f"  {key}: no fedavg model for ft_source={cfg.ft_source} seed={cfg.seed}")
            continue
        fd = FoldData(cfg, fold, device)
        model = build_model(cfg.arch, cfg.pretrained).to(device)
        res: dict = {"fedavg_finetune": {}, "local_only": {}}
        for c in fd.train_centers:
            set_seed(cfg.seed)
            model.load_state_dict(to_device(base, device))
            local_train(
                model,
                fd.client_train_loader[c],
                cfg.ft_epochs,
                cfg.lr,
                cfg.weight_decay,
                fd.class_weights,
                device,
                optimizer=cfg.optimizer,
            )
            bn = fd.bn_client(c, cfg.bn_source_local)
            metrics, _, _ = evaluate(model, fd.client_val_loader[c], device, bn, cfg)
            res["fedavg_finetune"][str(c)] = metrics
            print(
                f"  fold {fold} C{c} fedavg+ft({cfg.ft_epochs}ep): acc={metrics['accuracy']:.4f} "
                f"auc={metrics['auc']:.4f}",
                flush=True,
            )
            if cfg.local_only_epochs > 0:
                set_seed(cfg.seed)
                model = build_model(cfg.arch, cfg.pretrained).to(device)
                local_train(
                    model,
                    fd.client_train_loader[c],
                    cfg.local_only_epochs,
                    cfg.lr,
                    cfg.weight_decay,
                    fd.class_weights,
                    device,
                    optimizer=cfg.optimizer,
                )
                metrics, _, _ = evaluate(model, fd.client_val_loader[c], device, bn, cfg)
                res["local_only"][str(c)] = metrics
                print(
                    f"  fold {fold} C{c} local-only({cfg.local_only_epochs}ep): "
                    f"acc={metrics['accuracy']:.4f} "
                    f"auc={metrics['auc']:.4f}",
                    flush=True,
                )
        res["fedavg_finetune_mean_acc"] = float(
            np.mean([v["accuracy"] for v in res["fedavg_finetune"].values()])
        )
        if res["local_only"]:
            res["local_only_mean_acc"] = float(np.mean([v["accuracy"] for v in res["local_only"].values()]))
        res["done"] = True
        res["config"] = cfg.to_dict()
        results[key] = res
        save_json(results, out_path)
        print(f"  fold {fold}: fedavg+ft mean local acc = {res['fedavg_finetune_mean_acc']:.4f}")
        del fd
        _free_cuda()
    print(f"  saved {out_path}")
    return results


# ----------------------------------------------------------------------------------------------------------
#  sweep, localpred
# ----------------------------------------------------------------------------------------------------------


def stage_sweep(cfg: Config, methods: Sequence[str], seeds: Sequence[int]) -> None:
    """All (method, seed) runs of each fold from one data load; seed-major so early stops give whole folds."""
    maybe_mount_drive(cfg)
    device = get_device()
    for seed in seeds:
        for fold in cfg.folds:
            todo = [
                m
                for m in methods
                if not os.path.exists(
                    os.path.join(cfg.out_dir, "runs", run_tag(cfg, m, fold, seed), "result.json")
                )
            ]
            if not todo:
                print(f"  fold {fold} seed {seed}: all done")
                continue
            fd = FoldData(cfg, fold, device)
            for m in todo:
                stage_run(cfg, m, fold, seed, fd=fd)
            del fd
            _free_cuda()
    stage_tables(cfg)


def _localpred_run(
    cfg: Config,
    fd: FoldData,
    device: torch.device,
    name: str,
    d: str,
    seed: int,
    need: list,
    need_ft: list,
    need_fe: list,
) -> None:
    """Own-centre predictions for one finished run (plus fine-tune and swap-in variants for fedavg)."""
    stored = load_json(os.path.join(d, "result.json"))["config"]
    rcfg = run_config(cfg, name, stored)
    retention = Retention.from_config(rcfg)
    g = load_pt(os.path.join(d, "global_last.pt"))
    low = load_pt(os.path.join(d, "client_low.pt")) or {}
    personal = load_pt(os.path.join(d, "client_bn.pt")) or {}
    convs = retained_conv_layers(g, rcfg)
    model = build_model(cfg.arch, cfg.pretrained).to(device)
    ft = f"ft{cfg.ft_epochs}"

    def score(state: dict, c: int, path: str) -> float:
        model.load_state_dict(to_device(state, device))
        metrics, probs, labels = evaluate(
            model, fd.client_val_loader[c], device, fd.bn_client(c, cfg.bn_source_local), cfg
        )
        _save_pred(path, fd, c, probs, labels)
        return metrics["accuracy"]

    base = os.path.basename(d)
    for c in need:
        if is_retain_run(name) and c in low:
            state = retention.personalize(g, low[c], convs)
        elif name in PERSONAL_KEY_METHODS and c in personal:
            state = with_personal(g, personal[c])
        else:
            state = g
        acc = score(state, c, os.path.join(d, f"pred_local_C{c}.npz"))
        print(f"  {base} C{c}: acc={acc:.4f}", flush=True)
    for c in need_fe:
        acc = score(retention.personalize(g, low[c], convs), c, os.path.join(d, f"pred_local_fe_C{c}.npz"))
        print(f"  {base} C{c} fedfreq_eval: acc={acc:.4f}", flush=True)
    for c in need_ft:
        set_seed(seed)
        model.load_state_dict(to_device(g, device))
        local_train(
            model,
            fd.client_train_loader[c],
            cfg.ft_epochs,
            cfg.lr,
            cfg.weight_decay,
            fd.class_weights,
            device,
            optimizer=cfg.optimizer,
        )
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        acc = score(state, c, os.path.join(d, f"pred_local_{ft}_C{c}.npz"))
        print(f"  {base} C{c} +{ft}: acc={acc:.4f}", flush=True)


def stage_localpred(cfg: Config, seeds: Sequence[int]) -> None:
    """Per-client validation predictions for every finished run (pred_local_C{c}.npz and fedavg variants)."""
    maybe_mount_drive(cfg)
    device = get_device()
    for fold in cfg.folds:
        clients = [c for c in range(5) if c != fold]
        fd = None
        for seed in seeds:
            for name, d in run_dirs(cfg, fold, seed).items():
                ft = f"ft{cfg.ft_epochs}"
                need = [c for c in clients if not os.path.exists(os.path.join(d, f"pred_local_C{c}.npz"))]
                need_ft = [
                    c
                    for c in clients
                    if name == "fedavg" and not os.path.exists(os.path.join(d, f"pred_local_{ft}_C{c}.npz"))
                ]
                need_fe = [
                    c
                    for c in clients
                    if name == "fedavg" and not os.path.exists(os.path.join(d, f"pred_local_fe_C{c}.npz"))
                ]
                if not (need or need_ft or need_fe):
                    print(f"  skip {os.path.basename(d)}")
                    continue
                if fd is None:
                    fd = FoldData(cfg, fold, device)
                _localpred_run(cfg, fd, device, name, d, seed, need, need_ft, need_fe)
        del fd
        _free_cuda()


# ----------------------------------------------------------------------------------------------------------
#  divergence of client updates per band
# ----------------------------------------------------------------------------------------------------------


def _divergence_bands(states: Sequence[dict], convs: Sequence[str], cfg: Config) -> tuple[dict, dict]:
    """Across-client variance and CV of the DCT coefficients, summed per band and averaged per layer."""
    retention = Retention(cfg.t_low, cfg.t_high, cfg.dct_axis, "low")
    agg: dict = {b: {"var": [], "cv": [], "n": 0} for b in ("low", "mid", "high")}
    per_layer: dict = {}
    for k in convs:
        coef = torch.stack([retention.coefficients(st[k].float()) for st in states])
        var = coef.var(dim=0, unbiased=False)
        magnitude = coef.abs().mean(dim=0).clamp_min(1e-8)
        cv = var.sqrt() / magnitude
        low, mid, high = band_masks(coef.shape[-1], cfg.t_low, cfg.t_high)
        per_layer[k] = {}
        for b, mask in (("low", low), ("mid", mid), ("high", high)):
            if not bool(mask.any()):
                continue
            v, cvb = var[:, mask], cv[:, mask]
            agg[b]["var"].append(float(v.sum()))
            agg[b]["cv"].append(float(cvb.sum()))
            agg[b]["n"] += v.numel()
            per_layer[k][b] = {"var": float(v.mean()), "cv": float(cvb.mean())}
    summary = {
        b: {"var": sum(a["var"]) / a["n"], "cv": sum(a["cv"]) / a["n"], "n_coef": a["n"]}
        for b, a in agg.items()
    }
    for b in summary:
        summary[b]["var_ratio_vs_high"] = summary[b]["var"] / summary["high"]["var"]
        summary[b]["cv_ratio_vs_high"] = summary[b]["cv"] / summary["high"]["cv"]
    return summary, per_layer


def stage_divergence(cfg: Config, seeds: Sequence[int], source: str = "fedavg", epochs: int = 1) -> dict:
    """Train each client `epochs` from the final `source` model, then measure per-band client divergence."""
    maybe_mount_drive(cfg)
    device = get_device()
    out_path = os.path.join(cfg.out_dir, f"divergence_{source}.json")
    res = load_json(out_path) if os.path.exists(out_path) else {}
    for fold in cfg.folds:
        fd = None
        for seed in seeds:
            key = f"fold{fold}/seed{seed}"
            if key in res:
                print(f"  skip {key}")
                continue
            g = load_pt(os.path.join(cfg.out_dir, "runs", run_tag(cfg, source, fold, seed), "global_last.pt"))
            if g is None:
                print(f"  {key}: no {source} run")
                continue
            if fd is None:
                fd = FoldData(cfg, fold, device)
            model = build_model(cfg.arch, cfg.pretrained).to(device)
            states = []
            for c in fd.train_centers:
                set_seed(seed)
                model.load_state_dict(to_device(g, device))
                states.append(
                    local_train(
                        model,
                        fd.client_train_loader[c],
                        epochs,
                        cfg.lr,
                        cfg.weight_decay,
                        fd.class_weights,
                        device,
                        optimizer=cfg.optimizer,
                    )
                )
            summary, per_layer = _divergence_bands(states, conv_layer_names(g, cfg.pointwise), cfg)
            res[key] = {
                "bands": summary,
                "per_layer": per_layer,
                "epochs": epochs,
                "source": source,
                "dct_axis": cfg.dct_axis,
            }
            save_json(res, out_path)
            print(
                f"  {key}: "
                + "  ".join(
                    f"{b}: var={summary[b]['var']:.5f} ({summary[b]['var_ratio_vs_high']:.2f}x) "
                    f"cv={summary[b]['cv']:.3f} ({summary[b]['cv_ratio_vs_high']:.2f}x)"
                    for b in summary
                ),
                flush=True,
            )
        del fd
        _free_cuda()
    if res:
        print("\n  DIVERGENCE BY BAND (mean over folds and seeds)")
        for b in ("low", "mid", "high"):
            v = np.mean([r["bands"][b]["var"] for r in res.values()])
            c = np.mean([r["bands"][b]["cv"] for r in res.values()])
            vr = np.mean([r["bands"][b]["var_ratio_vs_high"] for r in res.values()])
            cr = np.mean([r["bands"][b]["cv_ratio_vs_high"] for r in res.values()])
            print(f"    {b:5s} raw var {v:.5f} ({vr:.2f}x high)   CV {c:.3f} ({cr:.2f}x high)")
    print(f"  saved {out_path}")
    return res


# ----------------------------------------------------------------------------------------------------------
#  tables and metrics dump
# ----------------------------------------------------------------------------------------------------------


def _tables_reeval(path: str) -> dict:
    """Summaries of reeval.json, mean over the folds present."""
    rows: dict = defaultdict(lambda: defaultdict(list))
    for key, v in load_json(path).items():
        _, method = key.split("/")
        for hk, m in v["heldout"].items():
            for metric in TABLE_METRICS:
                rows[(method, hk)][metric].append(m[metric])
        rows[(method, "local")]["accuracy"].append(v["local_mean_acc"])
    print("\nRE-EVALUATION OF SAVED CHECKPOINTS (mean over folds present)")
    print(
        f"{'method':12s} {'ckpt/bn':22s} {'n':>2s} {'acc':>7s} {'auc':>7s} {'spec':>7s} {'sens':>7s} "
        f"{'brier':>7s} {'ece':>7s} {'mce':>7s}"
    )
    for (method, hk), d in sorted(rows.items()):
        if hk != "local":
            values = " ".join(f"{np.mean(d[k]):7.4f}" for k in TABLE_METRICS[:-1])
            print(f"{method:12s} {hk:22s} {len(d['accuracy']):2d} {values}")
    print(
        "\nPER-CLIENT (personalized for fedfreq*, BN on client train, scored on client val), mean over folds"
    )
    for (method, hk), d in sorted(rows.items()):
        if hk == "local":
            print(f"  {method:12s} local acc {np.mean(d['accuracy']):.4f}  (folds={len(d['accuracy'])})")
    return {f"{m}/{h}": {k: float(np.mean(v)) for k, v in d.items()} for (m, h), d in rows.items()}


def _tables_runs(runs: str) -> dict:
    """Per-run lines and per-variant means over the finished runs of one results directory."""
    print("\nCORRECTED RUNS")
    agg: dict = defaultdict(list)
    for tag in sorted(os.listdir(runs)):
        rf = os.path.join(runs, tag, "result.json")
        if not os.path.exists(rf):
            continue
        r = load_json(rf)
        hl = r["heldout_last"]
        extra = ""
        if "local_mean_acc_fedfreq_eval" in r:
            extra = f" fedfreq_eval(local)={r['local_mean_acc_fedfreq_eval']:.4f}"
        print(
            f"  {tag:45s} heldout(last) acc={hl['accuracy']:.4f} auc={hl['auc']:.4f} ece={hl['ece']:.4f} "
            f"mce={hl['mce']:.4f} | local(last)={r['local_mean_acc_last']:.4f} "
            f"local(best_by_val)={r['local_mean_acc_best']:.4f}{extra}"
        )
        c = r["config"]
        variant = (
            r["method"],
            c.get("warm_start", 0),
            c.get("dct_axis", "flat"),
            c.get("t_low", 0.3),
            c.get("server_low", "freeze"),
            c.get("retain_layers", "all"),
        )
        agg[variant].append((hl["accuracy"], hl["auc"], r["local_mean_acc_last"]))
    for (m, ws, ax, tl, sl, rl), v in agg.items():
        a = np.array(v)
        if is_retain_method(m):
            label = f"ws={ws} axis={ax} t_low={tl} server_low={sl} retain={rl}"
        else:
            label = f"axis={ax}"
        print(
            f"  {m:14s} {label:55s} n={len(v)}: acc {a[:, 0].mean():.4f}±{a[:, 0].std():.4f} "
            f"auc {a[:, 1].mean():.4f}±{a[:, 1].std():.4f} local {a[:, 2].mean():.4f}±{a[:, 2].std():.4f}"
        )
    return {f"{m}_ws{ws}_{ax}_tl{tl}_sl{sl}_rl{rl}": v for (m, ws, ax, tl, sl, rl), v in agg.items()}


def stage_tables(cfg: Config) -> dict:
    """Collect reeval.json, finetune_*.json and every runs/*/result.json into tables.json."""
    out: dict = {}
    reeval_path = os.path.join(cfg.out_dir, "reeval.json")
    if os.path.exists(reeval_path):
        out["reeval"] = _tables_reeval(reeval_path)
    for f in sorted(os.listdir(cfg.out_dir)):
        if f.startswith("finetune_") and f.endswith(".json"):
            r = load_json(os.path.join(cfg.out_dir, f))
            accs = [
                v["fedavg_finetune_mean_acc"]
                for v in r.values()
                if isinstance(v, dict) and "fedavg_finetune_mean_acc" in v
            ]
            lo = [
                v["local_only_mean_acc"]
                for v in r.values()
                if isinstance(v, dict) and "local_only_mean_acc" in v
            ]
            print(
                f"\n{f}: fedavg+finetune local acc mean={np.mean(accs):.4f} over {len(accs)} folds"
                + (f"; local-only {np.mean(lo):.4f}" if lo else "")
            )
            out[f] = {"fedavg_finetune": accs, "local_only": lo}
    runs = os.path.join(cfg.out_dir, "runs")
    if os.path.isdir(runs):
        out["runs"] = _tables_runs(runs)
    save_json(out, os.path.join(cfg.out_dir, "tables.json"))
    return out


def stage_metrics(cfg: Config) -> list[dict]:
    """Dump the full metric set of every finished run (held-out and own-centre) to metrics.csv / .json."""
    runs = os.path.join(cfg.out_dir, "runs")
    rows: list[dict] = []
    if os.path.isdir(runs):
        for tag in sorted(os.listdir(runs)):
            rf = os.path.join(runs, tag, "result.json")
            if not os.path.exists(rf):
                continue
            r = load_json(rf)
            head = {"tag": tag, "method": r["method"], "fold": r["fold"], "seed": r["seed"]}
            for key in sorted(k for k in r if k.startswith("heldout_")):
                row = dict(head, split=key, client="")
                row.update({k: v for k, v in r[key].items() if not isinstance(v, list)})
                rows.append(row)
            for c, m in r["local_last"].items():
                rows.append(dict(head, split="local_last", client=c, **m))
            for c, m in r.get("local_fedfreq_eval", {}).items():
                rows.append(dict(head, split="local_fedfreq_eval", client=c, **m))
    columns: list[str] = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    with open(os.path.join(cfg.out_dir, "metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    save_json({"rows": rows}, os.path.join(cfg.out_dir, "metrics.json"))
    print(f"  {len(rows)} metric rows -> {os.path.join(cfg.out_dir, 'metrics.csv')}")
    return rows


# ----------------------------------------------------------------------------------------------------------
#  band (subset ablation) and arch (other backbones): sweep + localpred + bootstrap
# ----------------------------------------------------------------------------------------------------------


def retain_methods(cfg: Config) -> list[str]:
    """The FedRetain method names in cfg.methods (default: fedretain)."""
    names = [m for m in cfg.methods if is_retain_method(m)]
    return names or ["fedretain"]


def stage_band(
    cfg: Config, seeds: Sequence[int] | None = None, reference_rel: str | None = "results/fix"
) -> None:
    """Train FedRetain with cfg.subset_rule, score clients and bootstrap against the reference runs."""
    seeds = list(seeds if seeds is not None else cfg.seeds)
    stage_sweep(cfg, retain_methods(cfg), seeds)
    stage_localpred(cfg, seeds)
    stage_bootstrap_runs(cfg, seeds, n_boot=cfg.n_boot, seed=cfg.boot_seed, reference_rel=reference_rel)


def stage_arch(cfg: Config, methods: Sequence[str] | None = None, seeds: Sequence[int] | None = None) -> None:
    """Run the methods on cfg.arch (ImageNet backbone), then score clients and bootstrap."""
    methods = list(methods if methods is not None else cfg.methods)
    seeds = list(seeds if seeds is not None else cfg.seeds)
    model = build_model(cfg.arch, cfg.pretrained)
    state = model.state_dict()
    convs = conv_layer_names(state, cfg.pointwise)
    n_params = sum(p.numel() for p in model.parameters())
    kept = retained_count(state, convs, Retention.from_config(cfg))
    mb_full, mb_kept = upload_size_mb(n_params, kept)
    print(
        f"  {cfg.arch}: params {n_params:,}, {len(convs)} decomposable conv layers, retained = "
        f"{kept / n_params * 100:.1f}% of all params (MB/round full {mb_full:.1f}, FedRetain {mb_kept:.1f}); "
        f"rounds={cfg.rounds} epochs={cfg.local_epochs} lr={cfg.lr} warm_start={cfg.warm_start}"
    )
    stage_sweep(cfg, methods, seeds)
    stage_localpred(cfg, seeds)
    stage_bootstrap_runs(cfg, seeds, n_boot=cfg.n_boot, seed=cfg.boot_seed)
