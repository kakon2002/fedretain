"""Figures: accuracy over rounds, heat maps, bands/communication, divergence, retention, ablation, data."""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import torch

from fedretain.adapt import autocast_ctx, calibrate_bn
from fedretain.config import Config
from fedretain.data import FoldData, get_device, maybe_mount_drive
from fedretain.model import build_model, feature_pool
from fedretain.retention import Retention, conv_layer_names, retained_percent, upload_size_mb
from fedretain.train import (
    load_json,
    load_pt,
    parse_variant,
    retained_conv_layers,
    run_dirs,
    save_json,
    set_seed,
    to_device,
)

BAND_COLOURS = {"low": "C0", "random": "C1", "high": "C2", "random (raw weights)": "C3"}


def _plt() -> Any:
    """matplotlib.pyplot with the non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, fig_dir: str, name: str) -> None:
    """Write a figure as PNG and PDF and close it."""
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=200)
    fig.savefig(os.path.join(fig_dir, f"{name}.pdf"))
    _plt().close(fig)


def _fig_dir(cfg: Config) -> str:
    """The figures directory of the current results folder."""
    path = os.path.join(cfg.out_dir, "figures")
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------------------------------------------------------
#  figures from the run logs
# ----------------------------------------------------------------------------------------------------------


def accuracy_over_rounds(cfg: Config, seeds: Sequence[int], fig_dir: str) -> dict:
    """Held-out and own-centre accuracy per round, mean and std over (fold, seed) runs of each method."""
    plt = _plt()
    curves: dict = {}
    for fold in cfg.folds:
        for seed in seeds:
            for name, d in run_dirs(cfg, fold, seed).items():
                history = load_json(os.path.join(d, "result.json"))["history"]
                curves.setdefault(name, []).append(
                    ([r["heldout"]["accuracy"] for r in history], [r["local_mean_acc"] for r in history])
                )
    if not curves:
        print("  no runs found")
        return curves
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, cs in sorted(curves.items()):
        n = min(len(c[0]) for c in cs)
        heldout = np.array([c[0][:n] for c in cs])
        local = np.array([c[1][:n] for c in cs])
        x = np.arange(1, n + 1)
        for a, y, label in ((ax[0], heldout, "held-out centre"), (ax[1], local, "own centre (personalized)")):
            a.plot(x, y.mean(0), label=f"{name} (n={len(cs)})")
            a.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.15)
            a.set_xlabel("communication round")
            a.set_ylabel("accuracy")
            a.set_title(label)
        for r in range(n):
            rows.append(
                (
                    name,
                    r + 1,
                    float(heldout[:, r].mean()),
                    float(heldout[:, r].std()),
                    float(local[:, r].mean()),
                    float(local[:, r].std()),
                )
            )
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, fig_dir, "accuracy_over_rounds")
    with open(os.path.join(fig_dir, "accuracy_over_rounds.csv"), "w") as handle:
        handle.write("method,round,heldout_acc_mean,heldout_acc_std,local_acc_mean,local_acc_std\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")
    print(f"  saved {fig_dir}/accuracy_over_rounds.{{png,pdf,csv}} for {sorted(curves)}")
    return curves


def heatmaps(cfg: Config, seeds: Sequence[int], fig_dir: str) -> None:
    """Held-out AUC and accuracy per test centre (mean over seeds), one heat map per metric."""
    plt = _plt()
    for metric in ("auc", "accuracy"):
        table: dict = {}
        for fold in cfg.folds:
            for seed in seeds:
                for name, d in run_dirs(cfg, fold, seed).items():
                    value = load_json(os.path.join(d, "result.json"))["heldout_last"][metric]
                    table.setdefault(name, {}).setdefault(fold, []).append(value)
        names = sorted(table)
        matrix = np.array([[np.mean(table[nm].get(f, [np.nan])) for f in cfg.folds] for nm in names])
        fig, a = plt.subplots(figsize=(1.6 + 1.2 * len(cfg.folds), 0.5 + 0.45 * len(names)))
        im = a.imshow(matrix, cmap="viridis", aspect="auto")
        a.set_xticks(range(len(cfg.folds)))
        a.set_xticklabels([f"test C{f}" for f in cfg.folds])
        a.set_yticks(range(len(names)))
        a.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(cfg.folds)):
                colour = "w" if matrix[i, j] < np.nanmean(matrix) else "k"
                a.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7, color=colour)
        fig.colorbar(im, ax=a, fraction=0.04)
        a.set_title(f"held-out {metric} by test centre")
        fig.tight_layout()
        _save(fig, fig_dir, f"heatmap_{metric}")
        with open(os.path.join(fig_dir, f"heatmap_{metric}.csv"), "w") as handle:
            handle.write("method," + ",".join(f"C{f}" for f in cfg.folds) + "\n")
            for nm, row in zip(names, matrix):
                handle.write(nm + "," + ",".join(f"{v:.4f}" for v in row) + "\n")
    print(f"  saved {fig_dir}/heatmap_auc and heatmap_accuracy")


def bands_communication(cfg: Config, fig_dir: str) -> dict:
    """Share of conv coefficients per band and MB per client per round with and without retention."""
    plt = _plt()
    model = build_model(cfg.arch, cfg.pretrained)
    state = model.state_dict()
    convs = conv_layer_names(state, cfg.pointwise)
    n_params = sum(p.numel() for p in model.parameters())
    retention = Retention.from_config(cfg)
    counts = np.array([retention.band_counts(tuple(state[k].shape)) for k in convs]).sum(0)
    mb_full, mb_kept = upload_size_mb(n_params, int(counts[0]))
    saving = (1 - mb_kept / mb_full) * 100
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].bar(["retained", "mid", "high"], counts / counts.sum() * 100)
    ax[0].set_ylabel("% of conv coefficients")
    ax[0].set_title(f"band split (t_low={cfg.t_low}, t_high={cfg.t_high})")
    ax[1].bar(["FedAvg / others", "FedRetain"], [mb_full, mb_kept])
    ax[1].set_ylabel("MB per client per round (fp32)")
    ax[1].set_title(f"upload: {saving:.1f}% saving")
    fig.tight_layout()
    _save(fig, fig_dir, "bands_communication")
    info = {
        "params": n_params,
        "conv_layers": len(convs),
        "band_counts": counts.tolist(),
        "mb_full": mb_full,
        "mb_fedfreq": mb_kept,
        "saving_pct": saving,
    }
    save_json(info, os.path.join(fig_dir, "bands_communication.json"))
    print(
        f"  bands {np.round(counts / counts.sum() * 100, 1).tolist()}%, "
        f"MB/round {mb_full:.1f} -> {mb_kept:.1f} "
        f"({saving:.1f}% saving)"
    )
    return info


def divergence_figures(cfg: Config, fig_dir: str) -> None:
    """Bar charts of the per-band across-client divergence for every divergence_*.json in the results dir."""
    plt = _plt()
    bands = ("low", "mid", "high")
    for f in sorted(os.listdir(cfg.out_dir)):
        if not (f.startswith("divergence_") and f.endswith(".json")):
            continue
        r = load_json(os.path.join(cfg.out_dir, f))
        var = np.array([[x["bands"][b]["var"] for b in bands] for x in r.values()])
        cv = np.array([[x["bands"][b]["cv"] for b in bands] for x in r.values()])
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].bar(bands, var.mean(0), yerr=var.std(0), capsize=3)
        ax[0].set_title("across-client variance of DCT coefficients")
        ax[0].set_ylabel("variance")
        ax[1].bar(bands, cv.mean(0), yerr=cv.std(0), capsize=3)
        ax[1].set_title("magnitude-normalised (CV)")
        ax[1].set_ylabel("std / mean |coef|")
        fig.suptitle(f"{f[:-5]} (n={len(r)} fold x seed)", fontsize=9)
        fig.tight_layout()
        _save(fig, fig_dir, f[:-5])
        print(f"  saved {fig_dir}/{f[:-5]}.png")


def stage_figures(cfg: Config, seeds: Sequence[int]) -> None:
    """All log-based figures: curves, heat maps, bands/communication and divergence."""
    fig_dir = _fig_dir(cfg)
    if not accuracy_over_rounds(cfg, seeds, fig_dir):
        return
    heatmaps(cfg, seeds, fig_dir)
    bands_communication(cfg, fig_dir)
    divergence_figures(cfg, fig_dir)


# ----------------------------------------------------------------------------------------------------------
#  retention fraction and band ablation (from bootstrap_runs.json files, no GPU)
# ----------------------------------------------------------------------------------------------------------


def _retained_pct(cfg: Config, t_low: float) -> float:
    """% of conv coefficients kept local for the low band at t_low (or a random set of that size)."""
    state = build_model(cfg.arch, cfg.pretrained).state_dict()
    convs = conv_layer_names(state, cfg.pointwise)
    retention = Retention(t_low, max(cfg.t_high, t_low), cfg.dct_axis, "low")
    return retained_percent(state, convs, retention)[0]


def load_bootstraps(cfg: Config, rels: Sequence[str]) -> dict:
    """{(seed, split): {name: accuracy summary}} from the bootstrap_runs.json of several results dirs."""
    boots: dict = {}
    for rel in rels:
        path = os.path.join(cfg.project, rel, "bootstrap_runs.json")
        if not os.path.exists(path):
            continue
        for seed, res in load_json(path).items():
            for split in ("heldout", "local"):
                for name, v in res.get(split, {}).items():
                    if isinstance(v, dict) and "accuracy" in v and "mean" in v["accuracy"]:
                        boots.setdefault((int(seed), split), {})[name] = v["accuracy"]
    return boots


def retention_fraction(cfg: Config, boots: dict, fig_dir: str) -> None:
    """Own-centre and unseen-centre accuracy against the fraction of coefficients each client keeps."""
    plt = _plt()
    seeds = sorted({s for s, _ in boots})
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    panels = ((ax[0], "local", "own centre (personalized)"), (ax[1], "heldout", "unseen centre"))
    for si, seed in enumerate(seeds):
        offset = (si - (len(seeds) - 1) / 2) * 0.6
        for a, split, title in panels:
            b = boots.get((seed, split), {})
            points: dict = {}
            for name, acc in b.items():
                variant = parse_variant(name, cfg.t_low)
                if variant is None or variant[0] != cfg.warm_start:
                    continue
                pct = _retained_pct(cfg, variant[1])
                points.setdefault(variant[2], []).append((pct, acc))
                rows.append(
                    (seed, split, name, variant[2], variant[1], pct, acc["mean"], acc["lo"], acc["hi"])
                )
            if "fedavg" in b:
                points.setdefault("fedavg", []).append((0.0, b["fedavg"]))
                rows.append(
                    (
                        seed,
                        split,
                        "fedavg",
                        "none",
                        0.0,
                        0.0,
                        b["fedavg"]["mean"],
                        b["fedavg"]["lo"],
                        b["fedavg"]["hi"],
                    )
                )
            for band, lst in points.items():
                lst.sort(key=lambda item: item[0])
                x = np.array([p for p, _ in lst]) + offset
                m = np.array([q["mean"] for _, q in lst])
                err = np.array([[q["mean"] - q["lo"] for _, q in lst], [q["hi"] - q["mean"] for _, q in lst]])
                if band == "fedavg":
                    a.errorbar(
                        x,
                        m,
                        yerr=err,
                        fmt="s",
                        color="k",
                        capsize=3,
                        label=f"FedAvg (0% retained), seed {seed}" if si == 0 else None,
                    )
                else:
                    a.errorbar(
                        x,
                        m,
                        yerr=err,
                        fmt="o-" if len(lst) > 1 else "o",
                        color=BAND_COLOURS.get(band, "C3"),
                        capsize=3,
                        alpha=1.0 if si == 0 else 0.55,
                        label=f"{band} band, seed {seed}",
                    )
            if split == "local" and "fedavg_ft1" in b:
                ft = b["fedavg_ft1"]
                a.axhline(
                    ft["mean"],
                    color="gray",
                    ls="--",
                    lw=1,
                    label="FedAvg + 1 epoch fine-tune" if si == 0 else None,
                )
                a.axhspan(ft["lo"], ft["hi"], color="gray", alpha=0.12)
            a.set_xlabel("% of conv coefficients kept local")
            a.set_ylabel("accuracy (95% patient-bootstrap CI)")
            a.set_title(title)
    for a in ax:
        a.legend(fontsize=6.5)
        a.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir, "retention_fraction")
    with open(os.path.join(fig_dir, "retention_fraction.csv"), "w") as handle:
        handle.write("seed,split,method,band,t_low,pct_conv_retained,acc_mean,acc_lo,acc_hi\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")
    print(f"  saved {fig_dir}/retention_fraction.{{png,pdf,csv}}")


def band_ablation(cfg: Config, boots: dict, fig_dir: str) -> None:
    """Accuracy of every retained-subset rule at the default fraction: which coefficients are kept."""
    plt = _plt()
    seeds = sorted({s for s, _ in boots})
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
    panels = ((ax[0], "local", "own centre (personalized)"), (ax[1], "heldout", "unseen centre"))
    for a, split, title in panels:
        labels, means, errs, colours = [], [], [], []
        for seed in seeds:
            for name, acc in boots.get((seed, split), {}).items():
                variant = parse_variant(name, cfg.t_low)
                if variant is None or variant[0] != cfg.warm_start or abs(variant[1] - cfg.t_low) > 1e-9:
                    continue
                labels.append(f"{variant[2]}\nseed {seed}")
                means.append(acc["mean"])
                errs.append([acc["mean"] - acc["lo"], acc["hi"] - acc["mean"]])
                colours.append(BAND_COLOURS.get(variant[2], "C3"))
        if not means:
            continue
        x = np.arange(len(means))
        a.bar(x, means, yerr=np.array(errs).T, color=colours, capsize=3)
        a.set_xticks(x)
        a.set_xticklabels(labels, fontsize=7)
        a.set_title(title)
        a.set_ylabel("accuracy")
        lo = min(m - e[0] for m, e in zip(means, errs))
        hi = max(m + e[1] for m, e in zip(means, errs))
        a.set_ylim(lo - (hi - lo) * 0.5, hi + (hi - lo) * 0.5)
        for seed, ls in zip(seeds, ("--", ":")):
            b = boots.get((seed, split), {})
            ref = "fedavg_ft1" if split == "local" else "fedavg"
            if ref in b:
                ref_label = "FedAvg + fine-tune" if ref == "fedavg_ft1" else "FedAvg"
                a.axhline(b[ref]["mean"], color="gray", ls=ls, lw=1, label=f"{ref_label}, seed {seed}")
        a.legend(fontsize=6.5)
    fig.suptitle(
        f"retained band at t_low={cfg.t_low} (same number of coefficients), warm-up {cfg.warm_start}",
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, fig_dir, "band_ablation")
    print(f"  saved {fig_dir}/band_ablation.{{png,pdf}}")


def stage_retention(cfg: Config, band_rel: str = "results/band", fix_rel: str = "results/fix") -> None:
    """Retention-fraction curve and band ablation from the two bootstrap_runs.json files (no GPU)."""
    boots = load_bootstraps(cfg, (fix_rel, band_rel))
    if not boots:
        print("  no bootstrap_runs.json found")
        return
    fig_dir = os.path.join(cfg.project, fix_rel, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    retention_fraction(cfg, boots, fig_dir)
    band_ablation(cfg, boots, fig_dir)


# ----------------------------------------------------------------------------------------------------------
#  figures that need the patches
# ----------------------------------------------------------------------------------------------------------


def stain_rgb(fd: FoldData, fold: int, fig_dir: str) -> dict:
    """Per-centre mean RGB over the validation splits (and a sample of the test centre)."""
    plt = _plt()
    rgb = {c: ds.images.reshape(-1, 3).mean(0) / 255.0 for c, ds in fd.client_val_ds.items()}
    rgb[fold] = fd.test_ds.images[::20].reshape(-1, 3).mean(0) / 255.0
    centres = sorted(rgb)
    matrix = np.array([rgb[c] for c in centres])
    fig, a = plt.subplots(figsize=(5, 3))
    width = 0.25
    for j, (channel, colour) in enumerate(zip("RGB", ("tab:red", "tab:green", "tab:blue"))):
        a.bar(np.arange(len(centres)) + (j - 1) * width, matrix[:, j], width, label=channel, color=colour)
    a.set_xticks(range(len(centres)))
    a.set_xticklabels([f"C{c}" for c in centres])
    a.set_ylabel("mean intensity")
    a.set_ylim(0.3, 0.9)
    a.legend(fontsize=8)
    a.set_title(f"per-centre colour (max per-channel gap {np.ptp(matrix, 0).max():.3f})")
    fig.tight_layout()
    _save(fig, fig_dir, "stain_rgb")
    save_json({f"C{c}": rgb[c].tolist() for c in centres}, os.path.join(fig_dir, "stain_rgb.json"))
    print("  per-centre mean RGB: " + "  ".join(f"C{c}={np.round(rgb[c], 3).tolist()}" for c in centres))
    return rgb


def tsne_features(cfg: Config, fd: FoldData, fold: int, seed: int, fig_dir: str, n_tsne: int) -> bool:
    """t-SNE of penultimate features: random init, FedAvg global and FedRetain personalized, same patches."""
    plt = _plt()
    from sklearn.manifold import TSNE

    device = fd.device
    n_tsne = min(n_tsne, min(len(ds) for ds in fd.client_val_ds.values()))
    dirs = run_dirs(cfg, fold, seed)
    avg_dir = dirs.get("fedavg")
    ret_dir = dirs.get(f"fedretain_ws{cfg.warm_start}") or dirs.get(f"fedfreq_train_ws{cfg.warm_start}")
    if not (avg_dir and ret_dir):
        print("  t-SNE skipped: need fedavg and fedretain runs for this fold/seed")
        return False
    g_avg = load_pt(os.path.join(avg_dir, "global_last.pt"))
    g_ret = load_pt(os.path.join(ret_dir, "global_last.pt"))
    low = load_pt(os.path.join(ret_dir, "client_low.pt"))
    convs = retained_conv_layers(g_ret, cfg)
    retention = Retention.from_config(cfg)
    model = build_model(cfg.arch, cfg.pretrained).to(device)
    feats: dict = {}

    def keep_features(module: Any, inputs: Any, output: torch.Tensor) -> None:
        feats["f"] = output.flatten(1).float().cpu()

    hook = feature_pool(model).register_forward_hook(keep_features)

    def embed(state: dict, c: int) -> np.ndarray:
        model.load_state_dict(to_device(state, device))
        calibrate_bn(model, fd.bn_client(c, cfg.bn_source_local)[0], device)
        model.eval()
        out = []
        with torch.no_grad():
            for bx, _ in fd.client_val_loader[c]:
                with autocast_ctx(device):
                    model(bx)
                out.append(feats["f"])
                if sum(len(o) for o in out) >= n_tsne:
                    break
        return torch.cat(out)[:n_tsne].numpy()

    points: dict = {"before FL (random init)": [], "FedAvg global": [], "FedRetain personalized": []}
    labels: list = []
    set_seed(0)
    init = {k: v.cpu() for k, v in build_model(cfg.arch, cfg.pretrained).state_dict().items()}
    for c in fd.train_centers:
        points["before FL (random init)"].append(embed(init, c))
        points["FedAvg global"].append(embed(g_avg, c))
        points["FedRetain personalized"].append(embed(retention.personalize(g_ret, low[c], convs), c))
        labels += [c] * n_tsne
    hook.remove()
    label_arr = np.array(labels)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, (name, parts) in zip(ax, points.items()):
        z = TSNE(n_components=2, init="pca", random_state=0, perplexity=min(30, n_tsne - 1)).fit_transform(
            np.concatenate(parts)
        )
        for c in fd.train_centers:
            a.scatter(*z[label_arr == c].T, s=3, label=f"C{c}")
        a.set_title(name)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].legend(markerscale=4, fontsize=8)
    fig.suptitle(
        f"penultimate features, fold {fold} (test C{fold}), {n_tsne} val patches per centre", fontsize=9
    )
    fig.tight_layout()
    _save(fig, fig_dir, f"tsne_fold{fold}")
    print(f"  saved {fig_dir}/tsne_fold{fold}.png")
    return True


def stage_datafig(cfg: Config, seeds: Sequence[int], n_tsne: int = 600) -> None:
    """Figures that need the patches: per-centre stain RGB and the t-SNE of penultimate features."""
    maybe_mount_drive(cfg)
    device = get_device()
    fig_dir = _fig_dir(cfg)
    fold = cfg.folds[0]
    fd = FoldData(cfg, fold, device)
    stain_rgb(fd, fold, fig_dir)
    tsne_features(cfg, fd, fold, seeds[0], fig_dir, n_tsne)
