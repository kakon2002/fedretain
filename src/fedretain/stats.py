"""Patient-level paired cluster bootstrap, pooled over folds and reported per seed."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Sequence

import numpy as np

from fedretain.config import Config
from fedretain.metrics import fast_auc
from fedretain.train import run_dirs, save_json


def summarise(values: np.ndarray) -> dict:
    """Mean and 95% percentile interval of bootstrap replicates (NaN-aware)."""
    return {
        "mean": float(np.nanmean(values)),
        "lo": float(np.nanpercentile(values, 2.5)),
        "hi": float(np.nanpercentile(values, 97.5)),
    }


def cluster_boot(
    labels: np.ndarray,
    patient: np.ndarray,
    probs: dict,
    rng: np.random.RandomState,
    n_boot: int,
    refs: str | Sequence[str],
) -> dict:
    """Resample patients with replacement; accuracy and AUC per method plus paired differences vs refs."""
    groups = defaultdict(list)
    for i, p in enumerate(patient):
        groups[int(p)].append(i)
    group_ids = list(groups)
    group_idx = [np.array(groups[g]) for g in group_ids]
    methods = list(probs)
    acc = {m: np.zeros(n_boot) for m in methods}
    auc = {m: np.zeros(n_boot) for m in methods}
    for b in range(n_boot):
        pick = rng.randint(0, len(group_ids), len(group_ids))
        idx = np.concatenate([group_idx[k] for k in pick])
        y = labels[idx]
        for m in methods:
            p = probs[m][idx]
            acc[m][b] = np.mean((p >= 0.5) == y)
            auc[m][b] = fast_auc(y, p)
    res = {}
    for m in methods:
        point = {"accuracy": float(np.mean((probs[m] >= 0.5) == labels)), "auc": fast_auc(labels, probs[m])}
        res[m] = {"accuracy": summarise(acc[m]), "auc": summarise(auc[m]), "point": point}
    for ref in ([refs] if isinstance(refs, str) else refs):
        if ref in methods:
            res["diff_vs_" + ref] = {
                m: {"accuracy": summarise(acc[m] - acc[ref]), "auc": summarise(auc[m] - auc[ref])}
                for m in methods
                if m != ref
            }
    res["n_patients"] = len(group_ids)
    return res


def _print_ci(name: str, r: dict) -> None:
    """One line of accuracy and AUC with their intervals."""
    print(
        f"    {name:22s} acc {r['point']['accuracy']:.4f} "
        f"[{r['accuracy']['lo']:.4f}, {r['accuracy']['hi']:.4f}]  "
        f"auc {r['point']['auc']:.4f} [{r['auc']['lo']:.4f}, {r['auc']['hi']:.4f}]"
    )


def _print_diff(name: str, ref: str, d: dict) -> None:
    """One line of paired differences against a reference method."""
    print(
        f"    {name:22s} - {ref:14s} dAcc {d['accuracy']['mean']:+.4f} [{d['accuracy']['lo']:+.4f}, "
        f"{d['accuracy']['hi']:+.4f}]  dAUC {d['auc']['mean']:+.4f} "
        f"[{d['auc']['lo']:+.4f}, {d['auc']['hi']:+.4f}]"
    )


def stage_bootstrap(
    cfg: Config,
    ck: str = "best",
    src: str | None = None,
    n_boot: int = 1000,
    ref: str = "fedavg",
    seed: int = 0,
) -> dict:
    """Bootstrap over the reeval predictions of the submitted checkpoints (pred_fold{f}_{m}_{ck}_{src})."""
    src = src or cfg.bn_source_heldout
    rng = np.random.RandomState(seed)
    out: dict = {
        "config": {"checkpoint": ck, "bn_source": src, "n_boot": n_boot, "cluster": "patient"},
        "per_fold": {},
        "pooled": {},
    }
    pooled = defaultdict(list)
    pooled_meta = []
    for fold in cfg.folds:
        prefix, suffix = f"pred_fold{fold}_", f"_{ck}_{src}.npz"
        files = {}
        for f in os.listdir(cfg.out_dir):
            if f.startswith(prefix) and f.endswith(suffix):
                files[f[len(prefix) : -len(suffix)]] = np.load(os.path.join(cfg.out_dir, f))
        if not files:
            print(f"  fold {fold}: no predictions for {ck}/{src}; run reeval first")
            continue
        meta = next(iter(files.values()))
        labels, patient = meta["labels"], meta["patient"]
        probs = {m: files[m]["probs"] for m in files}
        out["per_fold"][str(fold)] = cluster_boot(labels, patient, probs, rng, n_boot, ref)
        for m in probs:
            pooled[m].append(probs[m])
        pooled_meta.append((labels, patient + 1000 * fold))
        print(
            f"  fold {fold} ({out['per_fold'][str(fold)]['n_patients']} patients): "
            + ", ".join(f"{m}={out['per_fold'][str(fold)][m]['accuracy']['mean']:.4f}" for m in probs)
        )
    if pooled:
        common = [m for m in pooled if len(pooled[m]) == len(pooled_meta)]
        labels = np.concatenate([x[0] for x in pooled_meta])
        patient = np.concatenate([x[1] for x in pooled_meta])
        out["pooled"] = cluster_boot(
            labels, patient, {m: np.concatenate(pooled[m]) for m in common}, rng, n_boot, ref
        )
        print(
            f"\n  pooled over folds, {out['pooled']['n_patients']} patients "
            "(patient-level bootstrap, 95% CI):"
        )
        for m, r in out["pooled"].items():
            if m not in ("diff_vs_" + ref, "n_patients"):
                _print_ci(m, r)
        for m, r in out["pooled"].get("diff_vs_" + ref, {}).items():
            _print_diff(m, ref, r)
    path = os.path.join(cfg.out_dir, f"bootstrap_{ck}_{src}.json")
    save_json(out, path)
    print(f"  saved {path}")
    return out


def _collect_predictions(cfg: Config, seed: int, reference_rel: str | None) -> tuple[dict, dict]:
    """Held-out and own-centre prediction files of every finished run of one seed, keyed by method name."""
    finetune_name = f"fedavg_ft{cfg.ft_epochs}"
    heldout, local = defaultdict(list), defaultdict(list)
    for fold in cfg.folds:
        for name, d in run_dirs(cfg, fold, seed, reference_rel).items():
            f = os.path.join(d, "pred_heldout_last_test_center.npz")
            if os.path.exists(f):
                z = np.load(f)
                heldout[name].append((z["probs"], z["labels"], z["patient"] + 1000 * fold))
            for c in [c for c in range(5) if c != fold]:
                variants = (
                    ("", name),
                    (f"ft{cfg.ft_epochs}_", finetune_name),
                    ("fe_", "fedavg_fedfreq_eval"),
                )
                for suffix, label in variants:
                    if suffix and name != "fedavg":
                        continue
                    f = os.path.join(d, f"pred_local_{suffix}C{c}.npz")
                    if os.path.exists(f):
                        z = np.load(f)
                        local[label].append((z["probs"], z["labels"], z["patient"] + 1000 * fold))
    return heldout, local


def stage_bootstrap_runs(
    cfg: Config, seeds: Sequence[int], n_boot: int = 1000, seed: int = 0, reference_rel: str | None = None
) -> dict:
    """Paired patient-level bootstrap over the finished runs, per seed, pooled over folds."""
    rng = np.random.RandomState(seed)
    finetune_name = f"fedavg_ft{cfg.ft_epochs}"
    refs_heldout = ["fedavg"]
    refs_local = ["fedavg", finetune_name, "fedbn", "fedper"]
    out = {}
    for s in seeds:
        heldout, local = _collect_predictions(cfg, s, reference_rel)
        res = {}
        for split, data, refs in (("heldout", heldout, refs_heldout), ("local", local, refs_local)):
            n = max((len(v) for v in data.values()), default=0)
            full = {m: v for m, v in data.items() if len(v) == n}
            if not full:
                print(f"  seed {s} {split}: nothing found")
                continue
            first = next(iter(full.values()))
            labels = np.concatenate([x[1] for x in first])
            patient = np.concatenate([x[2] for x in first])
            probs = {}
            for m, v in full.items():
                assert np.array_equal(np.concatenate([x[1] for x in v]), labels), f"{m}: label order differs"
                probs[m] = np.concatenate([x[0] for x in v])
            r = cluster_boot(labels, patient, probs, rng, n_boot, refs)
            res[split] = r
            print(
                f"\n  seed {s} {split}: {r['n_patients']} patient clusters, {len(labels):,} patches, 95% CI"
            )
            for m in probs:
                _print_ci(m, r[m])
            for ref in refs:
                for m, d in r.get("diff_vs_" + ref, {}).items():
                    _print_diff(m, ref, d)
        out[str(s)] = res
    path = os.path.join(cfg.out_dir, "bootstrap_runs.json")
    save_json(out, path)
    print(f"\n  saved {path}")
    return out
