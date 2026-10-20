# Task: turn two research scripts into a clean, reproducible Python package called `fedretain`

You are working in an empty git repository that contains only:

- `legacy/fedfreq_fix.py` (version 2026-10-20g) and `legacy/fedfreq_extra.py` (version 2026-10-20x4): the scripts that produced every number in the paper. Treat them as the ground truth for behaviour. Do not edit them.
- `legacy/tests/test_fix.py`, `test_extra.py`, `test_new.py`: smoke tests that build a synthetic Camelyon17-like dataset on CPU and run every stage. They already use relative paths and all pass (see `legacy/tests/README.md`); run them first to confirm your environment, then reuse their synthetic data generator. The legacy training loop has been verified to be bit-for-bit deterministic on CPU (same seed twice gives identical weights and histories), so the equivalence test below is achievable.

The experimental results are NOT in this repo. They live on the author's Google Drive under `HistTTA/results/` and were produced by the legacy scripts. The package you write must be able to read those result directories as they are (same directory names, same file names, same JSON keys), because they will not be regenerated.

## What the method is (for naming and docstrings)

FedRetain: federated learning where, after a warm-up of `warm_start` rounds of plain FedAvg, each client keeps a fixed subset of its convolutional weights local (never uploaded, never overwritten) and uploads only the rest; the server freezes its copy of that subset at its end-of-warm-up value and averages the remainder with sample-proportional weights. The subset is chosen by an index rule along the input-channel axis of every conv kernel: `low` (low-order DCT coefficients, the legacy default), `high`, `random` (random DCT coefficients, same count), `raw` (random raw weights, no transform). The paper shows all four behave the same. Legacy method name for this is `fedfreq_train`; the new public name is `fedretain`, with `fedfreq_train` kept as an alias everywhere results are loaded.

Baselines implemented in the legacy code: `fedavg`, `fedprox`, `fedbn`, `fedper` (local classifier head), `fedper4` (local head + last stage), `fedavg` + local fine-tuning (`finetune` stage), and `fedfreq_eval` (FedAvg weights with the client subset swapped in at evaluation only).

Evaluation protocol that must be preserved exactly: fixed number of rounds, last checkpoint reported, no test-based selection, test-time BatchNorm re-estimation with population statistics over the full unlabelled held-out set (identical for every method), own-centre evaluation with BN re-estimated on the client's train split and scored on the client's val split, patient-level paired cluster bootstrap (1000 resamples) pooled over folds, per seed.

## Target layout

```
fedretain/
  README.md
  pyproject.toml              package name fedretain, python >= 3.10, deps: torch, torchvision, numpy, scikit-learn, matplotlib, pyyaml, tqdm
  configs/
    base.yaml                 shared defaults (rounds 25, local_epochs 3, lr 1e-3, AdamW, batch size, folds 0 1 2, seeds 42 43, dct_axis cin, warm_start 15, t_low 0.3, t_high 0.7, weighting proportional, norm_source dataset_stats)
    main.yaml                 fedavg, fedprox, fedbn, fedper, fedretain, finetune
    warmup.yaml               fedretain warm_start 5 and 15
    fraction.yaml             fedretain random subset at t_low 0.1 / 0.3 / 0.5
    subset.yaml               fedretain low / high / random / raw at t_low 0.3
    dev.yaml                  the development-fold (centre 3) configurations
    arch_mobilenetv3.yaml     the cross-architecture run
  src/fedretain/
    __init__.py
    config.py                 dataclass Config + YAML loading + CLI overrides; keep every field name of the legacy Config
    data.py                   patch index, folds, client train/val splits, dataset-stat normalisation, train_cap
    model.py                  HistoFreqNet (identical module names and init), MobileNetV3 / EfficientNet-B2 factories
    retention.py              DCT1D, per-kernel transform along cin, subset rules low/high/random/raw, decompose/recompose, personalize, band counts and upload size
    methods.py                one small class per method with hooks: client_start(), on_client_done(), aggregate(); fedavg, fedprox, fedbn, fedper, fedper4, fedretain. No monkeypatching anywhere.
    train.py                  local_train (AdamW and the SGD variant for pretrained backbones, proximal term), the round loop with per-round resumable checkpoints, run_tag()
    adapt.py                  BN re-estimation (population statistics, full set), the capped per-round variant
    metrics.py                accuracy, AUC, sensitivity, specificity, F1, Brier, ECE, MCE (10 equal-width bins), worst-group accuracy
    stats.py                  patient-level paired cluster bootstrap, bootstrap_runs
    stages.py                 sweep, finetune, localpred, bootstrap_runs, divergence, tables, metrics dump
    figures.py                accuracy_over_rounds, heatmaps, bands_communication, divergence, retention_fraction, band_ablation, stain_rgb, tsne
    cli.py                    `fedretain <stage> --config configs/x.yaml [--override key=value ...]`
  notebooks/
    colab.ipynb               short cells: mount Drive, extract data from the cached tar, pip install -e ., one cell per config, figure cell, flush-and-disconnect cell
  tests/
    conftest.py               synthetic dataset fixture (reuse the generator from legacy/tests/test_fix.py)
    test_equivalence.py       see below, this is the most important test
    test_stages.py            every stage runs end to end on the synthetic data
  legacy/                     untouched
  .github/workflows/ci.yml    runs the tests on CPU
```

## Hard requirements

1. **Numerical equivalence.** `tests/test_equivalence.py` must run the legacy `fedfreq_fix.py` and the new package on the same synthetic data, same seed, CPU, for `fedavg`, `fedbn`, `fedper`, `fedretain` (low, 2 rounds, warm_start 1) and for `band random --raw`, then assert that `result.json['history']` and the saved `global_last.pt` tensors are identical (`torch.equal`, not allclose). Also assert that `bootstrap_runs.json` from both is identical for the same `--n_boot` and bootstrap seed. If something cannot be made identical, stop and report exactly which operation differs rather than loosening the test.
2. **Result compatibility.** Run directory names (`run_tag`), file names inside a run directory (`result.json`, `progress.json`, `global_last.pt`, `global_best_by_val.pt`, `client_low.pt`, `server_low.pt`, `client_bn.pt`, `pred_heldout_last_test_center.npz`, `pred_local_C{c}.npz`, `pred_local_ft1_C{c}.npz`, `pred_local_fe_C{c}.npz`), the JSON keys in `result.json`, `bootstrap_runs.json`, `divergence_fedavg.json`, `finetune_e1_runs_seed{S}.json`, and the `results/fix`, `results/band`, `results/dev`, `results/arch_*` sub-folders must stay exactly as the legacy code writes them. The new code may write new runs under the new method name `fedretain`, but every loader must accept both `fedfreq_train...` and `fedretain...` tags.
3. **No monkeypatching.** Everything `fedfreq_extra.py` does by patching (`WeightDCT.masks`, `DCT1D.forward`, `run_tag`, `HistoFreqNet`, `conv_layer_names`, `local_train`, `_run_dirs`) becomes a config option or a method class.
4. **Resumability.** Every stage must skip finished work and resume a run at the last completed round, as the legacy code does, because the author runs on Colab with frequent disconnects.
5. **Code style.** Plain, readable Python: type hints on public functions, one-line docstrings, no clever one-liners, no lines over 110 characters, no commented-out code, names that a reviewer can read (`retained_fraction`, not `tl`). Keep the legacy flag names available as CLI aliases so the author's old notebook cells still work.
6. **README.** How to install, where the data comes from (Camelyon17-WILDS v1.0 patches, 455,954 patches, 5 centres), how the Drive layout looks, one command per table and figure of the paper, how to run the tests, and a clear statement that the numbers in the paper were produced by the archived scripts in `legacy/`, which the package reproduces bit for bit on the equivalence test.
7. **Commits.** Small commits with meaningful messages, in this order: scaffold, config, data+model, retention, methods, train loop, adapt+metrics, stats, stages, figures, cli, notebook, tests, README, CI.

## Work order

1. Read both legacy scripts fully before writing anything. Draw the call graph of every stage in a scratch file so nothing is dropped.
2. Build `tests/conftest.py` and get the legacy smoke tests running from their new location with relative paths, so you have a known-good baseline.
3. Write the package module by module, running the equivalence test after `train.py` exists and after every later module.
4. Write the notebook last, from the CLI.
5. Finish by running the full test suite, then print a short report: what maps to what, any behaviour you could not reproduce identically, and anything in the legacy code that looks like a bug (report, do not fix silently).

Do not ask me questions unless a decision changes results; make the sensible choice and note it in the final report.
