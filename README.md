# fedretain

FedRetain: federated learning where, after a warm-up of `warm_start` rounds of plain FedAvg, every client keeps
a fixed subset of its convolutional weights local (never uploaded, never overwritten) and uploads only the rest.
The server freezes its own copy of that subset at its end-of-warm-up value and averages the remainder with
sample-proportional weights. The subset is chosen by an index rule along the input-channel axis of every conv
kernel: `low` (low-order DCT coefficients, the default), `high`, `random` (random DCT coefficients, same count)
or `raw` (random raw weights, no transform). The paper shows all four behave the same.

**Provenance.** Every number in the paper was produced by the two archived scripts in [`legacy/`](legacy/)
(`fedfreq_fix.py` 2026-10-20g and `fedfreq_extra.py` 2026-10-20x4). They are kept unchanged. This package is a
clean rewrite of them; `tests/test_equivalence.py` runs both on the same synthetic data and asserts that
training histories, saved weights (`torch.equal`), per-client predictions and bootstrap files are identical.
The result directories on Drive are read as they are: same directory names, same file names, same JSON keys.
The legacy method name `fedfreq_train` is accepted everywhere as an alias of `fedretain`.

## Install

Python 3.10 or newer. CPU is enough for the tests; the experiments ran on a Colab GPU.

```bash
git clone https://github.com/kakon2002/fedretain.git
cd fedretain
pip install -e ".[test]"
fedretain --help
```

## Data

Camelyon17-WILDS v1.0 patches: 455,954 patches of 96x96 pixels from 5 medical centres (hospitals), each
patch labelled tumour / normal and attributed to a patient and a lymph-node slide. The patches live on local
disk (extracted from a cached archive at the start of every Colab session) as

```
<patches_dir>/patient_{pid:03d}_node_{node}/patch_patient_{pid:03d}_node_{node}_x_{x}_y_{y}.png
```

The patch index is `metadata/metadata_enhanced.csv` on Drive with the columns `center, split, tumor, patient,
node, x_coord, y_coord, folder`. `split` is 0 for a centre's training patches and 1 for its validation
patches. Normalisation statistics and class weights come from `results/dataset_stats.json`.

Folds are leave-one-centre-out: fold `f` holds out centre `f` as the unseen test centre and the other four
centres are the federated clients. The paper uses folds 0, 1, 2 with seeds 42 and 43; centre 3 is the
development fold (`configs/dev.yaml`).

## Drive layout

```
HistTTA/
  metadata/metadata_enhanced.csv
  results/dataset_stats.json
  checkpoints/camelyon17_histofreqnet/fold_{f}/{method}_best.pt, {method}_global.pt, fedfreq_low_freq.pt
  results/fix/           main comparison, warm-up ablation, finetune baseline, divergence, figures
  results/band/          subset-rule and retained-fraction ablations
  results/dev/           development fold (centre 3)
  results/arch_mobilenetv3/, results/arch_efficientnetb2/   cross-architecture runs
```

Every results folder has the same structure:

```
results/<name>/
  runs/<tag>/result.json           config, per-round history, held-out metrics (last and val-selected)
  runs/<tag>/progress.json         resumable state (completed rounds, history, selection score)
  runs/<tag>/global_last.pt        global weights after the last round
  runs/<tag>/global_best_by_val.pt global weights at the best mean client-validation accuracy
  runs/<tag>/client_low.pt         every client's retained subset (also for the baselines: used by fedfreq_eval)
  runs/<tag>/server_low.pt         the server's frozen subset (FedRetain only)
  runs/<tag>/client_bn.pt          private tensors of fedbn / fedper / fedper4
  runs/<tag>/pred_heldout_{last,best_by_val}_{test_center,client_val}.npz
  runs/<tag>/pred_local_C{c}.npz, pred_local_ft1_C{c}.npz, pred_local_fe_C{c}.npz   own-centre predictions
  bootstrap_runs.json              patient-level paired bootstrap, per seed, pooled over folds
  finetune_e1_runs_seed{S}.json    FedAvg + 1 epoch of local fine-tuning
  divergence_fedavg.json           across-client divergence per band
  tables.json, metrics.csv         collected numbers
  figures/                         png, pdf and csv of every figure
```

Run tags: `{method}_fold{f}_seed{s}_{axis}` for the baselines; FedRetain adds `_ws{warm_start}`, then
`_tl{t_low}` when `t_low` is not 0.3, `_sl{server_low}` when not `freeze`, `_rl{retain_layers}` when not
`all`, `_rbhigh` / `_rbrandom` / `_rbrandom_raw` for the subset rules other than `low`, and `_pw` when 1x1
convolutions are included. Example: `fedretain_fold0_seed42_ws15_cin`, legacy `fedfreq_train_fold0_seed42_ws15_cin`.

## Evaluation protocol

Fixed number of rounds, last checkpoint reported, no test-based selection. Before scoring, BatchNorm statistics
are re-estimated with population statistics (cumulative average, momentum `None`) over the full unlabelled
held-out set, identically for every method. Own-centre evaluation re-estimates BN on the client's training
split and scores the client's validation split. Uncertainty: patient-level paired cluster bootstrap, 1000
resamples, pooled over folds, per seed.

## One command per table and figure

The paper settings are in `configs/base.yaml`; each experiment file extends it. `pipeline` runs the file's
`stages` list for every `variants` entry. Every stage skips finished work and resumes an interrupted run at
its last completed round (Colab disconnects are expected).

| Paper item | Command | Output |
| --- | --- | --- |
| Main comparison table (FedAvg, FedProx, FedBN, FedPer, FedRetain, FedAvg + fine-tuning) | `fedretain pipeline --config configs/main.yaml` | `results/fix/bootstrap_runs.json`, `tables.json` |
| Warm-up ablation (5 vs 15 rounds) | `fedretain pipeline --config configs/warmup.yaml` | `results/fix/bootstrap_runs.json` |
| Retained-fraction curve (random subset, t_low 0.1 / 0.3 / 0.5) | `fedretain pipeline --config configs/fraction.yaml` then `fedretain retention --config configs/main.yaml` | `results/fix/figures/retention_fraction.*` |
| Subset-rule ablation (low / high / random / raw) | `fedretain pipeline --config configs/subset.yaml` then `fedretain retention --config configs/main.yaml` | `results/fix/figures/band_ablation.*` |
| Development fold (centre 3) | `fedretain pipeline --config configs/dev.yaml` | `results/dev/` |
| Cross-architecture (MobileNetV3) | `fedretain pipeline --config configs/arch_mobilenetv3.yaml` | `results/arch_mobilenetv3/` |
| Cross-architecture (EfficientNet-B2, 1x1 convs included) | `fedretain pipeline --config configs/arch_mobilenetv3.yaml --override arch=efficientnetb2 pointwise=true out_rel=results/arch_efficientnetb2` | `results/arch_efficientnetb2/` |
| Accuracy over rounds, held-out heat maps, band split and upload size | `fedretain figures --config configs/main.yaml` | `results/fix/figures/accuracy_over_rounds.*`, `heatmap_*.*`, `bands_communication.*` |
| Across-client divergence per band | `fedretain divergence --config configs/main.yaml` then `fedretain figures --config configs/main.yaml` | `results/fix/divergence_fedavg.json`, `figures/divergence_fedavg.*` |
| Per-centre stain colour and t-SNE of features | `fedretain datafig --config configs/main.yaml --override seeds=42` | `results/fix/figures/stain_rgb.*`, `tsne_fold0.*` |
| Full metric dump of every run | `fedretain metrics --config configs/main.yaml` | `results/fix/metrics.csv` |
| Re-scoring of the originally submitted checkpoints | `fedretain reeval` then `fedretain bootstrap` | `results/fix/reeval.json`, `bootstrap_*.json` |

Any field can be overridden on the command line: `--override warm_start=5 t_low=0.1 seeds=42`. The legacy
flags still work as aliases, so the author's old notebook cells run unchanged after replacing
`python fedfreq_fix.py` / `python fedfreq_extra.py` with `fedretain`:

```bash
fedretain run --method fedavg --fold 0 --seed 42 --dct_axis cin --norm_source dataset_stats
fedretain sweep --methods fedfreq_train fedavg --seeds 42 43 --folds 0 1 2 --warm_start 15 --dct_axis cin
fedretain band --band random --raw --seeds 42 --folds 0 1 2 --warm_start 15 --dct_axis cin
fedretain arch --arch mobilenetv3 --methods fedavg fedfreq_train fedbn --seeds 42
```

Single stages: `check`, `reeval`, `bootstrap`, `finetune`, `run`, `sweep`, `tables`, `metrics`, `localpred`,
`bootstrap_runs`, `band`, `divergence`, `figures`, `datafig`, `arch`, `retention`, `pipeline`.
[`notebooks/colab.ipynb`](notebooks/colab.ipynb) has one cell per config.

## Package map

| Module | Contents | Legacy origin |
| --- | --- | --- |
| `config.py` | `Config` (every legacy field, plus `subset_rule`, `arch`, `pretrained`, `pointwise`, `optimizer`), YAML + overrides | `Config`, `parse` |
| `data.py` | patch index, folds, client splits, device-resident loaders, normalisation, `train_cap` | `Split`, `Loader`, `FoldData` |
| `model.py` | `HistoFreqNet` (identical modules and init), MobileNetV3 / EfficientNet-B2 factories | `HistoFreqNet`, `make_model` |
| `retention.py` | `DCT1D`, per-kernel transform along cin, subset rules, decompose / recompose, personalise, band counts | `WeightDCT` and the masks / DCT patches |
| `methods.py` | `FedAvg`, `FedProx`, `FedBN`, `FedPer`, `FedPer4`, `FedRetain` with `client_start` / `on_client_done` / `aggregate` | the `if method ==` branches of `stage_run` |
| `train.py` | `local_train` (AdamW or SGD, proximal term), resumable round loop, `run_tag`, `run_dirs` | `stage_run`, `run_tag`, `_run_dirs` |
| `adapt.py` | BN re-estimation with population statistics (full set or capped), prediction, evaluation | `calibrate_bn`, `predict`, `evaluate` |
| `metrics.py` | accuracy, AUC, sensitivity, specificity, F1, Brier, ECE / MCE (10 equal-width bins), worst-group | `metrics_from`, `_fast_auc` |
| `stats.py` | patient-level paired cluster bootstrap, `bootstrap_runs` | `_cluster_boot`, `stage_bootstrap*` |
| `stages.py` | check, reeval, finetune, sweep, localpred, divergence, tables, metrics dump, band, arch | both scripts |
| `figures.py` | curves, heat maps, bands / communication, divergence, retention fraction, band ablation, stain RGB, t-SNE | `stage_figures`, `stage_retention`, `stage_datafig` |
| `cli.py` | `fedretain <stage> --config ... [--override ...]` plus the legacy flags | `main` of both scripts |

## Tests

```bash
python -m pytest -q tests            # equivalence with the archived scripts + every stage, CPU, ~15 min
python legacy/tests/test_fix.py      # the original smoke tests of the archived scripts (in this order)
python legacy/tests/test_extra.py
python legacy/tests/test_new.py
```

`tests/test_equivalence.py` trains `fedavg`, `fedbn`, `fedper` and `fedretain` (low subset, 2 rounds,
warm-up 1) and `band --band random --raw` with both code bases on the same synthetic data and seed, then
asserts that `result.json['history']` (all but the wall-clock field), every saved tensor, every prediction
file and `bootstrap_runs.json` are identical. `tests/test_stages.py` runs every stage end to end through the
command line, including a resume from an interrupted round. CI runs both on CPU.
