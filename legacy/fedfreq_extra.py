"""
fedfreq_extra.py  -  additional experiments for the revision, built on fedfreq_fix.py (which it imports unchanged).

    !python fedfreq_extra.py band --band high   --seeds 42 --folds 0 1 2 ...   # retain the HIGH band locally instead of the low band
    !python fedfreq_extra.py band --band random --seeds 42 --folds 0 1 2 ...   # retain a seeded random 30% of coefficients
    !python fedfreq_extra.py divergence --seeds 42 43 --folds 0 1 2 ...        # cross-client weight divergence per band
    !python fedfreq_extra.py figures --out_rel results/fix                     # log-based figures: curves, AUC heatmap, bands/MB, divergence
    !python fedfreq_extra.py datafig --out_rel results/fix --seeds 42          # data-based figures: per-centre stain RGB, t-SNE of features
    !python fedfreq_extra.py arch --arch mobilenetv3 --methods fedavg fedfreq_train fedbn --seeds 42 ...   # other backbones

`band` trains fedfreq_train (warm_start from --warm_start) with the retained band swapped, writes to --out_rel
(default results/band), then scores per-client predictions and runs the paired patient-level bootstrap against the
FedAvg and low-band FedFreq runs found in results/fix. Nothing in results/fix is modified.
"""
import os, re, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import fedfreq_fix as ff

VERSION = "2026-10-20x4"

# ----------------------------------------------------------------------
#  retained-band swap: patch WeightDCT.masks so the "low" slot holds another band
# ----------------------------------------------------------------------

_orig_masks = ff.WeightDCT.masks
_rand_cache = {}

def _masks_high(n, t_low, t_high):
    low, mid, high = _orig_masks(n, t_low, t_high)
    return high, mid, low                      # slot 0 (retained) = high-index coefficients

def _masks_random(n, t_low, t_high):
    low, mid, high = _orig_masks(n, t_low, t_high)
    if n not in _rand_cache:
        g = torch.Generator().manual_seed(1000 + n)
        pick = torch.randperm(n, generator=g)[:int(low.sum())]
        m = torch.zeros(n, dtype=torch.bool); m[pick] = True
        _rand_cache[n] = m
    m = _rand_cache[n]
    return m, ~m, torch.zeros(n, dtype=torch.bool)   # slot 0 = random set of the same size as the low band

_orig_run_tag = ff.run_tag

def apply_band(band):
    ff.WeightDCT.masks = staticmethod({'low': _orig_masks, 'high': _masks_high, 'random': _masks_random}[band])
    ff.run_tag = _orig_run_tag if band == 'low' else (lambda cfg, method, fold, seed:
        _orig_run_tag(cfg, method, fold, seed).replace(f'_{cfg.dct_axis}', f'_rb{band}_{cfg.dct_axis}') if method == 'fedfreq_train'
        else _orig_run_tag(cfg, method, fold, seed))

_orig_dct_fwd, _orig_dct_inv = ff.DCT1D.forward, ff.DCT1D.inverse

def apply_raw(raw):
    """raw=True: no transform at all. The 'coefficients' are the raw weights along the input-channel axis, so the
    retained set is a fixed random subset of input channels per layer (same mask structure as --band random)."""
    ff.DCT1D.forward = staticmethod((lambda x: x) if raw else _orig_dct_fwd)
    ff.DCT1D.inverse = staticmethod((lambda x: x) if raw else _orig_dct_inv)
    if raw:
        prev = ff.run_tag
        ff.run_tag = (lambda cfg, method, fold, seed: prev(cfg, method, fold, seed).replace(f'_{cfg.dct_axis}', f'_raw_{cfg.dct_axis}')
                      if method == 'fedfreq_train' else prev(cfg, method, fold, seed))

_orig_run_dirs = ff._run_dirs

def _merged_run_dirs(cfg, fold, seed, base_rel='results/fix'):
    """runs of this out_rel plus the FedAvg and low-band FedFreq reference runs from results/fix."""
    out = _orig_run_dirs(cfg, fold, seed)
    ref = ff.Config(**{**ff.asdict(cfg), 'out_rel': base_rel})
    for name, d in _orig_run_dirs(ref, fold, seed).items():
        if name in ('fedavg', f'fedfreq_train_ws{cfg.warm_start}') and name not in out: out[name] = d
    return out

def stage_band(cfg, band, seeds, raw=False):
    apply_band(band); apply_raw(raw)
    try:
        ff.stage_sweep(cfg, ['fedfreq_train'], seeds)
        ff.stage_localpred(cfg, seeds)
        ff._run_dirs = _merged_run_dirs
        ff.stage_bootstrap_runs(cfg, seeds)
    finally:
        ff._run_dirs = _orig_run_dirs; apply_raw(False); apply_band('low')

# ----------------------------------------------------------------------
#  other backbones (ImageNet-pretrained, as in the submitted paper): patch the model factory, the conv-layer
#  selection (depthwise layers have no channel axis) and the optimiser (SGD 0.9 momentum, as the paper states)
# ----------------------------------------------------------------------

ARCH = {'histofreqnet': dict(rounds=25, local_epochs=3, lr=1e-3, opt='adamw'),
        'mobilenetv3':  dict(rounds=20, local_epochs=1, lr=1e-2, opt='sgd'),
        'efficientnetb2': dict(rounds=20, local_epochs=1, lr=1e-2, opt='sgd')}

def make_model(arch, pretrained=True):
    import torchvision, torch.nn as nn
    if arch == 'histofreqnet': return ff.HistoFreqNet()
    if arch == 'mobilenetv3':
        m = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    elif arch == 'efficientnetb2':
        m = torchvision.models.efficientnet_b2(weights='IMAGENET1K_V1' if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    else: raise ValueError(arch)
    return m

_orig_HistoFreqNet = ff.HistoFreqNet
_orig_conv_layer_names = ff.conv_layer_names
_orig_local_train = ff.local_train

def apply_arch(arch, pretrained=True, pointwise=False):
    """Route fedfreq_fix's model construction, conv selection and optimiser to `arch`. HistoFreqNet is untouched.
    pointwise=True also decomposes 1x1 convolutions (the remedy proposed in the paper for depthwise-separable nets)."""
    if arch == 'histofreqnet': return
    ff.HistoFreqNet = lambda num_classes=2: make_model(arch, pretrained)
    if pointwise:
        ff.conv_layer_names = lambda state: [k for k, p in state.items() if p.dim() == 4 and p.shape[1] > 1]       # every non-depthwise conv
        ff.run_tag = lambda cfg, method, fold, seed: _orig_run_tag(cfg, method, fold, seed).replace(f'_{cfg.dct_axis}', f'_pw_{cfg.dct_axis}') \
            if method == 'fedfreq_train' else _orig_run_tag(cfg, method, fold, seed)
    else:
        ff.conv_layer_names = lambda state: [k for k in _orig_conv_layer_names(state) if state[k].shape[1] > 1]   # skip depthwise
    def local_train_sgd(model, loader, epochs, lr, wd, class_weights, device, prox_mu=0.0):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        anchor = [p.detach().clone() for p in model.parameters()] if prox_mu > 0 else None
        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with ff.autocast_ctx(device): loss = crit(model(bx), by)
                if anchor is not None: loss = loss + 0.5 * prox_mu * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ff.local_train = local_train_sgd

def stage_arch(cfg, arch, methods, seeds, pretrained=True, pointwise=False, cli=None):
    apply_band('low'); apply_arch(arch, pretrained, pointwise)
    for k, v in ARCH[arch].items():
        if k != 'opt' and getattr(cli, k, None) is None: setattr(cfg, k, v)   # explicit --rounds/--local_epochs win
    m = ff.HistoFreqNet(); st = m.state_dict(); convs = ff.conv_layer_names(st); n = sum(p.numel() for p in m.parameters())
    lo = sum(ff.WeightDCT.band_counts(st[k].shape, cfg.t_low, cfg.t_high, cfg.dct_axis)[0] for k in convs)
    print(f'  {arch}: params {n:,}, {len(convs)} decomposable conv layers, low band = {lo/n*100:.1f}% of all params '
          f'(MB/round full {n*4/2**20:.1f}, FedFreq {(n-lo)*4/2**20:.1f}); rounds={cfg.rounds} epochs={cfg.local_epochs} lr={cfg.lr} warm_start={cfg.warm_start}')
    ff.stage_sweep(cfg, methods, seeds)
    ff.stage_localpred(cfg, seeds)
    ff.stage_bootstrap_runs(cfg, seeds)

# ----------------------------------------------------------------------
#  spectral divergence across clients, per band
# ----------------------------------------------------------------------

def stage_divergence(cfg, seeds, source='fedavg', epochs=1):
    """Each client trains `epochs` local epochs from the final global model of the `source` run; every conv layer is
    decomposed along cfg.dct_axis and, per coefficient, the variance across clients is computed. Reported per band:
    raw = mean variance; cv = mean of std / mean|coef| (magnitude-normalised). Ratios are relative to the high band."""
    ff.maybe_mount_drive(cfg); device = ff.get_device()
    out_path = os.path.join(cfg.out_dir, f'divergence_{source}.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for fold in cfg.folds:
        fd = None
        for seed in seeds:
            key = f'fold{fold}/seed{seed}'
            if key in res: print(f'  skip {key}'); continue
            d = os.path.join(cfg.out_dir, 'runs', ff.run_tag(cfg, source, fold, seed))
            g = ff._load_pt(os.path.join(d, 'global_last.pt'))
            if g is None: print(f'  {key}: no {source} run'); continue
            if fd is None: fd = ff.FoldData(cfg, fold, device)
            model = ff.HistoFreqNet().to(device); states = []
            for c in fd.train_centers:
                ff.set_seed(seed); model.load_state_dict({k: v.to(device) for k, v in g.items()})
                states.append(ff.local_train(model, fd.client_train_loader[c], epochs, cfg.lr, cfg.weight_decay, fd.class_weights, device))
            convs = ff.conv_layer_names(g); agg = {b: {'var': [], 'cv': [], 'n': 0} for b in ('low', 'mid', 'high')}
            per_layer = {}
            for k in convs:
                coef = torch.stack([ff.WeightDCT._coef(st[k].float(), cfg.dct_axis) for st in states])   # (K, vecs, n)
                var = coef.var(dim=0, unbiased=False); mag = coef.abs().mean(dim=0).clamp_min(1e-8)
                cv = var.sqrt() / mag
                low, mid, high = _orig_masks(coef.shape[-1], cfg.t_low, cfg.t_high)
                per_layer[k] = {}
                for b, m in (('low', low), ('mid', mid), ('high', high)):
                    if not bool(m.any()): continue
                    v = var[:, m]; cvb = cv[:, m]
                    agg[b]['var'].append(float(v.sum())); agg[b]['cv'].append(float(cvb.sum())); agg[b]['n'] += v.numel()
                    per_layer[k][b] = {'var': float(v.mean()), 'cv': float(cvb.mean())}
            summ = {b: {'var': sum(a['var']) / a['n'], 'cv': sum(a['cv']) / a['n'], 'n_coef': a['n']} for b, a in agg.items()}
            for b in summ: summ[b]['var_ratio_vs_high'] = summ[b]['var'] / summ['high']['var']; summ[b]['cv_ratio_vs_high'] = summ[b]['cv'] / summ['high']['cv']
            res[key] = {'bands': summ, 'per_layer': per_layer, 'epochs': epochs, 'source': source, 'dct_axis': cfg.dct_axis}
            json.dump(res, open(out_path, 'w'), indent=1)
            print(f'  {key}: ' + '  '.join(f'{b}: var={summ[b]["var"]:.5f} ({summ[b]["var_ratio_vs_high"]:.2f}x) cv={summ[b]["cv"]:.3f} ({summ[b]["cv_ratio_vs_high"]:.2f}x)' for b in summ), flush=True)
        del fd
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if res:
        print('\n  DIVERGENCE BY BAND (mean over folds and seeds)')
        for b in ('low', 'mid', 'high'):
            v = np.mean([r['bands'][b]['var'] for r in res.values()]); c = np.mean([r['bands'][b]['cv'] for r in res.values()])
            vr = np.mean([r['bands'][b]['var_ratio_vs_high'] for r in res.values()]); cr = np.mean([r['bands'][b]['cv_ratio_vs_high'] for r in res.values()])
            print(f'    {b:5s} raw var {v:.5f} ({vr:.2f}x high)   CV {c:.3f} ({cr:.2f}x high)')
    print(f'  saved {out_path}')
    return res

# ----------------------------------------------------------------------
#  figures from the run logs
# ----------------------------------------------------------------------

def stage_figures(cfg, seeds):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    runs = os.path.join(cfg.out_dir, 'runs'); fig_dir = os.path.join(cfg.out_dir, 'figures'); os.makedirs(fig_dir, exist_ok=True)
    curves = {}   # name -> list of (heldout acc per round, local mean acc per round)
    for fold in cfg.folds:
        for seed in seeds:
            for name, d in ff._run_dirs(cfg, fold, seed).items():
                h = json.load(open(os.path.join(d, 'result.json')))['history']
                curves.setdefault(name, []).append(([r['heldout']['accuracy'] for r in h], [r['local_mean_acc'] for r in h]))
    if not curves: print('  no runs found'); return
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, cs in sorted(curves.items()):
        n = min(len(c[0]) for c in cs)
        H = np.array([c[0][:n] for c in cs]); L = np.array([c[1][:n] for c in cs]); x = np.arange(1, n + 1)
        for a, Y, lab in ((ax[0], H, 'held-out centre'), (ax[1], L, 'own centre (personalized)')):
            a.plot(x, Y.mean(0), label=f'{name} (n={len(cs)})'); a.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), alpha=0.15)
            a.set_xlabel('communication round'); a.set_ylabel('accuracy'); a.set_title(lab)
        for r in range(n): rows.append((name, r + 1, float(H[:, r].mean()), float(H[:, r].std()), float(L[:, r].mean()), float(L[:, r].std())))
    ax[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'accuracy_over_rounds.png'), dpi=200); fig.savefig(os.path.join(fig_dir, 'accuracy_over_rounds.pdf'))
    with open(os.path.join(fig_dir, 'accuracy_over_rounds.csv'), 'w') as f:
        f.write('method,round,heldout_acc_mean,heldout_acc_std,local_acc_mean,local_acc_std\n')
        for r in rows: f.write(','.join(str(v) for v in r) + '\n')
    print(f'  saved {fig_dir}/accuracy_over_rounds.{{png,pdf,csv}} for {sorted(curves)}')
    plt.close(fig)

    # AUC / accuracy per held-out centre, mean over seeds (heatmap, replaces Figure 5)
    for metric in ('auc', 'accuracy'):
        tab = {}
        for fold in cfg.folds:
            for seed in seeds:
                for name, d in ff._run_dirs(cfg, fold, seed).items():
                    tab.setdefault(name, {}).setdefault(fold, []).append(json.load(open(os.path.join(d, 'result.json')))['heldout_last'][metric])
        names = sorted(tab); M = np.array([[np.mean(tab[nm].get(f, [np.nan])) for f in cfg.folds] for nm in names])
        fig, a = plt.subplots(figsize=(1.6 + 1.2 * len(cfg.folds), 0.5 + 0.45 * len(names)))
        im = a.imshow(M, cmap='viridis', aspect='auto'); a.set_xticks(range(len(cfg.folds))); a.set_xticklabels([f'test C{f}' for f in cfg.folds])
        a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(cfg.folds)): a.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=7, color='w' if M[i, j] < np.nanmean(M) else 'k')
        fig.colorbar(im, ax=a, fraction=0.04); a.set_title(f'held-out {metric} by test centre'); fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'heatmap_{metric}.png'), dpi=200); fig.savefig(os.path.join(fig_dir, f'heatmap_{metric}.pdf')); plt.close(fig)
        with open(os.path.join(fig_dir, f'heatmap_{metric}.csv'), 'w') as f:
            f.write('method,' + ','.join(f'C{f}' for f in cfg.folds) + '\n')
            for nm, row in zip(names, M): f.write(nm + ',' + ','.join(f'{v:.4f}' for v in row) + '\n')
    print(f'  saved {fig_dir}/heatmap_auc and heatmap_accuracy')

    # band fractions and communication (replaces Figure 7a/b), from the model definition
    m = ff.HistoFreqNet(); st = m.state_dict(); convs = ff.conv_layer_names(st); n = sum(p.numel() for p in m.parameters())
    bc = np.array([ff.WeightDCT.band_counts(st[k].shape, cfg.t_low, cfg.t_high, cfg.dct_axis) for k in convs]).sum(0)
    mb_full = n * 4 / 2**20; mb_ff = (n - bc[0]) * 4 / 2**20
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].bar(['low', 'mid', 'high'], bc / bc.sum() * 100); ax[0].set_ylabel('% of conv coefficients'); ax[0].set_title(f'band split (t_low={cfg.t_low}, t_high={cfg.t_high})')
    ax[1].bar(['FedAvg / others', 'FedFreq'], [mb_full, mb_ff]); ax[1].set_ylabel('MB per client per round (fp32)'); ax[1].set_title(f'upload: {(1 - mb_ff / mb_full) * 100:.1f}% saving')
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'bands_communication.png'), dpi=200); fig.savefig(os.path.join(fig_dir, 'bands_communication.pdf')); plt.close(fig)
    json.dump({'params': n, 'conv_layers': len(convs), 'band_counts': bc.tolist(), 'mb_full': mb_full, 'mb_fedfreq': mb_ff,
               'saving_pct': (1 - mb_ff / mb_full) * 100}, open(os.path.join(fig_dir, 'bands_communication.json'), 'w'), indent=1)
    print(f'  bands {np.round(bc / bc.sum() * 100, 1).tolist()}%, MB/round {mb_full:.1f} -> {mb_ff:.1f} ({(1 - mb_ff / mb_full) * 100:.1f}% saving)')

    # divergence bars (replaces Figure 8), if stage_divergence has run
    for f in sorted(os.listdir(cfg.out_dir)):
        if f.startswith('divergence_') and f.endswith('.json'):
            r = json.load(open(os.path.join(cfg.out_dir, f))); bands = ('low', 'mid', 'high')
            V = np.array([[x['bands'][b]['var'] for b in bands] for x in r.values()]); C = np.array([[x['bands'][b]['cv'] for b in bands] for x in r.values()])
            fig, ax = plt.subplots(1, 2, figsize=(7, 3))
            ax[0].bar(bands, V.mean(0), yerr=V.std(0), capsize=3); ax[0].set_title('across-client variance of DCT coefficients'); ax[0].set_ylabel('variance')
            ax[1].bar(bands, C.mean(0), yerr=C.std(0), capsize=3); ax[1].set_title('magnitude-normalised (CV)'); ax[1].set_ylabel('std / mean |coef|')
            fig.suptitle(f'{f[:-5]} (n={len(r)} fold x seed)', fontsize=9); fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f'{f[:-5]}.png'), dpi=200); fig.savefig(os.path.join(fig_dir, f'{f[:-5]}.pdf')); plt.close(fig)
            print(f'  saved {fig_dir}/{f[:-5]}.png')

def _parse_variant(name, t_low_default):
    """'fedfreq_train_ws15_tl0.1_rbrandom' -> (warm_start, t_low, band); None for non-fedfreq_train names."""
    m = re.match(r'fedfreq_train_ws(\d+)(?:_tl([\d.]+))?(?:_rb([a-z]+))?(_raw)?$', name)
    if not m: return None
    return int(m.group(1)), float(m.group(2) or t_low_default), (m.group(3) or 'low') + (' (raw weights)' if m.group(4) else '')

def _retained_pct(cfg, t_low):
    """% of convolutional coefficients and MB/round saved when the low band at t_low (or a random band of the same size) is retained."""
    st = ff.HistoFreqNet().state_dict(); convs = ff.conv_layer_names(st)
    bc = np.array([ff.WeightDCT.band_counts(st[k].shape, t_low, max(cfg.t_high, t_low), cfg.dct_axis) for k in convs]).sum(0)
    n = sum(p.numel() for p in st.values())
    return 100.0 * bc[0] / bc.sum(), 100.0 * bc[0] / n

def stage_retention(cfg, band_rel='results/band', fix_rel='results/fix'):
    """Retention-fraction curve and band ablation from the two bootstrap_runs.json files (no GPU)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    boots = {}
    for rel in (fix_rel, band_rel):
        p = os.path.join(cfg.project, rel, 'bootstrap_runs.json')
        if os.path.exists(p):
            for seed, res in json.load(open(p)).items():
                for split in ('heldout', 'local'):
                    for name, v in res.get(split, {}).items():
                        if isinstance(v, dict) and 'accuracy' in v and 'mean' in v['accuracy']:
                            boots.setdefault((int(seed), split), {})[name] = v['accuracy']
    if not boots: print('  no bootstrap_runs.json found'); return
    fig_dir = os.path.join(cfg.project, fix_rel, 'figures'); os.makedirs(fig_dir, exist_ok=True)
    seeds = sorted({s for s, _ in boots}); rows = []
    colors = {'low': 'C0', 'random': 'C1', 'high': 'C2', 'random (raw weights)': 'C3'}

    # 1) own-centre and unseen-centre accuracy against the fraction of coefficients each client keeps
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for si, seed in enumerate(seeds):
        for a, split, title in ((ax[0], 'local', 'own centre (personalized)'), (ax[1], 'heldout', 'unseen centre')):
            b = boots.get((seed, split), {}); off = (si - (len(seeds) - 1) / 2) * 0.6
            pts = {}
            for name, acc in b.items():
                v = _parse_variant(name, cfg.t_low)
                if v is None or v[0] != cfg.warm_start: continue
                pct, _ = _retained_pct(cfg, v[1]); pts.setdefault(v[2], []).append((pct, acc))
                rows.append((seed, split, name, v[2], v[1], pct, acc['mean'], acc['lo'], acc['hi']))
            if 'fedavg' in b:
                pts.setdefault('fedavg', []).append((0.0, b['fedavg']))
                rows.append((seed, split, 'fedavg', 'none', 0.0, 0.0, b['fedavg']['mean'], b['fedavg']['lo'], b['fedavg']['hi']))
            for band, lst in pts.items():
                lst.sort(); x = np.array([p for p, _ in lst]) + off; m = np.array([q['mean'] for _, q in lst])
                err = np.array([[q['mean'] - q['lo'] for _, q in lst], [q['hi'] - q['mean'] for _, q in lst]])
                if band == 'fedavg':
                    a.errorbar(x, m, yerr=err, fmt='s', color='k', capsize=3, label=f'FedAvg (0% retained), seed {seed}' if si == 0 else None)
                else:
                    a.errorbar(x, m, yerr=err, fmt='o-' if len(lst) > 1 else 'o', color=colors.get(band, 'C3'), capsize=3,
                               alpha=1.0 if si == 0 else 0.55, label=f'{band} band, seed {seed}')
            if split == 'local' and 'fedavg_ft1' in b:
                ft = b['fedavg_ft1']; a.axhline(ft['mean'], color='gray', ls='--', lw=1, label='FedAvg + 1 epoch fine-tune' if si == 0 else None)
                a.axhspan(ft['lo'], ft['hi'], color='gray', alpha=0.12)
            a.set_xlabel('% of conv coefficients kept local'); a.set_ylabel('accuracy (95% patient-bootstrap CI)'); a.set_title(title)
    for a in ax: a.legend(fontsize=6.5); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'retention_fraction.png'), dpi=200); fig.savefig(os.path.join(fig_dir, 'retention_fraction.pdf')); plt.close(fig)
    with open(os.path.join(fig_dir, 'retention_fraction.csv'), 'w') as f:
        f.write('seed,split,method,band,t_low,pct_conv_retained,acc_mean,acc_lo,acc_hi\n')
        for r in rows: f.write(','.join(str(v) for v in r) + '\n')
    print(f'  saved {fig_dir}/retention_fraction.{{png,pdf,csv}}')

    # 2) band ablation at the default fraction: which coefficients are kept does not matter
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for a, split, title in ((ax[0], 'local', 'own centre (personalized)'), (ax[1], 'heldout', 'unseen centre')):
        labels, means, errs, cols = [], [], [], []
        for seed in seeds:
            b = boots.get((seed, split), {})
            for name, acc in b.items():
                v = _parse_variant(name, cfg.t_low)
                if v is None or v[0] != cfg.warm_start or abs(v[1] - cfg.t_low) > 1e-9: continue
                labels.append(f'{v[2]}\nseed {seed}'); means.append(acc['mean']); errs.append([acc['mean'] - acc['lo'], acc['hi'] - acc['mean']]); cols.append(colors.get(v[2], 'C3'))
        if not means: continue
        x = np.arange(len(means)); a.bar(x, means, yerr=np.array(errs).T, color=cols, capsize=3)
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=7); a.set_title(title); a.set_ylabel('accuracy')
        lo = min(m - e[0] for m, e in zip(means, errs)); hi = max(m + e[1] for m, e in zip(means, errs)); a.set_ylim(lo - (hi - lo) * 0.5, hi + (hi - lo) * 0.5)
        for seed, ls in zip(seeds, ('--', ':')):
            b = boots.get((seed, split), {})
            ref = 'fedavg_ft1' if split == 'local' else 'fedavg'
            if ref in b: a.axhline(b[ref]['mean'], color='gray', ls=ls, lw=1, label=f'{"FedAvg + fine-tune" if ref == "fedavg_ft1" else "FedAvg"}, seed {seed}')
        a.legend(fontsize=6.5)
    fig.suptitle(f'retained band at t_low={cfg.t_low} (same number of coefficients), warm-up {cfg.warm_start}', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'band_ablation.png'), dpi=200); fig.savefig(os.path.join(fig_dir, 'band_ablation.pdf')); plt.close(fig)
    print(f'  saved {fig_dir}/band_ablation.{{png,pdf}}')

def stage_datafig(cfg, seeds, n_tsne=600):
    """Figures that need the patches: per-centre mean RGB (Figure 4) and a t-SNE of penultimate features under the
    FedAvg global model and the FedFreq personalized models (Figure 9), one fold per call (cfg.folds[0])."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ff.maybe_mount_drive(cfg); device = ff.get_device(); fig_dir = os.path.join(cfg.out_dir, 'figures'); os.makedirs(fig_dir, exist_ok=True)
    fold = cfg.folds[0]; fd = ff.FoldData(cfg, fold, device)
    # per-centre mean RGB over the validation splits (+ test centre), in [0,1]
    rgb = {}
    for c, ds in fd.client_val_ds.items(): rgb[c] = ds.images.reshape(-1, 3).mean(0) / 255.0
    rgb[fold] = fd.test_ds.images[::20].reshape(-1, 3).mean(0) / 255.0
    cs = sorted(rgb); M = np.array([rgb[c] for c in cs])
    fig, a = plt.subplots(figsize=(5, 3)); w = 0.25
    for j, (ch, col) in enumerate(zip('RGB', ('tab:red', 'tab:green', 'tab:blue'))): a.bar(np.arange(len(cs)) + (j - 1) * w, M[:, j], w, label=ch, color=col)
    a.set_xticks(range(len(cs))); a.set_xticklabels([f'C{c}' for c in cs]); a.set_ylabel('mean intensity'); a.set_ylim(0.3, 0.9); a.legend(fontsize=8)
    a.set_title(f'per-centre colour (max per-channel gap {np.ptp(M, 0).max():.3f})'); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'stain_rgb.png'), dpi=200); fig.savefig(os.path.join(fig_dir, 'stain_rgb.pdf')); plt.close(fig)
    json.dump({f'C{c}': rgb[c].tolist() for c in cs}, open(os.path.join(fig_dir, 'stain_rgb.json'), 'w'), indent=1)
    print('  per-centre mean RGB: ' + '  '.join(f'C{c}={np.round(rgb[c], 3).tolist()}' for c in cs))
    # t-SNE of penultimate features: FedAvg global vs FedFreq personalized, same patches
    n_tsne = min(n_tsne, min(len(ds) for ds in fd.client_val_ds.values()))
    seed = seeds[0]; dirs = ff._run_dirs(cfg, fold, seed)
    fa = dirs.get('fedavg'); fq = dirs.get(f'fedfreq_train_ws{cfg.warm_start}')
    if not (fa and fq): print('  t-SNE skipped: need fedavg and fedfreq_train runs for this fold/seed'); return
    from sklearn.manifold import TSNE
    g_avg = ff._load_pt(os.path.join(fa, 'global_last.pt')); g_fq = ff._load_pt(os.path.join(fq, 'global_last.pt')); low = ff._load_pt(os.path.join(fq, 'client_low.pt'))
    convs = ff.conv_layer_names(g_fq); model = ff.HistoFreqNet().to(device)
    feats = {}; pool = getattr(model, 'pool', None) or model.avgpool; hook = pool.register_forward_hook(lambda m, i, o: feats.__setitem__('f', o.flatten(1).float().cpu()))
    def embed(st, loader, c):
        model.load_state_dict({k: v.to(device) for k, v in st.items()})
        ff.calibrate_bn(model, fd.bn_client(c, cfg.bn_source_local)[0], device); model.eval(); out = []
        with torch.no_grad():
            for i, (bx, by) in enumerate(loader):
                with ff.autocast_ctx(device): model(bx)
                out.append(feats['f'])
                if sum(len(o) for o in out) >= n_tsne: break
        return torch.cat(out)[:n_tsne].numpy()
    pts = {'before FL (random init)': [], 'FedAvg global': [], 'FedFreq personalized': []}; labels = []
    ff.set_seed(0); init = {k: v.cpu() for k, v in ff.HistoFreqNet().state_dict().items()}
    for c in fd.train_centers:
        loader = fd.client_val_loader[c]
        pts['before FL (random init)'].append(embed(init, loader, c)); pts['FedAvg global'].append(embed(g_avg, loader, c))
        pts['FedFreq personalized'].append(embed(ff.personalize(g_fq, low[c], convs, cfg), loader, c)); labels += [c] * n_tsne
    hook.remove(); labels = np.array(labels)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, (name, P) in zip(ax, pts.items()):
        Z = TSNE(n_components=2, init='pca', random_state=0, perplexity=min(30, n_tsne - 1)).fit_transform(np.concatenate(P))
        for c in fd.train_centers: a.scatter(*Z[labels == c].T, s=3, label=f'C{c}')
        a.set_title(name); a.set_xticks([]); a.set_yticks([])
    ax[0].legend(markerscale=4, fontsize=8); fig.suptitle(f'penultimate features, fold {fold} (test C{fold}), {n_tsne} val patches per centre', fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'tsne_fold{fold}.png'), dpi=200); fig.savefig(os.path.join(fig_dir, f'tsne_fold{fold}.pdf')); plt.close(fig)
    print(f'  saved {fig_dir}/tsne_fold{fold}.png')

# ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['band', 'divergence', 'figures', 'datafig', 'arch', 'retention'])
    ap.add_argument('--arch', choices=list(ARCH), default='histofreqnet'); ap.add_argument('--no_pretrained', action='store_true')
    ap.add_argument('--pointwise', action='store_true', help='arch: also decompose 1x1 convolutions')
    ap.add_argument('--band', choices=['low', 'high', 'random'], default='high'); ap.add_argument('--raw', action='store_true', help='band: retain raw weights, no DCT')
    ap.add_argument('--source', default='fedavg'); ap.add_argument('--div_epochs', type=int, default=1)
    a, rest = ap.parse_known_args(argv)
    _, cfg = ff.parse(['run'] + rest)          # reuse every fedfreq_fix flag
    seeds = ff.parse(['run'] + rest)[0].seeds
    cli = ff.parse(['run'] + rest)[0]
    if a.stage in ('figures', 'datafig', 'band', 'divergence') and a.arch != 'histofreqnet':
        apply_arch(a.arch, not a.no_pretrained, a.pointwise)
        for k, v in ARCH[a.arch].items():
            if k != 'opt' and getattr(cli, k, None) is None: setattr(cfg, k, v)
    if a.stage == 'band':
        if cfg.out_rel == ff.Config.out_rel: cfg.out_rel = 'results/band'
        stage_band(cfg, a.band, seeds, a.raw)
    elif a.stage == 'divergence': stage_divergence(cfg, seeds, a.source, a.div_epochs)
    elif a.stage == 'figures': stage_figures(cfg, seeds)
    elif a.stage == 'retention': stage_retention(cfg)
    elif a.stage == 'datafig': stage_datafig(cfg, seeds)
    elif a.stage == 'arch':
        if cfg.out_rel == ff.Config.out_rel: cfg.out_rel = f'results/arch_{a.arch}'
        stage_arch(cfg, a.arch, cli.methods, seeds, not a.no_pretrained, a.pointwise, cli)

if __name__ == '__main__':
    main()
