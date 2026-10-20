"""Smoke test for fedfreq_fix.py on CPU with synthetic data and fake checkpoints."""
import os, sys, json, shutil
import numpy as np, pandas as pd, torch
from PIL import Image
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # legacy/ holds fedfreq_fix.py and fedfreq_extra.py
FAKE = os.environ.get('FEDFREQ_TEST_DIR', os.path.join(HERE, '_synthetic'))   # scratch data for the tests
import fedfreq_fix as ff

ROOT = FAKE
shutil.rmtree(ROOT, ignore_errors=True)
PROJECT = os.path.join(ROOT, 'HistTTA'); PATCHES = os.path.join(ROOT, 'patches')
os.makedirs(os.path.join(PROJECT, 'metadata')); os.makedirs(os.path.join(PROJECT, 'results'))

# synthetic metadata: 5 centers, 4 patients each, 2 nodes, 6 patches per node -> 48 patches/center; split 0/1
rng = np.random.RandomState(0); rows = []
for c in range(5):
    for p in range(4):
        pid = c * 10 + p
        for node in range(2):
            for k in range(6):
                rows.append(dict(center=c, patient=pid, node=node, x_coord=k * 100, y_coord=node * 100,
                                 tumor=int(rng.rand() < 0.5), split=int(k >= 4), folder=f'patient_{pid:03d}_node_{node}'))
df = pd.DataFrame(rows); df.to_csv(os.path.join(PROJECT, 'metadata', 'metadata_enhanced.csv'), index=False)
for _, r in df.iterrows():
    d = os.path.join(PATCHES, r['folder']); os.makedirs(d, exist_ok=True)
    # center-dependent colour so BN calibration source matters
    base = np.array([150 + 20 * r['center'], 100, 180 - 15 * r['center']], dtype=np.uint8)
    img = np.clip(base + rng.randint(-30, 30, (96, 96, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(ff.patch_path(r, PATCHES))
json.dump({'mean': [0.6, 0.4, 0.6], 'std': [0.2, 0.2, 0.2], 'class_weights': [1.0, 1.0]},
          open(os.path.join(PROJECT, 'results', 'dataset_stats.json'), 'w'))

# fake checkpoints for fold 0
cfg = ff.Config(project=PROJECT, patches_dir=PATCHES, folds=(0,), rounds=2, local_epochs=1, batch_size=8,
                eval_batch_size=16, ft_epochs=1)
d = os.path.join(cfg.ckpt_base, 'fold_0'); os.makedirs(d)
torch.manual_seed(1); m = ff.HistoFreqNet(); st = {k: v.clone() for k, v in m.state_dict().items()}
for meth in ('fedfreq', 'fedavg', 'fedprox', 'fedbn', 'freqfed'):
    torch.save(st, os.path.join(d, f'{meth}_best.pt')); torch.save(st, os.path.join(d, f'{meth}_global.pt'))
convs = ff.conv_layer_names(st)
low = {c: {k: ff.WeightDCT.decompose(st[k] * (1 + 0.1 * c), 0.3, 0.7, 'flat')[0] for k in convs} for c in (1, 2, 3, 4)}
torch.save(low, os.path.join(d, 'fedfreq_low_freq.pt'))

# --- unit checks on the DCT ---
w = torch.randn(8, 16, 3, 3)
for ax in ('flat', 'cin'):
    l, mi, h = ff.WeightDCT.decompose(w, 0.3, 0.7, ax)
    assert (ff.WeightDCT.recompose(l, mi, h) - w).abs().max() < 1e-4, ax
    # bands orthogonal: decomposing the low band again returns itself
    l2 = ff.WeightDCT.decompose(l, 0.3, 0.7, ax)[0]
    assert (l2 - l).abs().max() < 1e-4, ax
# flat axis reproduces Fed.ipynb's reshape (spatial fastest)
flat = w.reshape(8, 144); assert torch.allclose(ff.DCT1D.inverse(ff.DCT1D.forward(flat)), flat, atol=1e-4)
# linearity: average of bands == band of average
ws = [torch.randn(8, 16, 3, 3) for _ in range(3)]
avg = sum(ws) / 3
assert (ff.WeightDCT.decompose(avg, 0.3, 0.7, 'flat')[0] - sum(ff.WeightDCT.decompose(x, 0.3, 0.7, 'flat')[0] for x in ws) / 3).abs().max() < 1e-4
print('DCT checks ok; band counts flat', ff.WeightDCT.band_counts((8, 16, 3, 3), 0.3, 0.7, 'flat'), 'cin', ff.WeightDCT.band_counts((8, 16, 3, 3), 0.3, 0.7, 'cin'))

# --- metrics check ---
probs = np.array([0.1, 0.9, 0.8, 0.2, 0.6, 0.4]); labels = np.array([0, 1, 1, 0, 0, 1])
mm = ff.metrics_from(probs, labels, 5); print('metrics sample:', {k: mm[k] for k in ('accuracy', 'auc', 'ece', 'mce', 'bin_counts')})
assert abs(mm['auc'] - ff._fast_auc(labels, probs)) < 1e-9

# --- stages ---
print('\n--- check'); assert ff.stage_check(cfg)
print('\n--- reeval'); r = ff.stage_reeval(cfg)
print('\n--- bootstrap'); b = ff.stage_bootstrap(cfg, ck='best', src='client_val', n_boot=20)
print('\n--- finetune'); f = ff.stage_finetune(cfg)
print('\n--- run fedavg'); ff.stage_run(cfg, 'fedavg', 0, 1)
print('\n--- run fedprox'); ff.stage_run(cfg, 'fedprox', 0, 1); ff.stage_run(cfg, 'fedbn', 0, 1)
print('\n--- run fedfreq_train ws1'); cfg.warm_start = 1; ff.stage_run(cfg, 'fedfreq_train', 0, 1)
print('\n--- run fedfreq_train cin'); cfg.dct_axis = 'cin'; ff.stage_run(cfg, 'fedfreq_train', 0, 1)
print('\n--- resume (should skip to final)'); ff.stage_run(cfg, 'fedfreq_train', 0, 1)
print('\n--- tables'); ff.stage_tables(cfg)
print('\n--- localpred'); cfg.dct_axis = 'flat'; ff.stage_localpred(cfg, [1])
print('\n--- localpred again (all skip)'); ff.stage_localpred(cfg, [1])
print('\n--- bootstrap_runs'); br = ff.stage_bootstrap_runs(cfg, [1], n_boot=20)
r = br['1']; assert 'fedavg_ft1' in r['local'] and 'fedfreq_train_ws1' in r['local'] and 'fedbn' in r['heldout']
assert 'diff_vs_fedavg_ft1' in r['local'] and 'fedfreq_train_ws1' in r['local']['diff_vs_fedavg_ft1']
import glob; assert len(glob.glob(os.path.join(cfg.out_dir, 'runs', 'fedavg_fold0_seed1_flat', 'pred_local_*'))) == 12
print('localpred/bootstrap_runs ok')

# invariant: fedfreq_train server never updates its low band after warm start
tag = 'fedfreq_train_fold0_seed1_ws1_cin'; rd = os.path.join(cfg.out_dir, 'runs', tag)
sl = torch.load(os.path.join(rd, 'server_low.pt'), weights_only=True); g = torch.load(os.path.join(rd, 'global_last.pt'), weights_only=True)
k = convs[3]; gl = ff.WeightDCT.decompose(g[k], 0.3, 0.7, 'cin')[0]
assert (gl - sl[k]).abs().max() < 1e-4, 'server low band drifted'
print('\nserver low band frozen: ok')
print('\nALL SMOKE TESTS PASSED')
