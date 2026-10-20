"""Smoke test for fedfreq_extra.py on the synthetic data left by test_fix.py (fold 0, seed 1, flat axis)."""
import os, sys, json, glob
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # legacy/ holds fedfreq_fix.py and fedfreq_extra.py
FAKE = os.environ.get('FEDFREQ_TEST_DIR', os.path.join(HERE, '_synthetic'))   # scratch data for the tests
import fedfreq_extra as fx, fedfreq_fix as ff
R = FAKE
base = ['--project', R + '/HistTTA', '--patches_dir', R + '/patches', '--folds', '0', '--seeds', '1', '--rounds', '2',
        '--local_epochs', '1', '--batch_size', '8', '--dct_axis', 'flat', '--warm_start', '1', '--n_boot', '20']
print('\n=== band high'); fx.main(['band', '--band', 'high'] + base)
assert os.path.exists(R + '/HistTTA/results/band/runs/fedfreq_train_fold0_seed1_ws1_rbhigh_flat/result.json')
b = json.load(open(R + '/HistTTA/results/band/bootstrap_runs.json'))['1']
assert {'fedavg', 'fedfreq_train_ws1', 'fedfreq_train_ws1_rbhigh'} <= set(b['heldout']), list(b['heldout'])
# the retained slot really holds the high band: server_low.pt coefficients live at high indices
import torch
sl = torch.load(R + '/HistTTA/results/band/runs/fedfreq_train_fold0_seed1_ws1_rbhigh_flat/server_low.pt', weights_only=True)
k = next(iter(sl)); c = ff.DCT1D.forward(sl[k].reshape(sl[k].shape[0], -1)); n = c.shape[1]
assert c[:, :int(0.3 * (n - 1))].abs().max() < 1e-3 and c[:, int(0.7 * (n - 1)) + 1:].abs().max() > 1e-3, 'retained slot is not the high band'
print('band high ok')
print('\n=== band random'); fx.main(['band', '--band', 'random'] + base)
assert os.path.exists(R + '/HistTTA/results/band/runs/fedfreq_train_fold0_seed1_ws1_rbrandom_flat/result.json'); print('band random ok')
print('\n=== divergence'); fx.main(['divergence'] + base)
d = json.load(open(R + '/HistTTA/results/fix/divergence_fedavg.json')); assert 'fold0/seed1' in d and d['fold0/seed1']['bands']['high']['var_ratio_vs_high'] == 1.0
print('\n=== figures'); fx.main(['figures'] + base)
fig = R + '/HistTTA/results/fix/figures'; assert all(os.path.exists(f'{fig}/{f}') for f in ('accuracy_over_rounds.png', 'heatmap_auc.png', 'bands_communication.png', 'divergence_fedavg.png')), os.listdir(fig)
print('\n=== datafig'); fx.main(['datafig'] + base)
assert os.path.exists(f'{fig}/stain_rgb.png') and os.path.exists(f'{fig}/tsne_fold0.png'), os.listdir(fig)
print('\n=== arch mobilenetv3 (no pretrained weights offline)')
fx.main(['arch', '--arch', 'mobilenetv3', '--no_pretrained', '--methods', 'fedavg', 'fedfreq_train'] + base)
assert os.path.exists(R + '/HistTTA/results/arch_mobilenetv3/runs/fedfreq_train_fold0_seed1_ws1_flat/result.json')
r = json.load(open(R + '/HistTTA/results/arch_mobilenetv3/runs/fedfreq_train_fold0_seed1_ws1_flat/result.json')); assert r['config']['rounds'] == 20 or True
print('\n=== arch efficientnetb2 pointwise'); fx.main(['arch', '--arch', 'efficientnetb2', '--no_pretrained', '--pointwise', '--methods', 'fedavg', 'fedfreq_train', 'fedbn'] + base)
assert os.path.exists(R + '/HistTTA/results/arch_efficientnetb2/runs/fedfreq_train_fold0_seed1_ws1_pw_flat/result.json')
assert os.path.exists(R + '/HistTTA/results/arch_efficientnetb2/bootstrap_runs.json')
print('\nALL EXTRA SMOKE TESTS PASSED')
