"""Smoke test: fedper / fedper4 methods and raw-weight random retention, on the synthetic data from test_fix.py."""
import os, sys, json, torch
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # legacy/ holds fedfreq_fix.py and fedfreq_extra.py
FAKE = os.environ.get('FEDFREQ_TEST_DIR', os.path.join(HERE, '_synthetic'))   # scratch data for the tests
import fedfreq_extra as fx, fedfreq_fix as ff
R = FAKE
base = ['--project', R + '/HistTTA', '--patches_dir', R + '/patches', '--folds', '0', '--seeds', '1', '--rounds', '2',
        '--local_epochs', '1', '--batch_size', '8', '--dct_axis', 'flat', '--warm_start', '1', '--n_boot', '20']
print('=== fedper, fedper4'); ff.main(['sweep', '--methods', 'fedper', 'fedper4', '--out_rel', 'results/fix'] + base)
for m in ('fedper', 'fedper4'):
    d = R + f'/HistTTA/results/fix/runs/{m}_fold0_seed1_flat'; assert os.path.exists(d + '/result.json'), m
    cb = torch.load(d + '/client_bn.pt', weights_only=True); ks = set(next(iter(cb.values())))
    assert ks == {'fc.weight', 'fc.bias'} if m == 'fedper' else all(k.startswith(('fc.', 'stage4.')) for k in ks) and len(ks) > 2, ks
    # heads differ between clients (they were never averaged)
    cs = list(cb); assert not torch.equal(cb[cs[0]]['fc.weight'], cb[cs[1]]['fc.weight']), 'heads were averaged'
ff.main(['localpred', '--out_rel', 'results/fix'] + base); ff.main(['bootstrap_runs', '--out_rel', 'results/fix'] + base)
b = json.load(open(R + '/HistTTA/results/fix/bootstrap_runs.json'))['1']
assert 'fedper' in b['local'] and 'fedper4' in b['local'] and 'diff_vs_fedper' in b['local'], list(b['local'])
print('fedper ok')
print('=== band random --raw'); fx.main(['band', '--band', 'random', '--raw'] + base)
d = R + '/HistTTA/results/band/runs/fedfreq_train_fold0_seed1_ws1_rbrandom_raw_flat'; assert os.path.exists(d + '/result.json')
# retained slot holds raw weights: identity transform means client_low entries equal masked raw weights (no DCT applied)
g = torch.load(d + '/global_last.pt', weights_only=True); cl = torch.load(d + '/client_low.pt', weights_only=True)
k = next(iter(next(iter(cl.values())))); low = next(iter(cl.values()))[k]
nz = (low.reshape(low.shape[0], -1).abs().sum(0) > 0); print('  retained raw columns:', int(nz.sum()), '/', nz.numel())
assert 0 < int(nz.sum()) < nz.numel()
# DCT restored after the stage
assert ff.DCT1D.forward is fx._orig_dct_fwd or ff.DCT1D.forward(torch.eye(4)).allclose(fx._orig_dct_fwd(torch.eye(4)))
bb = json.load(open(R + '/HistTTA/results/band/bootstrap_runs.json'))['1']; assert 'fedfreq_train_ws1_rbrandom_raw' in bb['local'], list(bb['local'])
assert fx._parse_variant('fedfreq_train_ws15_rbrandom_raw', 0.3) == (15, 0.3, 'random (raw weights)')
print('=== retention fig'); fx.main(['retention'] + base)
assert os.path.exists(R + '/HistTTA/results/fix/figures/retention_fraction.png')
print('\nNEW SMOKE TESTS PASSED')
