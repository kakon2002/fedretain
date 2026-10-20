"""
fedfreq_fix.py  -  corrected evaluation and training for FedFreq (Camelyon17, HistoFreqNet)

One file. Drop it next to the Drive mount in Colab and run stages one at a time:

    !python fedfreq_fix.py check                       # what is on Drive, what is missing
    !python fedfreq_fix.py reeval                      # re-score the 15 saved checkpoints under corrected BN protocols
    !python fedfreq_fix.py bootstrap                   # patient-level bootstrap on the saved predictions
    !python fedfreq_fix.py finetune --ft_epochs 1      # FedAvg + local fine-tuning baseline (the one that tests the personalization win)
    !python fedfreq_fix.py run --method fedavg --fold 0 --seed 1
    !python fedfreq_fix.py run --method fedfreq_train --fold 0 --seed 1 --warm_start 3   # Phase 3: real low-band retention
    !python fedfreq_fix.py tables                      # collect every result JSON into one table
    !python fedfreq_fix.py localpred --seeds 42 43     # per-client predictions (+ fine-tune baseline) for finished runs
    !python fedfreq_fix.py bootstrap_runs --seeds 42 43   # patient-level paired bootstrap over the corrected runs

Everything is resumable: each stage writes JSON to RESULTS/fix/ and skips work it already finds there.

What this fixes relative to Fed.ipynb (review items in brackets):
  C3  BN calibration source is explicit; the held-out center is never used for calibration or model selection
      unless you ask for the legacy row ('test_center') for comparison.
  C4  WeightDCT has dct_axis='flat' (old behaviour, needed to read old checkpoints) and 'cin' (true channel axis).
  C6  Real MCE (max bin gap) with bin counts; the old cost-weighted error is reported under its own name.
  C7  One config, fixed rounds, no early stopping, config stored in every result.
  R5  Patient-level cluster bootstrap. McNemar removed.
  R6  Seed is an argument.
  R8  calibrate_bn uses cumulative averaging (momentum=None).
  C1/C2  'fedfreq_train' keeps the low band on the client during training; the server holds a frozen low band
      after warm_start rounds and only ever receives mid/high.
"""

import os, json, time, math, argparse
VERSION = "2026-10-20g"
from dataclasses import dataclass, asdict
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, confusion_matrix, brier_score_loss)

# ======================================================================
#  CONFIG
# ======================================================================

@dataclass
class Config:
    project: str = '/content/drive/MyDrive/HistTTA'
    patches_dir: str = '/content/data/camelyon17/data/camelyon17_v1.0/patches'
    metadata_rel: str = 'metadata/metadata_enhanced.csv'
    ckpt_rel: str = 'checkpoints/camelyon17_histofreqnet'
    results_rel: str = 'results'
    out_rel: str = 'results/fix'

    folds: tuple = (0, 1, 2)
    t_low: float = 0.3
    t_high: float = 0.7
    dct_axis: str = 'flat'            # 'flat' = old behaviour (reads old checkpoints); 'cin' = true channel axis

    rounds: int = 25
    local_epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    eval_batch_size: int = 512
    weighting: str = 'proportional'   # 'proportional' or 'equal'
    seed: int = 42

    bn_source_heldout: str = 'client_val'   # 'checkpoint' | 'client_val' | 'test_center'(legacy, for the comparison row only)
    bn_source_local: str = 'client_train'   # 'client_train' | 'client_val'(legacy)
    bn_max_batches: int = 0                 # batches used for BN calibration; 0 = the whole loader
    bn_calib_patches: int = 20000           # reeval only: training patches per client loaded for BN calibration
    bn_max_batches_round: int = 100         # run: cap on calibration batches for the per-round log evaluations
    norm_source: str = 'train_centers'      # 'train_centers' = per-fold mean/std from the 4 training centres; 'dataset_stats' = legacy file
    ft_source: str = 'runs'                 # finetune: 'runs' = new equal-budget fedavg runs; 'checkpoints' = original submitted checkpoints
    train_cap: int = 0                      # >0: seeded subsample of each client's training split to this many patches (low-data experiment)
    n_bins: int = 10

    ft_epochs: int = 1
    local_only_epochs: int = 0              # >0 runs a local-only baseline (from scratch per client)

    warm_start: int = 0                     # fedfreq_train: rounds of plain FedAvg before low-band retention starts
    server_low: str = 'freeze'              # fedfreq_train: server low band after warm-up: 'freeze' (never updated) | 'avg' (mean of client low bands)
    retain_layers: str = 'all'              # fedfreq_train: which conv layers keep a client low band: 'all' | 'early' (stem, stage1, stage2) | 'late' (stage3, stage4)
    prox_mu: float = 0.01                   # fedprox: proximal coefficient

    @property
    def metadata(self): return os.path.join(self.project, self.metadata_rel)
    @property
    def ckpt_base(self): return os.path.join(self.project, self.ckpt_rel)
    @property
    def results_dir(self): return os.path.join(self.project, self.results_rel)
    @property
    def out_dir(self):
        d = os.path.join(self.project, self.out_rel); os.makedirs(d, exist_ok=True); return d


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autocast_ctx(device):
    if device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return torch.autocast('cpu', enabled=False)

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def maybe_mount_drive(cfg):
    if cfg.project.startswith('/content/drive') and not os.path.isdir(cfg.project):
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception:
            pass

# ======================================================================
#  DCT
# ======================================================================

class DCT1D:
    _cache = {}
    @staticmethod
    def matrix(n):
        if n not in DCT1D._cache:
            i = torch.arange(n, dtype=torch.float64)
            k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
            mat = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
            mat[0, :] = 1.0 / math.sqrt(n)
            DCT1D._cache[n] = mat.float()
        return DCT1D._cache[n]
    @staticmethod
    def forward(x):  return x @ DCT1D.matrix(x.shape[-1]).t()
    @staticmethod
    def inverse(x):  return x @ DCT1D.matrix(x.shape[-1])


class WeightDCT:
    """
    dct_axis='flat': reshape (C_out, C_in, K, K) -> (C_out, C_in*K*K) row-major and DCT along that axis.
                     This is what Fed.ipynb does. Spatial position is the fastest index.
    dct_axis='cin' : DCT over C_in separately for each (output filter, spatial position).
                     This is what the paper's text describes.
    Bands: index/(N-1) <= t_low -> low; < t_high -> mid; else high.
    """
    @staticmethod
    def _coef(weight, dct_axis):
        c_out, c_in, kh, kw = weight.shape
        if dct_axis == 'flat':
            flat = weight.reshape(c_out, c_in * kh * kw)
        elif dct_axis == 'cin':
            flat = weight.permute(0, 2, 3, 1).reshape(c_out * kh * kw, c_in)
        else:
            raise ValueError(dct_axis)
        return DCT1D.forward(flat)

    @staticmethod
    def _uncoef(coef, shape, dct_axis):
        c_out, c_in, kh, kw = shape
        w = DCT1D.inverse(coef)
        if dct_axis == 'flat':
            return w.reshape(c_out, c_in, kh, kw)
        return w.reshape(c_out, kh, kw, c_in).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def masks(n, t_low, t_high):
        idx = torch.arange(n, dtype=torch.float32) / max(n - 1, 1)
        low = idx <= t_low
        high = idx >= t_high
        mid = ~(low | high)
        return low, mid, high

    @staticmethod
    def decompose(weight, t_low=0.3, t_high=0.7, dct_axis='flat'):
        weight = weight.detach().cpu().float()
        if weight.dim() != 4 or weight.shape[2] * weight.shape[3] <= 1:
            z = torch.zeros_like(weight); return z.clone(), weight.clone(), z.clone()
        coef = WeightDCT._coef(weight, dct_axis)
        low, mid, high = WeightDCT.masks(coef.shape[-1], t_low, t_high)
        out = []
        for m in (low, mid, high):
            out.append(WeightDCT._uncoef(coef * m.float(), weight.shape, dct_axis))
        return tuple(out)

    @staticmethod
    def recompose(w_low, w_mid, w_high):
        return w_low + w_mid + w_high

    @staticmethod
    def n_coefficients(weight_shape, dct_axis):
        c_out, c_in, kh, kw = weight_shape
        return c_in * kh * kw if dct_axis == 'flat' else c_in

    @staticmethod
    def band_counts(weight_shape, t_low, t_high, dct_axis):
        n = WeightDCT.n_coefficients(weight_shape, dct_axis)
        low, mid, high = WeightDCT.masks(n, t_low, t_high)
        per_vec = (int(low.sum()), int(mid.sum()), int(high.sum()))
        vecs = weight_shape[0] if dct_axis == 'flat' else weight_shape[0] * weight_shape[2] * weight_shape[3]
        return tuple(v * vecs for v in per_vec)


def conv_layer_names(state):
    return [k for k, p in state.items() if p.dim() == 4 and p.shape[2] * p.shape[3] > 1]

# ======================================================================
#  MODEL  (identical module names to Fed.ipynb so checkpoints load)
# ======================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(w))))
        return x * w.view(b, c, 1, 1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                                          nn.BatchNorm2d(out_ch))
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return F.silu(out + self.shortcut(x))

class HistoFreqNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(inplace=True))
        self.stage1 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64, 1))
        self.stage2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128, 1))
        self.stage3 = nn.Sequential(ResBlock(128, 192, 2), ResBlock(192, 192, 1))
        self.stage4 = nn.Sequential(ResBlock(192, 256, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(256, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        return self.fc(self.dropout(self.pool(x).flatten(1)))

PERSONAL_KEY_METHODS = ('fedbn', 'fedper', 'fedper4')

def personal_keys(model, method):
    """state-dict keys that stay on the client: BN tensors (fedbn), the classifier (fedper), classifier + last
    stage (fedper4). The server averages everything else; the held-out model uses the averaged tensors."""
    if method == 'fedbn': return bn_keys(model)
    if method == 'fedper': return [k for k in model.state_dict() if k.startswith('fc.')]
    if method == 'fedper4': return [k for k in model.state_dict() if k.startswith(('fc.', 'stage4.'))]
    return []

def bn_keys(model):
    keys = set()
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for k in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'):
                keys.add(f'{name}.{k}')
    return keys

# ======================================================================
#  DATA
# ======================================================================

def patch_path(row, patches_dir):
    return os.path.join(patches_dir, row['folder'],
                        f"patch_patient_{int(row['patient']):03d}_node_{int(row['node'])}_x_{int(row['x_coord'])}_y_{int(row['y_coord'])}.png")

class Split:
    """One split of patches, preloaded as uint8 (N, 96, 96, 3), with the per-patch metadata the bootstrap needs."""
    def __init__(self, df, patches_dir, name):
        self.df = df.reset_index(drop=True)
        self.labels = self.df['tumor'].values.astype(np.int64)
        self.patient = self.df['patient'].values.astype(np.int64)
        self.node = self.df['node'].values.astype(np.int64)
        self.center = self.df['center'].values.astype(np.int64)
        self.images = np.zeros((len(self.df), 96, 96, 3), dtype=np.uint8)
        t0 = time.time(); ok = 0
        for i, row in self.df.iterrows():
            try:
                img = Image.open(patch_path(row, patches_dir)).convert('RGB')
                if img.size != (96, 96): img = img.resize((96, 96), Image.Resampling.BILINEAR)
                self.images[i] = np.asarray(img); ok += 1; img.close()
            except Exception:
                pass
        print(f'      {name}: {ok:,}/{len(self.df):,} patches in {time.time()-t0:.0f}s', flush=True)
    def __len__(self): return len(self.labels)


def load_metadata(cfg):
    df = pd.read_csv(cfg.metadata)
    need = ['center', 'split', 'tumor', 'patient', 'node', 'x_coord', 'y_coord', 'folder']
    missing = [c for c in need if c not in df.columns]
    if missing: raise RuntimeError(f'metadata is missing columns {missing}; have {list(df.columns)}')
    return df

def load_stats(cfg):
    with open(os.path.join(cfg.results_dir, 'dataset_stats.json')) as f:
        s = json.load(f)
    return s['mean'], s['std'], s['class_weights']


class Loader:
    """Batches over a Split held on `device` as uint8. Normalisation and augmentation happen on the device.
    Yields (x, y). `generator` drives the shuffle and is reseeded by calibrate_bn so BN statistics are deterministic."""
    def __init__(self, images, labels, mean, std, batch_size, device, shuffle=False, augment=False,
                 drop_last=False, order=None, seed=0):
        """`images` is a uint8 tensor already placed on its storage device (GPU when it fits, else host RAM);
        batches are moved to `device` (the compute device) as they are produced."""
        self.x = images if torch.is_tensor(images) else torch.from_numpy(images)
        self.y = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.device = device
        self.set_norm(mean, std)
        self.bs, self.shuffle, self.augment, self.drop_last = batch_size, shuffle, augment, drop_last
        self.order = None if order is None else torch.as_tensor(order, dtype=torch.long)
        self.generator = torch.Generator().manual_seed(seed)
    def set_norm(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
    def __len__(self):
        n = len(self.y); return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs
    def __iter__(self):
        n = len(self.y); sd = self.x.device
        if self.order is not None: idx = self.order.to(sd)
        elif self.shuffle: idx = torch.randperm(n, generator=self.generator).to(sd)
        else: idx = torch.arange(n, device=sd)
        for i in range(0, n, self.bs):
            b = idx[i:i + self.bs]
            if self.drop_last and len(b) < self.bs: break
            x = self.x[b].to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0).sub_(self.mean).div_(self.std)
            if self.augment:
                B = x.shape[0]
                m = (torch.rand(B, device=self.device) < 0.5).view(-1, 1, 1, 1); x = torch.where(m, x.flip(3), x)
                m = (torch.rand(B, device=self.device) < 0.5).view(-1, 1, 1, 1); x = torch.where(m, x.flip(2), x)
                k = torch.randint(0, 4, (B,), device=self.device)
                for r in (1, 2, 3):
                    sel = k == r
                    if bool(sel.any()): x[sel] = torch.rot90(x[sel], r, (2, 3))
            yield x.contiguous(memory_format=torch.channels_last), self.y[b.to(self.device)]


class FoldData:
    """Everything for one leave-one-centre-out fold, resident on `device` (about 13 GB with full training splits)."""
    def __init__(self, cfg, fold, device, need_train=True):
        """need_train: True = full training splits (run/finetune); 'calib' = a seeded cfg.bn_calib_patches sample
        per client for BN calibration only (reeval); False = none."""
        df = load_metadata(cfg); mean, std, self.class_weights = load_stats(cfg)
        self.cfg, self.fold, self.device = cfg, fold, device
        self.train_centers = [c for c in range(5) if c != fold]
        print(f'  Fold {fold}: test=C{fold}, train={self.train_centers}')
        bs, ebs = cfg.batch_size, cfg.eval_batch_size
        # patches live on the GPU when it has room (A100), otherwise in host RAM with per-batch transfer
        need_gb = (2.0 if need_train is True else 0.3) * 4 + 2.5
        store = device if device.type == 'cuda' and torch.cuda.mem_get_info()[0] / 1024**3 > need_gb else torch.device('cpu')
        if store.type == 'cpu' and device.type == 'cuda':
            print(f'  GPU has {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB free, need ~{need_gb:.0f}: patches stay in host RAM', flush=True)
        def up(split): return torch.from_numpy(split.images).to(store)

        self.test_ds = Split(df[df['center'] == fold], cfg.patches_dir, f'test C{fold}')
        tx = up(self.test_ds)
        self.test_loader = Loader(tx, self.test_ds.labels, mean, std, ebs, device)
        self.test_calib_loader = Loader(tx, self.test_ds.labels, mean, std, ebs, device, shuffle=True, seed=cfg.seed)
        # Fed.ipynb's test loader order (train split then val split, batch 512, unshuffled): only for the 'notebook' row
        sp = self.test_ds.df['split'].values
        self.test_notebook_loader = Loader(tx, self.test_ds.labels, mean, std, 512, device,
                                           order=np.concatenate([np.where(sp == 0)[0], np.where(sp == 1)[0]]))

        self.client_train_ds, self.client_val_ds, self.client_calib_ds = {}, {}, {}
        self.client_train_loader, self.client_val_loader, self.client_bn_loader = {}, {}, {}
        vx, vy = [], []
        for c in self.train_centers:
            self.client_val_ds[c] = Split(df[(df['center'] == c) & (df['split'] == 1)], cfg.patches_dir, f'val C{c}')
            x = up(self.client_val_ds[c]); vx.append(x); vy.append(self.client_val_ds[c].labels)
            self.client_val_loader[c] = Loader(x, self.client_val_ds[c].labels, mean, std, ebs, device)
            trdf = df[(df['center'] == c) & (df['split'] == 0)]
            if need_train == 'calib':
                trdf = trdf.sample(n=min(cfg.bn_calib_patches, len(trdf)), random_state=cfg.seed)
                self.client_calib_ds[c] = Split(trdf, cfg.patches_dir, f'train-sample C{c}')
                self.client_bn_loader[c] = Loader(up(self.client_calib_ds[c]), self.client_calib_ds[c].labels, mean, std, ebs, device, shuffle=True, seed=cfg.seed)
            elif need_train:
                if cfg.train_cap > 0: trdf = trdf.sample(n=min(cfg.train_cap, len(trdf)), random_state=cfg.seed)
                self.client_train_ds[c] = Split(trdf, cfg.patches_dir, f'train C{c}')
                tr = up(self.client_train_ds[c]); self.client_train_ds[c].images = None   # host copy not needed
                self.client_train_loader[c] = Loader(tr, self.client_train_ds[c].labels, mean, std, bs, device,
                                                     shuffle=True, augment=True, drop_last=True, seed=cfg.seed + c)
                self.client_bn_loader[c] = Loader(tr, self.client_train_ds[c].labels, mean, std, ebs, device, shuffle=True, seed=cfg.seed)
            else:
                self.client_bn_loader[c] = None
        self.pooled_val_loader = Loader(torch.cat(vx), np.concatenate(vy), mean, std, ebs, device, shuffle=True, seed=cfg.seed)

        # normalisation: per fold from the training centres only (never the held-out centre)
        self.mean, self.std = mean, std
        if cfg.norm_source == 'train_centers':
            srcs = {c: self.client_train_loader[c].x for c in self.client_train_loader} or \
                   {c: self.client_bn_loader[c].x for c in self.client_bn_loader if self.client_bn_loader[c] is not None} or \
                   {c: self.client_val_loader[c].x for c in self.client_val_loader}
            g = torch.Generator().manual_seed(cfg.seed); acc = torch.zeros(3, dtype=torch.float64); acc2 = torch.zeros(3, dtype=torch.float64); n = 0
            for x in srcs.values():
                idx = torch.randperm(len(x), generator=g)[:20000]
                for j in range(0, len(idx), 1000):
                    v = x[idx[j:j + 1000].to(x.device)].to(device).float().div_(255.0)
                    acc += v.sum(dim=(0, 1, 2)).double().cpu(); acc2 += (v ** 2).sum(dim=(0, 1, 2)).double().cpu(); n += v.shape[0] * 96 * 96
            self.mean = (acc / n).tolist(); self.std = (acc2 / n - (acc / n) ** 2).sqrt().tolist()
            for l in self.all_loaders(): l.set_norm(self.mean, self.std)
            print(f'  norm (train centres only): mean={np.round(self.mean, 4).tolist()} std={np.round(self.std, 4).tolist()}')
        if store.type == 'cuda': print(f'  data on GPU: {torch.cuda.memory_allocated() / 1024**3:.1f} GB', flush=True)

    def all_loaders(self):
        return [self.test_loader, self.test_calib_loader, self.test_notebook_loader, self.pooled_val_loader] + \
               list(self.client_val_loader.values()) + list(self.client_train_loader.values()) + \
               [l for l in self.client_bn_loader.values() if l is not None]

    def bn_heldout(self, source):
        """(loader, ema) for re-estimating BN statistics before scoring the held-out centre.
        ema=True reproduces Fed.ipynb's calibrate_bn exactly (file order, momentum 0.1)."""
        return {'client_val': (self.pooled_val_loader, False), 'test_center': (self.test_calib_loader, False),
                'notebook': (self.test_notebook_loader, True), 'checkpoint': (None, False)}[source]

    def bn_client(self, c, source):
        """(loader, ema) for a client's own model: its training patches (default) or its validation split (legacy)."""
        if source == 'client_train':
            if self.client_bn_loader[c] is None: raise RuntimeError('client training data not loaded; use need_train=True')
            return self.client_bn_loader[c], False
        return self.client_val_loader[c], False

# ======================================================================
#  TRAIN / CALIBRATE / EVALUATE
# ======================================================================

def local_train(model, loader, epochs, lr, wd, class_weights, device, prox_mu=0.0):
    """prox_mu > 0 adds FedProx's proximal term (mu/2)||w - w_global||^2 against the weights at entry."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    anchor = [p.detach().clone() for p in model.parameters()] if prox_mu > 0 else None
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            if bx.dim() == 4: bx = bx.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                loss = crit(model(bx), by)
            if anchor is not None:
                loss = loss + 0.5 * prox_mu * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def calibrate_bn(model, loader, device, max_batches=0, ema=False):
    """Reset BN running stats and re-estimate them over `loader`. (R8)
    ema=False: cumulative average over the whole loader (population statistics; order-independent).
    ema=True : Fed.ipynb's original procedure (momentum 0.1, one pass in loader order), kept only to
               reproduce the submitted numbers for the old-vs-new comparison."""
    if loader is None: return
    if getattr(loader, 'generator', None) is not None: loader.generator.manual_seed(12345)
    saved = {}
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            saved[m] = m.momentum; m.reset_running_stats(); m.momentum = 0.1 if ema else None
    model.train()
    for i, (bx, _) in enumerate(loader):
        if max_batches and i >= max_batches: break
        with autocast_ctx(device):
            model(bx.to(device, non_blocking=True))
    for m, mom in saved.items(): m.momentum = mom
    model.eval()


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); probs, labels = [], []
    for bx, by in loader:
        with autocast_ctx(device):
            logits = model(bx.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()); labels.append(by.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics_from(probs, labels, n_bins=10):
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    two = len(np.unique(labels)) > 1
    bounds = np.linspace(0, 1, n_bins + 1); ece = 0.0; mce = 0.0; counts = []
    for b in range(n_bins):
        m = (probs >= bounds[b]) & ((probs < bounds[b + 1]) if b < n_bins - 1 else (probs <= bounds[b + 1]))
        counts.append(int(m.sum()))
        if m.sum():
            gap = abs(probs[m].mean() - labels[m].mean())
            ece += m.sum() * gap; mce = max(mce, gap)
    return {
        'n': int(len(labels)),
        'accuracy': float((preds == labels).mean()),
        'auc': float(roc_auc_score(labels, probs)) if two else float('nan'),
        'pr_auc': float(average_precision_score(labels, probs)) if two else float('nan'),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'sensitivity': float(sens), 'specificity': float(spec),
        'wga': float(min(sens, spec)),
        'brier': float(brier_score_loss(labels, probs)),
        'ece': float(ece / len(labels)), 'mce': float(mce), 'bin_counts': counts,
        'cost_error_10to1': float((10 * fn + fp) / len(labels)),   # the quantity Fed.ipynb called ECM and the paper called MCE
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def evaluate(model, loader, device, bn, cfg):
    """bn = (calibration loader or None, ema flag) from FoldData.bn_heldout / bn_client."""
    calibrate_bn(model, bn[0], device, cfg.bn_max_batches, bn[1])
    probs, labels = predict(model, loader, device)
    return metrics_from(probs, labels, cfg.n_bins), probs, labels

# ======================================================================
#  AGGREGATION
# ======================================================================

def weighted_average(states, weights):
    total = float(sum(weights)); out = OrderedDict()
    for k in states[0]:
        if states[0][k].dtype in (torch.int64, torch.int32, torch.long):
            out[k] = states[0][k].clone()
        else:
            out[k] = sum(s[k].float() * (w / total) for s, w in zip(states, weights))
    return out

def personalize(global_state, client_low, conv_layers, cfg):
    """client's low band + global mid/high for every conv layer."""
    st = OrderedDict((k, v.clone()) for k, v in global_state.items())
    for k in conv_layers:
        _, gm, gh = WeightDCT.decompose(global_state[k], cfg.t_low, cfg.t_high, cfg.dct_axis)
        st[k] = WeightDCT.recompose(client_low[k], gm, gh)
    return st

# ======================================================================
#  STAGE: check
# ======================================================================

def stage_check(cfg):
    maybe_mount_drive(cfg)
    print('=' * 70); print('CHECK'); print('=' * 70)
    print(f'  project:  {cfg.project}  exists={os.path.isdir(cfg.project)}')
    ok = os.path.isdir(cfg.patches_dir)
    print(f'  patches:  {cfg.patches_dir}  exists={ok}' + ('' if ok else '   <- extract camleyn.zip first (old cell 0)'))
    try:
        df = load_metadata(cfg)
        print(f'  metadata: {len(df):,} rows; columns ok (patient, node present)')
        for c in range(5):
            d = df[df.center == c]
            print(f'    C{c}: train={int((d.split==0).sum()):,} val={int((d.split==1).sum()):,} '
                  f'patients={d.patient.nunique()} nodes={d.groupby(["patient","node"]).ngroups} tumor={d.tumor.mean():.3f}')
        row = df.iloc[0]; p = patch_path(row, cfg.patches_dir)
        print(f'  sample patch exists: {os.path.exists(p)}  ({p})')
    except Exception as e:
        ok = False; print(f'  metadata problem: {e}')
    try:
        mean, std, cw = load_stats(cfg); print(f'  dataset_stats.json: mean={np.round(mean,3).tolist()} class_weights={cw}')
    except Exception as e:
        ok = False; print(f'  dataset_stats.json problem: {e}')
    print('\n  checkpoints:')
    for fold in cfg.folds:
        d = os.path.join(cfg.ckpt_base, f'fold_{fold}')
        if not os.path.isdir(d):
            print(f'    fold_{fold}: MISSING'); ok = False; continue
        files = sorted(f for f in os.listdir(d) if f.endswith('.pt'))
        print(f'    fold_{fold}: ' + ', '.join(f'{f} ({os.path.getsize(os.path.join(d,f))/2**20:.0f}MB)' for f in files))
        for m in ('fedfreq', 'fedavg', 'fedprox', 'fedbn', 'freqfed'):
            for suf in ('_best.pt', '_global.pt'):
                if m + suf not in files: print(f'      missing {m}{suf}')
        if 'fedfreq_low_freq.pt' not in files: print('      missing fedfreq_low_freq.pt (needed for personalized eval)')
    # model/checkpoint compatibility
    try:
        d = os.path.join(cfg.ckpt_base, f'fold_{cfg.folds[0]}'); f = os.path.join(d, 'fedavg_best.pt')
        if os.path.exists(f):
            st = torch.load(f, map_location='cpu', weights_only=True)
            m = HistoFreqNet(); missing, unexpected = m.load_state_dict(st, strict=False)
            print(f'\n  checkpoint/model key match: missing={len(missing)} unexpected={len(unexpected)}')
            if missing or unexpected: ok = False; print('   ', missing[:5], unexpected[:5])
            n = sum(p.numel() for p in m.parameters()); print(f'  params: {n:,}')
            convs = conv_layer_names(st)
            for ax in ('flat', 'cin'):
                lo = sum(WeightDCT.band_counts(st[k].shape, cfg.t_low, cfg.t_high, ax)[0] for k in convs)
                tot = sum(st[k].numel() for k in convs)
                print(f'  dct_axis={ax}: {len(convs)} conv layers, low-band coeffs={lo:,} of conv params {tot:,} ({lo/tot*100:.1f}%), '
                      f'of all params {lo/n*100:.1f}%')
    except Exception as e:
        ok = False; print(f'  checkpoint load problem: {e}')
    print(f'\n  out dir: {cfg.out_dir}')
    print('\n  RESULT:', 'ready' if ok else 'fix the items above first')
    return ok

# ======================================================================
#  STAGE: reeval  (no training)
# ======================================================================

def _load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=True) if os.path.exists(path) else None

REEVAL_PROTOCOL = 3   # 3 = single device-resident loader (2026-10-17b); 2 = same protocol via DataLoader; rows from older protocols are recomputed

def stage_reeval(cfg, methods=('fedfreq', 'fedavg', 'fedprox', 'fedbn', 'freqfed'),
                 heldout_sources=('checkpoint', 'client_val', 'test_center', 'notebook'), redo=()):
    """redo: held-out sources to recompute even if a finished row already has them (e.g. ('notebook',))."""
    maybe_mount_drive(cfg); device = get_device()
    out_path = os.path.join(cfg.out_dir, 'reeval.json')
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    # rows written by an older protocol (first-40-batches calibration) are recomputed
    results = {k: v for k, v in results.items() if v.get('protocol') == REEVAL_PROTOCOL}
    for fold in cfg.folds:
        fd = FoldData(cfg, fold, device, need_train=('calib' if cfg.bn_source_local == 'client_train' else False))
        d = os.path.join(cfg.ckpt_base, f'fold_{fold}')
        extra = [m for m in ('fedfreq_v3', 'centralized') if os.path.exists(os.path.join(d, f'{m}_best.pt'))]
        for method in list(methods) + extra:
            key = f'fold{fold}/{method}'
            prev = results.get(key) if results.get(key, {}).get('done') else None
            best = _load_pt(os.path.join(d, f'{method}_best.pt')); last = _load_pt(os.path.join(d, f'{method}_global.pt'))
            if best is None and last is None: print(f'  {key}: no checkpoint'); continue
            todo = [(ck, src) for ck in ('best', 'last') for src in heldout_sources
                    if prev is None or src in redo or f'{ck}/{src}' not in prev['heldout']]
            todo = [(ck, src) for ck, src in todo if (best if ck == 'best' else last) is not None]
            if prev is not None and not todo: print(f'  skip {key}'); continue
            model = HistoFreqNet().to(device); res = prev if prev is not None else {'heldout': {}, 'local': {}}
            convs = conv_layer_names(best if best is not None else last)
            # held-out center, best and last checkpoints, each BN source
            for ck_name, ck in (('best', best), ('last', last)):
                if ck is None: continue
                for src in heldout_sources:
                    if (ck_name, src) not in todo: continue
                    model.load_state_dict({k: v.to(device) for k, v in ck.items()})
                    m, probs, labels = evaluate(model, fd.test_loader, device, fd.bn_heldout(src), cfg)
                    res['heldout'][f'{ck_name}/{src}'] = m
                    np.savez_compressed(os.path.join(cfg.out_dir, f'pred_fold{fold}_{method}_{ck_name}_{src}.npz'),
                                        probs=probs, labels=labels, patient=fd.test_ds.patient, node=fd.test_ds.node,
                                        center=fd.test_ds.center)
                    print(f'  {key} {ck_name:4s} bn={src:11s} acc={m["accuracy"]:.4f} auc={m["auc"]:.4f} '
                          f'spec={m["specificity"]:.4f} ece={m["ece"]:.4f} mce={m["mce"]:.4f}', flush=True)
            # per-client (personalized for fedfreq*, global for the rest), BN on client train, scored on client val
            base = last if last is not None else best
            low = None; low_file = None
            if method.startswith('fedfreq'):
                for cand in (f'{method}_low_freq.pt', 'fedfreq_low_freq.pt'):
                    if os.path.exists(os.path.join(d, cand)):
                        low = _load_pt(os.path.join(d, cand)); low_file = cand; break
            res['low_band_file'] = low_file
            for c in fd.train_centers:
                if prev is not None and str(c) in prev['local']: continue
                st = base
                if low is not None and c in low:
                    st = personalize(base, low[c], convs, cfg)
                model.load_state_dict({k: v.to(device) for k, v in st.items()})
                m, probs, labels = evaluate(model, fd.client_val_loader[c], device,
                                            fd.bn_client(c, cfg.bn_source_local), cfg)
                res['local'][str(c)] = m
                np.savez_compressed(os.path.join(cfg.out_dir, f'pred_fold{fold}_{method}_local_C{c}.npz'),
                                    probs=probs, labels=labels, patient=fd.client_val_ds[c].patient,
                                    node=fd.client_val_ds[c].node, center=fd.client_val_ds[c].center)
            res['local_mean_acc'] = float(np.mean([v['accuracy'] for v in res['local'].values()]))
            res['personalized'] = low is not None
            res['done'] = True; res['protocol'] = REEVAL_PROTOCOL; res['config'] = asdict(cfg)
            results[key] = res
            json.dump(results, open(out_path, 'w'), indent=1)
            print(f'  {key} local mean acc={res["local_mean_acc"]:.4f} personalized={res["personalized"]}')
        del fd
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f'\n  saved {out_path}')
    return results

# ======================================================================
#  STAGE: bootstrap (patient-level, paired)
# ======================================================================

def _fast_auc(labels, probs):
    # rank-based AUC (Mann-Whitney), ties get average rank
    from scipy.stats import rankdata
    npos = int(labels.sum()); nneg = len(labels) - npos
    if npos == 0 or nneg == 0: return float('nan')
    ranks = rankdata(probs, method='average')
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def stage_bootstrap(cfg, ck='best', src=None, n_boot=1000, ref='fedavg', seed=0):
    src = src or cfg.bn_source_heldout
    rng = np.random.RandomState(seed)
    out = {'config': {'checkpoint': ck, 'bn_source': src, 'n_boot': n_boot, 'cluster': 'patient'}, 'per_fold': {}, 'pooled': {}}
    pooled = defaultdict(list); pooled_meta = []
    for fold in cfg.folds:
        files = {}
        for f in os.listdir(cfg.out_dir):
            if f.startswith(f'pred_fold{fold}_') and f.endswith(f'_{ck}_{src}.npz'):
                method = f[len(f'pred_fold{fold}_'):-len(f'_{ck}_{src}.npz')]
                files[method] = np.load(os.path.join(cfg.out_dir, f))
        if not files: print(f'  fold {fold}: no predictions for {ck}/{src}; run reeval first'); continue
        meta = next(iter(files.values()))
        labels = meta['labels']; patient = meta['patient']
        probs = {m: files[m]['probs'] for m in files}
        out['per_fold'][str(fold)] = _cluster_boot(labels, patient, probs, rng, n_boot, ref)
        for m in probs: pooled[m].append(probs[m])
        pooled_meta.append((labels, patient + 1000 * fold))   # keep patients distinct across folds
        print(f'  fold {fold} ({out["per_fold"][str(fold)]["n_patients"]} patients): ' +
              ', '.join(f'{m}={out["per_fold"][str(fold)][m]["accuracy"]["mean"]:.4f}' for m in probs))
    if pooled:
        common = [m for m in pooled if len(pooled[m]) == len(pooled_meta)]
        labels = np.concatenate([x[0] for x in pooled_meta]); patient = np.concatenate([x[1] for x in pooled_meta])
        out['pooled'] = _cluster_boot(labels, patient, {m: np.concatenate(pooled[m]) for m in common}, rng, n_boot, ref)
        print(f'\n  pooled over folds, {out["pooled"]["n_patients"]} patients (patient-level bootstrap, 95% CI):')
        for m, r in out['pooled'].items():
            if m == 'diff_vs_' + ref or m == 'n_patients': continue
            print(f'    {m:12s} acc {r["accuracy"]["mean"]:.4f} [{r["accuracy"]["lo"]:.4f}, {r["accuracy"]["hi"]:.4f}]  '
                  f'auc {r["auc"]["mean"]:.4f} [{r["auc"]["lo"]:.4f}, {r["auc"]["hi"]:.4f}]')
        for m, r in out['pooled'].get('diff_vs_' + ref, {}).items():
            print(f'    {m:12s} - {ref}: dAcc {r["accuracy"]["mean"]:+.4f} [{r["accuracy"]["lo"]:+.4f}, {r["accuracy"]["hi"]:+.4f}]  '
                  f'dAUC {r["auc"]["mean"]:+.4f} [{r["auc"]["lo"]:+.4f}, {r["auc"]["hi"]:+.4f}]')
    path = os.path.join(cfg.out_dir, f'bootstrap_{ck}_{src}.json'); json.dump(out, open(path, 'w'), indent=1)
    print(f'  saved {path}')
    return out

def _cluster_boot(labels, patient, probs, rng, n_boot, ref):
    groups = defaultdict(list)
    for i, p in enumerate(patient): groups[int(p)].append(i)
    gids = list(groups); gidx = [np.array(groups[g]) for g in gids]
    methods = list(probs)
    acc = {m: np.zeros(n_boot) for m in methods}; auc = {m: np.zeros(n_boot) for m in methods}
    for b in range(n_boot):
        pick = rng.randint(0, len(gids), len(gids))
        idx = np.concatenate([gidx[k] for k in pick]); y = labels[idx]
        for m in methods:
            p = probs[m][idx]; acc[m][b] = np.mean((p >= 0.5) == y); auc[m][b] = _fast_auc(y, p)
    def summ(a): return {'mean': float(np.nanmean(a)), 'lo': float(np.nanpercentile(a, 2.5)), 'hi': float(np.nanpercentile(a, 97.5))}
    res = {m: {'accuracy': summ(acc[m]), 'auc': summ(auc[m]), 'point': {'accuracy': float(np.mean((probs[m] >= 0.5) == labels)), 'auc': _fast_auc(labels, probs[m])}} for m in methods}
    for r in ([ref] if isinstance(ref, str) else ref):
        if r in methods:
            res['diff_vs_' + r] = {m: {'accuracy': summ(acc[m] - acc[r]), 'auc': summ(auc[m] - auc[r])} for m in methods if m != r}
    res['n_patients'] = len(gids)
    return res

# ======================================================================
#  STAGE: finetune baseline (FedAvg + local fine-tuning; optional local-only)
# ======================================================================

def stage_finetune(cfg):
    maybe_mount_drive(cfg); device = get_device()
    tag = f'finetune_e{cfg.ft_epochs}' + ('' if cfg.ft_source == 'checkpoints' else f'_runs_seed{cfg.seed}')
    out_path = os.path.join(cfg.out_dir, f'{tag}.json')
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for fold in cfg.folds:
        key = f'fold{fold}'
        if key in results and results[key].get('done'): print(f'  skip {key}'); continue
        if cfg.ft_source == 'checkpoints':
            d = os.path.join(cfg.ckpt_base, f'fold_{fold}')
            base = _load_pt(os.path.join(d, 'fedavg_global.pt')) or _load_pt(os.path.join(d, 'fedavg_best.pt'))
        else:
            base = _load_pt(os.path.join(cfg.out_dir, 'runs', run_tag(cfg, 'fedavg', fold, cfg.seed), 'global_last.pt'))
        if base is None: print(f'  {key}: no fedavg model for ft_source={cfg.ft_source} seed={cfg.seed}'); continue
        fd = FoldData(cfg, fold, device)
        model = HistoFreqNet().to(device); res = {'fedavg_finetune': {}, 'local_only': {}}
        for c in fd.train_centers:
            set_seed(cfg.seed)
            model.load_state_dict({k: v.to(device) for k, v in base.items()})
            local_train(model, fd.client_train_loader[c], cfg.ft_epochs, cfg.lr, cfg.weight_decay, fd.class_weights, device)
            m, _, _ = evaluate(model, fd.client_val_loader[c], device, fd.bn_client(c, cfg.bn_source_local), cfg)
            res['fedavg_finetune'][str(c)] = m
            print(f'  fold {fold} C{c} fedavg+ft({cfg.ft_epochs}ep): acc={m["accuracy"]:.4f} auc={m["auc"]:.4f}', flush=True)
            if cfg.local_only_epochs > 0:
                set_seed(cfg.seed); model = HistoFreqNet().to(device)
                local_train(model, fd.client_train_loader[c], cfg.local_only_epochs, cfg.lr, cfg.weight_decay, fd.class_weights, device)
                m, _, _ = evaluate(model, fd.client_val_loader[c], device, fd.bn_client(c, cfg.bn_source_local), cfg)
                res['local_only'][str(c)] = m
                print(f'  fold {fold} C{c} local-only({cfg.local_only_epochs}ep): acc={m["accuracy"]:.4f} auc={m["auc"]:.4f}', flush=True)
        res['fedavg_finetune_mean_acc'] = float(np.mean([v['accuracy'] for v in res['fedavg_finetune'].values()]))
        if res['local_only']: res['local_only_mean_acc'] = float(np.mean([v['accuracy'] for v in res['local_only'].values()]))
        res['done'] = True; res['config'] = asdict(cfg); results[key] = res
        json.dump(results, open(out_path, 'w'), indent=1)
        print(f'  fold {fold}: fedavg+ft mean local acc = {res["fedavg_finetune_mean_acc"]:.4f}')
        del fd
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f'  saved {out_path}')
    return results

# ======================================================================
#  STAGE: run  (corrected federated runner)
# ======================================================================

def run_tag(cfg, method, fold, seed):
    t = f'{method}_fold{fold}_seed{seed}'
    if method == 'fedfreq_train':
        t += f'_ws{cfg.warm_start}'
        if cfg.t_low != 0.3: t += f'_tl{cfg.t_low}'
        if cfg.server_low != 'freeze': t += f'_sl{cfg.server_low}'
        if cfg.retain_layers != 'all': t += f'_rl{cfg.retain_layers}'
    return t + f'_{cfg.dct_axis}'

def stage_run(cfg, method, fold, seed, fd=None):
    """
    method:
      fedavg          plain FedAvg
      fedfreq_train   low band retained on the client during training. Clients start each round from
                      client_low + global_mid_high. Server averages mid/high only. For the first `warm_start`
                      rounds plain FedAvg runs; after that the server's own low band is frozen and the server
                      never receives a client low band again.
      fedprox         FedAvg with the proximal term (cfg.prox_mu) in local training.
      fedbn           BatchNorm parameters (affine + running stats) stay on each client; the server averages
                      everything else. Client models keep their own BN; the held-out model takes the averaged
                      non-BN weights with BN affine averaged and running stats re-estimated like every method.
      fedper          FedPer: the classifier head stays on each client from round 1; the server averages the body.
      fedper4         FedPer variant keeping the last stage and the head local (about 40% of the parameters).
    Fixed rounds, no early stopping. Selection: mean client-val accuracy (never the test center).
    """
    maybe_mount_drive(cfg); device = get_device()
    tag = run_tag(cfg, method, fold, seed)
    run_dir = os.path.join(cfg.out_dir, 'runs', tag); os.makedirs(run_dir, exist_ok=True)
    prog_path = os.path.join(run_dir, 'progress.json')
    if os.path.exists(os.path.join(run_dir, 'result.json')):
        print(f'  skip {tag} (result.json exists)'); return json.load(open(os.path.join(run_dir, 'result.json')))
    if fd is None:
        fd = FoldData(cfg, fold, device)
    cids = fd.train_centers
    set_seed(seed)
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
    model = HistoFreqNet().to(device).to(memory_format=torch.channels_last)
    global_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    convs = conv_layer_names(global_state)
    if cfg.retain_layers == 'early': convs = [k for k in convs if k.split('.')[0] in ('stem', 'stage1', 'stage2')]
    elif cfg.retain_layers == 'late': convs = [k for k in convs if k.split('.')[0] in ('stage3', 'stage4')]
    client_low = {}; server_low = None
    bnk = personal_keys(model, method); client_bn = {}      # fedbn/fedper: each client's own private tensors
    history = []; start = 0; best_sel = -1.0
    if os.path.exists(prog_path):
        pr = json.load(open(prog_path)); history = pr['history']; start = pr['completed_rounds']; best_sel = pr['best_sel']
        global_state = _load_pt(os.path.join(run_dir, 'global_last.pt')) or global_state
        client_low = _load_pt(os.path.join(run_dir, 'client_low.pt')) or {}
        server_low = _load_pt(os.path.join(run_dir, 'server_low.pt'))
        client_bn = _load_pt(os.path.join(run_dir, 'client_bn.pt')) or {}
        print(f'  resuming {tag} at round {start}')
    def client_start(c, rnd):
        if method == 'fedfreq_train' and c in client_low and rnd >= cfg.warm_start:
            return personalize(global_state, client_low[c], convs, cfg)
        if method in PERSONAL_KEY_METHODS and c in client_bn:
            st = OrderedDict((k, v.clone()) for k, v in global_state.items()); st.update(client_bn[c]); return st
        return global_state
    for rnd in range(start, cfg.rounds):
        t0 = time.time(); states, weights = [], []
        retaining = method == 'fedfreq_train' and rnd >= cfg.warm_start
        for c in cids:
            model.load_state_dict({k: v.to(device) for k, v in client_start(c, rnd).items()})
            states.append(local_train(model, fd.client_train_loader[c], cfg.local_epochs, cfg.lr, cfg.weight_decay,
                                      fd.class_weights, device, prox_mu=cfg.prox_mu if method == 'fedprox' else 0.0))
            weights.append(len(fd.client_train_ds[c]) if cfg.weighting == 'proportional' else 1.0)
        # decompose client states
        for c, st in zip(cids, states):
            client_low[c] = {k: WeightDCT.decompose(st[k], cfg.t_low, cfg.t_high, cfg.dct_axis)[0] for k in convs}
            if method in PERSONAL_KEY_METHODS: client_bn[c] = {k: st[k].clone() for k in bnk}
        new_state = weighted_average(states, weights)
        if retaining:
            if cfg.server_low == 'freeze' and server_low is None:   # server low band fixed at its end-of-warm-up state
                server_low = {k: WeightDCT.decompose(global_state[k], cfg.t_low, cfg.t_high, cfg.dct_axis)[0] for k in convs}
            if cfg.server_low == 'freeze':
                total = float(sum(weights))
                for k in convs:
                    bands = [WeightDCT.decompose(st[k], cfg.t_low, cfg.t_high, cfg.dct_axis) for st in states]
                    new_state[k] = WeightDCT.recompose(server_low[k], sum(b[1] * (w / total) for b, w in zip(bands, weights)),
                                                       sum(b[2] * (w / total) for b, w in zip(bands, weights)))
            # 'avg': new_state already holds the weighted mean of every band; clients still restart from their own low band
        global_state = new_state
        # evaluation (log + selection): held-out under the test-centre protocol, per client (personalized for
        # fedfreq*, BN on client train). Selection uses client val only. Calibration capped for speed here.
        full_cap = cfg.bn_max_batches; cfg.bn_max_batches = cfg.bn_max_batches_round
        model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
        gm, _, _ = evaluate(model, fd.test_loader, device, fd.bn_heldout('test_center'), cfg)
        local = {}
        for c in cids:
            if method.startswith('fedfreq'): st = personalize(global_state, client_low[c], convs, cfg)
            elif method in PERSONAL_KEY_METHODS: st = OrderedDict((k, v.clone()) for k, v in global_state.items()); st.update(client_bn[c])
            else: st = global_state
            model.load_state_dict({k: v.to(device) for k, v in st.items()})
            lm, _, _ = evaluate(model, fd.client_val_loader[c], device, fd.bn_client(c, cfg.bn_source_local), cfg)
            local[str(c)] = {'accuracy': lm['accuracy'], 'auc': lm['auc']}
        sel = float(np.mean([v['accuracy'] for v in local.values()]))
        cfg.bn_max_batches = full_cap
        history.append({'round': rnd + 1, 'heldout': {k: gm[k] for k in ('accuracy', 'auc', 'specificity', 'sensitivity', 'ece', 'mce', 'brier')},
                        'local': local, 'local_mean_acc': sel, 'time_s': time.time() - t0, 'retaining': retaining})
        improved = sel > best_sel
        if improved:
            best_sel = sel; torch.save(global_state, os.path.join(run_dir, 'global_best_by_val.pt'))
            torch.save(client_low, os.path.join(run_dir, 'client_low_best_by_val.pt'))
        torch.save(global_state, os.path.join(run_dir, 'global_last.pt')); torch.save(client_low, os.path.join(run_dir, 'client_low.pt'))
        if server_low is not None: torch.save(server_low, os.path.join(run_dir, 'server_low.pt'))
        if client_bn: torch.save(client_bn, os.path.join(run_dir, 'client_bn.pt'))
        json.dump({'completed_rounds': rnd + 1, 'history': history, 'best_sel': best_sel, 'config': asdict(cfg),
                   'method': method, 'fold': fold, 'seed': seed}, open(prog_path, 'w'), indent=1)
        print(f'  {tag} R{rnd+1:2d}/{cfg.rounds} heldout acc={gm["accuracy"]:.4f} auc={gm["auc"]:.4f} | '
              f'local {sel:.4f} ' + ' '.join(f'C{c}:{local[str(c)]["accuracy"]:.2f}' for c in cids) +
              f' | {time.time()-t0:.0f}s' + (' *' if improved else '') + (' [retain]' if retaining else ''), flush=True)
    # final: held-out metrics for last and val-selected checkpoints, full metric set
    final = {'method': method, 'fold': fold, 'seed': seed, 'config': asdict(cfg), 'history': history}
    for name, st in (('last', global_state), ('best_by_val', _load_pt(os.path.join(run_dir, 'global_best_by_val.pt')))):
        if st is None: continue
        for src in ('test_center', 'client_val'):
            model.load_state_dict({k: v.to(device) for k, v in st.items()})
            m, probs, labels = evaluate(model, fd.test_loader, device, fd.bn_heldout(src), cfg)
            final[f'heldout_{name}_{src}'] = m
            if src == 'test_center': final[f'heldout_{name}'] = m      # paper protocol; what stage_tables reads
            np.savez_compressed(os.path.join(run_dir, f'pred_heldout_{name}_{src}.npz'), probs=probs, labels=labels,
                                patient=fd.test_ds.patient, node=fd.test_ds.node)
    final['local_last'] = history[-1]['local']; final['local_mean_acc_last'] = history[-1]['local_mean_acc']
    final['local_mean_acc_best'] = best_sel
    if method == 'fedavg':   # FedFreq-eval (V2 semantics): FedAvg global, client low band swapped in at evaluation
        pers = {}
        for c in cids:
            st = personalize(global_state, client_low[c], convs, cfg)
            model.load_state_dict({k: v.to(device) for k, v in st.items()})
            lm, _, _ = evaluate(model, fd.client_val_loader[c], device, fd.bn_client(c, cfg.bn_source_local), cfg)
            pers[str(c)] = {'accuracy': lm['accuracy'], 'auc': lm['auc']}
        final['local_fedfreq_eval'] = pers; final['local_mean_acc_fedfreq_eval'] = float(np.mean([v['accuracy'] for v in pers.values()]))
    json.dump(final, open(os.path.join(run_dir, 'result.json'), 'w'), indent=1)
    print(f'  done {tag}: heldout(last) acc={final["heldout_last"]["accuracy"]:.4f} auc={final["heldout_last"]["auc"]:.4f} '
          f'local(last)={final["local_mean_acc_last"]:.4f}')
    return final

def stage_sweep(cfg, methods, seeds):
    """All (method, seed) runs of each fold from one data load. Order: seed-major so early stops still give
    complete folds for the first seed."""
    maybe_mount_drive(cfg); device = get_device()
    for seed in seeds:
        for fold in cfg.folds:
            todo = [m for m in methods if not os.path.exists(os.path.join(cfg.out_dir, 'runs', run_tag(cfg, m, fold, seed), 'result.json'))]
            if not todo: print(f'  fold {fold} seed {seed}: all done'); continue
            fd = FoldData(cfg, fold, device)
            for m in todo:
                stage_run(cfg, m, fold, seed, fd=fd)
            del fd
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    stage_tables(cfg)

# ======================================================================
#  STAGE: localpred + bootstrap_runs  (statistics for the corrected runs)
# ======================================================================

def _run_dirs(cfg, fold, seed):
    """{short name: run dir} for every finished run of this fold/seed on cfg.dct_axis.
    Short name = tag without the fold/seed/axis parts, e.g. 'fedavg', 'fedfreq_train_ws15'."""
    runs = os.path.join(cfg.out_dir, 'runs'); out = {}
    if not os.path.isdir(runs): return out
    mid, end = f'_fold{fold}_seed{seed}', f'_{cfg.dct_axis}'
    for tag in sorted(os.listdir(runs)):
        if mid in tag and tag.endswith(end) and os.path.exists(os.path.join(runs, tag, 'result.json')):
            out[tag.replace(mid, '')[:-len(end)]] = os.path.join(runs, tag)
    return out

def _save_pred(path, fd, c, probs, labels):
    ds = fd.client_val_ds[c]
    np.savez_compressed(path, probs=probs, labels=labels, patient=ds.patient, node=ds.node, center=ds.center)

def stage_localpred(cfg, seeds):
    """Per-client validation predictions for every finished run, saved next to it as pred_local_C{c}.npz
    (personalized for fedfreq*, client BN for fedbn, global otherwise). For fedavg runs also:
    pred_local_ft{E}_C{c}.npz = FedAvg + E epochs of local fine-tuning (the strongest simple baseline), and
    pred_local_fe_C{c}.npz = FedAvg with the client low band swapped in at evaluation only (FedFreq-eval).
    No training except the fine-tuning epochs; BN is re-estimated on the full client training split."""
    maybe_mount_drive(cfg); device = get_device()
    for fold in cfg.folds:
        cids = [c for c in range(5) if c != fold]; fd = None
        for seed in seeds:
            for name, d in _run_dirs(cfg, fold, seed).items():
                ft = f'ft{cfg.ft_epochs}'
                need = [c for c in cids if not os.path.exists(os.path.join(d, f'pred_local_C{c}.npz'))]
                need_ft = [c for c in cids if name == 'fedavg' and not os.path.exists(os.path.join(d, f'pred_local_{ft}_C{c}.npz'))]
                need_fe = [c for c in cids if name == 'fedavg' and not os.path.exists(os.path.join(d, f'pred_local_fe_C{c}.npz'))]
                if not (need or need_ft or need_fe): print(f'  skip {os.path.basename(d)}'); continue
                if fd is None: fd = FoldData(cfg, fold, device)
                rc = json.load(open(os.path.join(d, 'result.json')))['config']
                tcfg = Config(**{**asdict(cfg), 't_low': rc['t_low'], 't_high': rc['t_high'], 'dct_axis': rc['dct_axis']})
                g = _load_pt(os.path.join(d, 'global_last.pt'))
                low = _load_pt(os.path.join(d, 'client_low.pt')) or {}; cbn = _load_pt(os.path.join(d, 'client_bn.pt')) or {}
                convs = conv_layer_names(g); rl = rc.get('retain_layers', 'all')
                if rl == 'early': convs = [k for k in convs if k.split('.')[0] in ('stem', 'stage1', 'stage2')]
                elif rl == 'late': convs = [k for k in convs if k.split('.')[0] in ('stage3', 'stage4')]
                model = HistoFreqNet().to(device)
                def score(st, c, path):
                    model.load_state_dict({k: v.to(device) for k, v in st.items()})
                    m, probs, labels = evaluate(model, fd.client_val_loader[c], device, fd.bn_client(c, cfg.bn_source_local), cfg)
                    _save_pred(path, fd, c, probs, labels); return m['accuracy']
                for c in need:
                    if name.startswith('fedfreq') and c in low: st = personalize(g, low[c], convs, tcfg)
                    elif name in PERSONAL_KEY_METHODS and c in cbn: st = OrderedDict((k, v.clone()) for k, v in g.items()); st.update(cbn[c])
                    else: st = g
                    print(f'  {os.path.basename(d)} C{c}: acc={score(st, c, os.path.join(d, f"pred_local_C{c}.npz")):.4f}', flush=True)
                for c in need_fe:
                    print(f'  {os.path.basename(d)} C{c} fedfreq_eval: acc={score(personalize(g, low[c], convs, tcfg), c, os.path.join(d, f"pred_local_fe_C{c}.npz")):.4f}', flush=True)
                for c in need_ft:
                    set_seed(seed); model.load_state_dict({k: v.to(device) for k, v in g.items()})
                    local_train(model, fd.client_train_loader[c], cfg.ft_epochs, cfg.lr, cfg.weight_decay, fd.class_weights, device)
                    st = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    print(f'  {os.path.basename(d)} C{c} +{ft}: acc={score(st, c, os.path.join(d, f"pred_local_{ft}_C{c}.npz")):.4f}', flush=True)
        del fd
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def stage_bootstrap_runs(cfg, seeds, n_boot=1000, seed=0):
    """Patient-level paired bootstrap over the corrected runs, per seed, pooled over folds (a patient that
    appears in several folds is treated as a separate cluster in each, as in stage_bootstrap).
    heldout: pred_heldout_last_test_center.npz of every run. local: pred_local_*.npz from stage_localpred."""
    rng = np.random.RandomState(seed); ft = f'fedavg_ft{cfg.ft_epochs}'
    refs_h = ['fedavg']; refs_l = ['fedavg', ft, 'fedbn', 'fedper']; out = {}
    for s in seeds:
        H, L = defaultdict(list), defaultdict(list)
        for fold in cfg.folds:
            for name, d in _run_dirs(cfg, fold, s).items():
                f = os.path.join(d, 'pred_heldout_last_test_center.npz')
                if os.path.exists(f): z = np.load(f); H[name].append((z['probs'], z['labels'], z['patient'] + 1000 * fold))
                for c in [c for c in range(5) if c != fold]:
                    for suf, nm in (('', name), (f'ft{cfg.ft_epochs}_', ft), ('fe_', 'fedavg_fedfreq_eval')):
                        if suf and name != 'fedavg': continue
                        f = os.path.join(d, f'pred_local_{suf}C{c}.npz')
                        if os.path.exists(f): z = np.load(f); L[nm].append((z['probs'], z['labels'], z['patient'] + 1000 * fold))
        res = {}
        for split, data, refs in (('heldout', H, refs_h), ('local', L, refs_l)):
            n = max((len(v) for v in data.values()), default=0)
            full = {m: v for m, v in data.items() if len(v) == n}
            if not full: print(f'  seed {s} {split}: nothing found'); continue
            labels = np.concatenate([x[1] for x in next(iter(full.values()))]); patient = np.concatenate([x[2] for x in next(iter(full.values()))])
            probs = {}
            for m, v in full.items():
                assert np.array_equal(np.concatenate([x[1] for x in v]), labels), f'{m}: label order differs'
                probs[m] = np.concatenate([x[0] for x in v])
            r = _cluster_boot(labels, patient, probs, rng, n_boot, refs); res[split] = r
            print(f'\n  seed {s} {split}: {r["n_patients"]} patient clusters, {len(labels):,} patches, 95% CI')
            for m in probs:
                print(f'    {m:22s} acc {r[m]["point"]["accuracy"]:.4f} [{r[m]["accuracy"]["lo"]:.4f}, {r[m]["accuracy"]["hi"]:.4f}]  '
                      f'auc {r[m]["point"]["auc"]:.4f} [{r[m]["auc"]["lo"]:.4f}, {r[m]["auc"]["hi"]:.4f}]')
            for ref in refs:
                for m, dd in r.get('diff_vs_' + ref, {}).items():
                    print(f'    {m:22s} - {ref:14s} dAcc {dd["accuracy"]["mean"]:+.4f} [{dd["accuracy"]["lo"]:+.4f}, {dd["accuracy"]["hi"]:+.4f}]  '
                          f'dAUC {dd["auc"]["mean"]:+.4f} [{dd["auc"]["lo"]:+.4f}, {dd["auc"]["hi"]:+.4f}]')
        out[str(s)] = res
    path = os.path.join(cfg.out_dir, 'bootstrap_runs.json'); json.dump(out, open(path, 'w'), indent=1)
    print(f'\n  saved {path}')
    return out

# ======================================================================
#  STAGE: tables
# ======================================================================

def stage_tables(cfg):
    out = {}
    rp = os.path.join(cfg.out_dir, 'reeval.json')
    if os.path.exists(rp):
        r = json.load(open(rp)); rows = defaultdict(lambda: defaultdict(list))
        for key, v in r.items():
            fold, method = key.split('/')
            for hk, m in v['heldout'].items():
                for met in ('accuracy', 'auc', 'specificity', 'sensitivity', 'brier', 'ece', 'mce', 'wga'):
                    rows[(method, hk)][met].append(m[met])
            rows[(method, 'local')]['accuracy'].append(v['local_mean_acc'])
        print('\nRE-EVALUATION OF SAVED CHECKPOINTS (mean over folds present)')
        print(f'{"method":12s} {"ckpt/bn":22s} {"n":>2s} {"acc":>7s} {"auc":>7s} {"spec":>7s} {"sens":>7s} {"brier":>7s} {"ece":>7s} {"mce":>7s}')
        for (method, hk), d in sorted(rows.items()):
            if hk == 'local': continue
            print(f'{method:12s} {hk:22s} {len(d["accuracy"]):2d} ' + ' '.join(f'{np.mean(d[k]):7.4f}' for k in ('accuracy', 'auc', 'specificity', 'sensitivity', 'brier', 'ece', 'mce')))
        print('\nPER-CLIENT (personalized for fedfreq*, BN on client train, scored on client val), mean over folds')
        for (method, hk), d in sorted(rows.items()):
            if hk == 'local': print(f'  {method:12s} local acc {np.mean(d["accuracy"]):.4f}  (folds={len(d["accuracy"])})')
        out['reeval'] = {f'{m}/{h}': {k: float(np.mean(v)) for k, v in d.items()} for (m, h), d in rows.items()}
    for f in sorted(os.listdir(cfg.out_dir)):
        if f.startswith('finetune_') and f.endswith('.json'):
            r = json.load(open(os.path.join(cfg.out_dir, f)))
            accs = [v['fedavg_finetune_mean_acc'] for v in r.values() if isinstance(v, dict) and 'fedavg_finetune_mean_acc' in v]
            lo = [v['local_only_mean_acc'] for v in r.values() if isinstance(v, dict) and 'local_only_mean_acc' in v]
            print(f'\n{f}: fedavg+finetune local acc mean={np.mean(accs):.4f} over {len(accs)} folds' + (f'; local-only {np.mean(lo):.4f}' if lo else ''))
            out[f] = {'fedavg_finetune': accs, 'local_only': lo}
    runs = os.path.join(cfg.out_dir, 'runs')
    if os.path.isdir(runs):
        print('\nCORRECTED RUNS')
        agg = defaultdict(list)
        for tag in sorted(os.listdir(runs)):
            rf = os.path.join(runs, tag, 'result.json')
            if not os.path.exists(rf): continue
            r = json.load(open(rf)); hl = r['heldout_last']
            extra = f' fedfreq_eval(local)={r["local_mean_acc_fedfreq_eval"]:.4f}' if 'local_mean_acc_fedfreq_eval' in r else ''
            print(f'  {tag:45s} heldout(last) acc={hl["accuracy"]:.4f} auc={hl["auc"]:.4f} ece={hl["ece"]:.4f} mce={hl["mce"]:.4f} | '
                  f'local(last)={r["local_mean_acc_last"]:.4f} local(best_by_val)={r["local_mean_acc_best"]:.4f}{extra}')
            c = r['config']; agg[(r['method'], c.get('warm_start', 0), c.get('dct_axis', 'flat'), c.get('t_low', 0.3), c.get('server_low', 'freeze'), c.get('retain_layers', 'all'))].append((hl['accuracy'], hl['auc'], r['local_mean_acc_last']))
        for (m, ws, ax, tl, sl, rl), v in agg.items():
            a = np.array(v); var = f'ws={ws} axis={ax} t_low={tl} server_low={sl} retain={rl}' if m == 'fedfreq_train' else f'axis={ax}'
            print(f'  {m:14s} {var:55s} n={len(v)}: acc {a[:,0].mean():.4f}±{a[:,0].std():.4f} auc {a[:,1].mean():.4f}±{a[:,1].std():.4f} local {a[:,2].mean():.4f}±{a[:,2].std():.4f}')
        out['runs'] = {f'{m}_ws{ws}_{ax}_tl{tl}_sl{sl}_rl{rl}': v for (m, ws, ax, tl, sl, rl), v in agg.items()}
    json.dump(out, open(os.path.join(cfg.out_dir, 'tables.json'), 'w'), indent=1)
    return out

# ======================================================================
#  CLI
# ======================================================================

def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['check', 'reeval', 'bootstrap', 'finetune', 'run', 'sweep', 'tables', 'localpred', 'bootstrap_runs'])
    ap.add_argument('--project'); ap.add_argument('--patches_dir'); ap.add_argument('--out_rel', help='results folder under project (default results/fix)')
    ap.add_argument('--folds', type=int, nargs='+'); ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--method', default='fedavg'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rounds', type=int); ap.add_argument('--local_epochs', type=int)
    ap.add_argument('--weighting', choices=['proportional', 'equal'])
    ap.add_argument('--dct_axis', choices=['flat', 'cin']); ap.add_argument('--t_low', type=float)
    ap.add_argument('--server_low', choices=['freeze', 'avg']); ap.add_argument('--retain_layers', choices=['all', 'early', 'late'])
    ap.add_argument('--bn_source_heldout', choices=['checkpoint', 'client_val', 'test_center', 'notebook'])
    ap.add_argument('--bn_source_local', choices=['client_train', 'client_val'])
    ap.add_argument('--warm_start', type=int); ap.add_argument('--ft_epochs', type=int); ap.add_argument('--local_only_epochs', type=int)
    ap.add_argument('--n_boot', type=int, default=1000); ap.add_argument('--ckpt', default='best')
    ap.add_argument('--batch_size', type=int)
    ap.add_argument('--bn_calib_patches', type=int)
    ap.add_argument('--redo', nargs='*', default=[], help='reeval: held-out BN sources to recompute on finished rows')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43]); ap.add_argument('--methods', nargs='+', default=['fedfreq_train', 'fedavg'])
    ap.add_argument('--norm_source', choices=['train_centers', 'dataset_stats']); ap.add_argument('--ft_source', choices=['runs', 'checkpoints'])
    ap.add_argument('--train_cap', type=int, help='low-data experiment: training patches per client (seeded subsample)')
    a = ap.parse_args(argv)
    cfg = Config()
    for k in ('project', 'patches_dir', 'rounds', 'local_epochs', 'weighting', 'dct_axis', 'bn_source_heldout', 'bn_source_local',
              'warm_start', 'ft_epochs', 'local_only_epochs', 'batch_size', 'bn_calib_patches', 'norm_source', 'ft_source', 'out_rel',
              't_low', 'server_low', 'retain_layers', 'train_cap'):
        v = getattr(a, k)
        if v is not None: setattr(cfg, k, v)
    if a.folds: cfg.folds = tuple(a.folds)
    cfg.seed = a.seed
    return a, cfg

def main(argv=None):
    a, cfg = parse(argv)
    if a.stage == 'check': stage_check(cfg)
    elif a.stage == 'reeval': cfg.norm_source = 'dataset_stats'; stage_reeval(cfg, redo=tuple(a.redo))
    elif a.stage == 'bootstrap':
        # without --bn_source_heldout, bootstrap both candidate protocols plus the legacy row
        srcs = [a.bn_source_heldout] if a.bn_source_heldout else ['client_val', 'test_center', 'notebook']
        for s in srcs:
            print(f'\n=== bootstrap ck={a.ckpt} bn={s} ==='); stage_bootstrap(cfg, ck=a.ckpt, src=s, n_boot=a.n_boot)
    elif a.stage == 'finetune': stage_finetune(cfg)
    elif a.stage == 'run': stage_run(cfg, a.method, a.fold, a.seed)
    elif a.stage == 'sweep': stage_sweep(cfg, a.methods, a.seeds)
    elif a.stage == 'tables': stage_tables(cfg)
    elif a.stage == 'localpred': stage_localpred(cfg, a.seeds)
    elif a.stage == 'bootstrap_runs': stage_bootstrap_runs(cfg, a.seeds, n_boot=a.n_boot)

if __name__ == '__main__':
    main()
