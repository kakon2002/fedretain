"""Patch index, leave-one-centre-out folds, client splits, device-resident loaders and normalisation."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from fedretain.config import Config

PATCH_SIZE = 96
CENTRES = (0, 1, 2, 3, 4)
METADATA_COLUMNS = ("center", "split", "tumor", "patient", "node", "x_coord", "y_coord", "folder")


def get_device() -> torch.device:
    """CUDA when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_mount_drive(cfg: Config) -> None:
    """Mount Google Drive when the project lives there and is not mounted yet (no-op outside Colab)."""
    if cfg.project.startswith("/content/drive") and not os.path.isdir(cfg.project):
        try:
            from google.colab import drive  # type: ignore[import-not-found]
            drive.mount("/content/drive")
        except Exception:
            pass


def patch_path(row: pd.Series, patches_dir: str) -> str:
    """File name of one Camelyon17-WILDS patch, from its metadata row."""
    name = (f"patch_patient_{int(row['patient']):03d}_node_{int(row['node'])}"
            f"_x_{int(row['x_coord'])}_y_{int(row['y_coord'])}.png")
    return os.path.join(patches_dir, row["folder"], name)


def load_metadata(cfg: Config) -> pd.DataFrame:
    """Read the patch metadata CSV and check that every needed column is present."""
    df = pd.read_csv(cfg.metadata)
    missing = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"metadata is missing columns {missing}; have {list(df.columns)}")
    return df


def load_stats(cfg: Config) -> tuple[list, list, list]:
    """Mean, std and class weights from the legacy dataset_stats.json."""
    with open(os.path.join(cfg.results_dir, "dataset_stats.json")) as handle:
        stats = json.load(handle)
    return stats["mean"], stats["std"], stats["class_weights"]


class Split:
    """One split of patches preloaded as uint8 (N, 96, 96, 3) with the metadata the bootstrap needs."""

    def __init__(self, df: pd.DataFrame, patches_dir: str, name: str) -> None:
        self.df = df.reset_index(drop=True)
        self.labels = self.df["tumor"].values.astype(np.int64)
        self.patient = self.df["patient"].values.astype(np.int64)
        self.node = self.df["node"].values.astype(np.int64)
        self.center = self.df["center"].values.astype(np.int64)
        self.images: np.ndarray | None = np.zeros((len(self.df), PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        start = time.time()
        loaded = 0
        rows = self.df.iterrows()
        show_bar = sys.stderr.isatty() and len(self.df) > 5000
        for i, row in tqdm(rows, total=len(self.df), disable=not show_bar, leave=False, desc=name):
            try:
                img = Image.open(patch_path(row, patches_dir)).convert("RGB")
                if img.size != (PATCH_SIZE, PATCH_SIZE):
                    img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BILINEAR)
                self.images[i] = np.asarray(img)
                loaded += 1
                img.close()
            except Exception:
                pass
        print(f"      {name}: {loaded:,}/{len(self.df):,} patches in {time.time() - start:.0f}s", flush=True)

    def __len__(self) -> int:
        return len(self.labels)


class Loader:
    """Batches over uint8 images held on a storage device; normalisation and augmentation run on `device`."""

    def __init__(self, images: torch.Tensor | np.ndarray, labels: np.ndarray, mean: list, std: list,
                 batch_size: int, device: torch.device, shuffle: bool = False, augment: bool = False,
                 drop_last: bool = False, order: np.ndarray | None = None, seed: int = 0) -> None:
        self.x = images if torch.is_tensor(images) else torch.from_numpy(images)
        self.y = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.device = device
        self.set_norm(mean, std)
        self.bs = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.drop_last = drop_last
        self.order = None if order is None else torch.as_tensor(order, dtype=torch.long)
        self.generator = torch.Generator().manual_seed(seed)

    def set_norm(self, mean: list, std: list) -> None:
        """Set the per-channel normalisation applied to every batch."""
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        n = len(self.y)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def _indices(self) -> torch.Tensor:
        """Index order for one pass (fixed order, seeded shuffle, or natural order)."""
        n = len(self.y)
        storage = self.x.device
        if self.order is not None:
            return self.order.to(storage)
        if self.shuffle:
            return torch.randperm(n, generator=self.generator).to(storage)
        return torch.arange(n, device=storage)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random flips and 90-degree rotations, drawn from the global torch RNG like the legacy loader."""
        n = x.shape[0]
        flip = (torch.rand(n, device=self.device) < 0.5).view(-1, 1, 1, 1)
        x = torch.where(flip, x.flip(3), x)
        flip = (torch.rand(n, device=self.device) < 0.5).view(-1, 1, 1, 1)
        x = torch.where(flip, x.flip(2), x)
        turns = torch.randint(0, 4, (n,), device=self.device)
        for k in (1, 2, 3):
            selected = turns == k
            if bool(selected.any()):
                x[selected] = torch.rot90(x[selected], k, (2, 3))
        return x

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        idx = self._indices()
        n = len(self.y)
        for i in range(0, n, self.bs):
            batch = idx[i:i + self.bs]
            if self.drop_last and len(batch) < self.bs:
                break
            x = self.x[batch].to(self.device, non_blocking=True).permute(0, 3, 1, 2).float()
            x = x.div_(255.0).sub_(self.mean).div_(self.std)
            if self.augment:
                x = self._augment(x)
            yield x.contiguous(memory_format=torch.channels_last), self.y[batch.to(self.device)]


class FoldData:
    """Everything for one leave-one-centre-out fold, resident on `device` (or host RAM when it does not fit)."""

    def __init__(self, cfg: Config, fold: int, device: torch.device, need_train: bool | str = True) -> None:
        """need_train: True = full training splits; 'calib' = a seeded BN-calibration sample; False = none."""
        df = load_metadata(cfg)
        mean, std, self.class_weights = load_stats(cfg)
        self.cfg, self.fold, self.device = cfg, fold, device
        self.train_centers = [c for c in CENTRES if c != fold]
        print(f"  Fold {fold}: test=C{fold}, train={self.train_centers}")
        bs, ebs = cfg.batch_size, cfg.eval_batch_size
        store = self._storage_device(device, need_train)

        def up(split: Split) -> torch.Tensor:
            return torch.from_numpy(split.images).to(store)

        self.test_ds = Split(df[df["center"] == fold], cfg.patches_dir, f"test C{fold}")
        test_x = up(self.test_ds)
        self.test_loader = Loader(test_x, self.test_ds.labels, mean, std, ebs, device)
        self.test_calib_loader = Loader(test_x, self.test_ds.labels, mean, std, ebs, device,
                                        shuffle=True, seed=cfg.seed)
        split_flag = self.test_ds.df["split"].values
        notebook_order = np.concatenate([np.where(split_flag == 0)[0], np.where(split_flag == 1)[0]])
        self.test_notebook_loader = Loader(test_x, self.test_ds.labels, mean, std, 512, device,
                                           order=notebook_order)

        self.client_train_ds: dict[int, Split] = {}
        self.client_val_ds: dict[int, Split] = {}
        self.client_calib_ds: dict[int, Split] = {}
        self.client_train_loader: dict[int, Loader] = {}
        self.client_val_loader: dict[int, Loader] = {}
        self.client_bn_loader: dict[int, Loader | None] = {}
        val_x, val_y = [], []
        for c in self.train_centers:
            self.client_val_ds[c] = Split(df[(df["center"] == c) & (df["split"] == 1)], cfg.patches_dir, f"val C{c}")
            x = up(self.client_val_ds[c])
            val_x.append(x)
            val_y.append(self.client_val_ds[c].labels)
            self.client_val_loader[c] = Loader(x, self.client_val_ds[c].labels, mean, std, ebs, device)
            train_df = df[(df["center"] == c) & (df["split"] == 0)]
            if need_train == "calib":
                train_df = train_df.sample(n=min(cfg.bn_calib_patches, len(train_df)), random_state=cfg.seed)
                self.client_calib_ds[c] = Split(train_df, cfg.patches_dir, f"train-sample C{c}")
                self.client_bn_loader[c] = Loader(up(self.client_calib_ds[c]), self.client_calib_ds[c].labels,
                                                  mean, std, ebs, device, shuffle=True, seed=cfg.seed)
            elif need_train:
                if cfg.train_cap > 0:
                    train_df = train_df.sample(n=min(cfg.train_cap, len(train_df)), random_state=cfg.seed)
                self.client_train_ds[c] = Split(train_df, cfg.patches_dir, f"train C{c}")
                train_x = up(self.client_train_ds[c])
                self.client_train_ds[c].images = None
                self.client_train_loader[c] = Loader(train_x, self.client_train_ds[c].labels, mean, std, bs, device,
                                                     shuffle=True, augment=True, drop_last=True, seed=cfg.seed + c)
                self.client_bn_loader[c] = Loader(train_x, self.client_train_ds[c].labels, mean, std, ebs, device,
                                                  shuffle=True, seed=cfg.seed)
            else:
                self.client_bn_loader[c] = None
        self.pooled_val_loader = Loader(torch.cat(val_x), np.concatenate(val_y), mean, std, ebs, device,
                                        shuffle=True, seed=cfg.seed)

        self.mean, self.std = mean, std
        if cfg.norm_source == "train_centers":
            self._normalise_from_train_centres(device)
        if store.type == "cuda":
            print(f"  data on GPU: {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB", flush=True)

    @staticmethod
    def _storage_device(device: torch.device, need_train: bool | str) -> torch.device:
        """Keep patches on the GPU when it has room, otherwise in host RAM with per-batch transfer."""
        need_gb = (2.0 if need_train is True else 0.3) * 4 + 2.5
        if device.type != "cuda":
            return torch.device("cpu")
        free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
        if free_gb > need_gb:
            return device
        print(f"  GPU has {free_gb:.1f} GB free, need ~{need_gb:.0f}: patches stay in host RAM", flush=True)
        return torch.device("cpu")

    def _normalise_from_train_centres(self, device: torch.device) -> None:
        """Per-fold mean/std from a seeded sample of the training centres only (never the held-out centre)."""
        sources = {c: self.client_train_loader[c].x for c in self.client_train_loader}
        if not sources:
            sources = {c: ld.x for c, ld in self.client_bn_loader.items() if ld is not None}
        if not sources:
            sources = {c: self.client_val_loader[c].x for c in self.client_val_loader}
        gen = torch.Generator().manual_seed(self.cfg.seed)
        total = torch.zeros(3, dtype=torch.float64)
        total_sq = torch.zeros(3, dtype=torch.float64)
        count = 0
        for x in sources.values():
            idx = torch.randperm(len(x), generator=gen)[:20000]
            for j in range(0, len(idx), 1000):
                v = x[idx[j:j + 1000].to(x.device)].to(device).float().div_(255.0)
                total += v.sum(dim=(0, 1, 2)).double().cpu()
                total_sq += (v ** 2).sum(dim=(0, 1, 2)).double().cpu()
                count += v.shape[0] * PATCH_SIZE * PATCH_SIZE
        self.mean = (total / count).tolist()
        self.std = (total_sq / count - (total / count) ** 2).sqrt().tolist()
        for loader in self.all_loaders():
            loader.set_norm(self.mean, self.std)
        print(f"  norm (train centres only): mean={np.round(self.mean, 4).tolist()} "
              f"std={np.round(self.std, 4).tolist()}")

    def all_loaders(self) -> list[Loader]:
        """Every loader of the fold (used to update the normalisation in place)."""
        loaders = [self.test_loader, self.test_calib_loader, self.test_notebook_loader, self.pooled_val_loader]
        loaders += list(self.client_val_loader.values()) + list(self.client_train_loader.values())
        loaders += [ld for ld in self.client_bn_loader.values() if ld is not None]
        return loaders

    def bn_heldout(self, source: str) -> tuple[Loader | None, bool]:
        """(loader, ema) for re-estimating BN statistics before scoring the held-out centre."""
        return {
            "client_val": (self.pooled_val_loader, False),
            "test_center": (self.test_calib_loader, False),
            "notebook": (self.test_notebook_loader, True),
            "checkpoint": (None, False),
        }[source]

    def bn_client(self, c: int, source: str) -> tuple[Loader, bool]:
        """(loader, ema) for a client's own model: its training patches (default) or its validation split."""
        if source == "client_train":
            if self.client_bn_loader[c] is None:
                raise RuntimeError("client training data not loaded; use need_train=True")
            return self.client_bn_loader[c], False
        return self.client_val_loader[c], False
