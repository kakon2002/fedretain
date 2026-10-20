"""DCT along the input-channel axis, subset rules, decompose/recompose, personalisation and upload size."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable

import torch

from fedretain.config import SUBSET_RULES, Config

StateDict = dict


class DCT1D:
    """Orthonormal DCT-II along the last axis, with the basis matrix cached per length."""

    _cache: dict[int, torch.Tensor] = {}

    @staticmethod
    def matrix(n: int) -> torch.Tensor:
        """The n x n DCT-II basis (computed in float64, stored as float32 like the legacy code)."""
        if n not in DCT1D._cache:
            i = torch.arange(n, dtype=torch.float64)
            k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
            mat = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
            mat[0, :] = 1.0 / math.sqrt(n)
            DCT1D._cache[n] = mat.float()
        return DCT1D._cache[n]

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Coefficients of each row of x."""
        return x @ DCT1D.matrix(x.shape[-1]).t()

    @staticmethod
    def inverse(x: torch.Tensor) -> torch.Tensor:
        """Rows back from their coefficients."""
        return x @ DCT1D.matrix(x.shape[-1])


def band_masks(n: int, t_low: float, t_high: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low / mid / high index masks: index/(n-1) <= t_low is low, >= t_high is high, the rest mid."""
    idx = torch.arange(n, dtype=torch.float32) / max(n - 1, 1)
    low = idx <= t_low
    high = idx >= t_high
    mid = ~(low | high)
    return low, mid, high


_random_masks: dict[tuple[int, int], torch.Tensor] = {}


def random_mask(n: int, count: int) -> torch.Tensor:
    """A fixed random choice of `count` of n indices (generator seeded 1000 + n, as the legacy helper)."""
    key = (n, count)
    if key not in _random_masks:
        gen = torch.Generator().manual_seed(1000 + n)
        pick = torch.randperm(n, generator=gen)[:count]
        mask = torch.zeros(n, dtype=torch.bool)
        mask[pick] = True
        _random_masks[key] = mask
    return _random_masks[key]


def subset_masks(n: int, t_low: float, t_high: float, rule: str) -> tuple[torch.Tensor, ...]:
    """(retained, shared_a, shared_b) index masks for a subset rule; slot 0 is what the client keeps."""
    low, mid, high = band_masks(n, t_low, t_high)
    if rule == "low":
        return low, mid, high
    if rule == "high":
        return high, mid, low
    if rule in ("random", "raw"):
        picked = random_mask(n, int(low.sum()))
        return picked, ~picked, torch.zeros(n, dtype=torch.bool)
    raise ValueError(f"subset_rule must be one of {SUBSET_RULES}, got {rule!r}")


class Retention:
    """Which coefficients of every conv kernel a client keeps: transform, axis, thresholds and rule."""

    def __init__(self, t_low: float = 0.3, t_high: float = 0.7, dct_axis: str = "flat",
                 subset_rule: str = "low") -> None:
        if dct_axis not in ("flat", "cin"):
            raise ValueError(dct_axis)
        if subset_rule not in SUBSET_RULES:
            raise ValueError(subset_rule)
        self.t_low, self.t_high, self.dct_axis, self.subset_rule = t_low, t_high, dct_axis, subset_rule

    @classmethod
    def from_config(cls, cfg: Config) -> "Retention":
        """Retention settings of a Config."""
        return cls(cfg.t_low, cfg.t_high, cfg.dct_axis, cfg.subset_rule)

    @property
    def transform(self) -> str:
        """'dct' for the frequency rules, 'identity' when raw weights are retained."""
        return "identity" if self.subset_rule == "raw" else "dct"

    def coefficients(self, weight: torch.Tensor) -> torch.Tensor:
        """Rows of coefficients: (c_out, c_in*k*k) for the flat axis, (c_out*k*k, c_in) for the channel axis."""
        c_out, c_in, kh, kw = weight.shape
        if self.dct_axis == "flat":
            flat = weight.reshape(c_out, c_in * kh * kw)
        else:
            flat = weight.permute(0, 2, 3, 1).reshape(c_out * kh * kw, c_in)
        return flat if self.transform == "identity" else DCT1D.forward(flat)

    def from_coefficients(self, coef: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Kernel tensor of `shape` back from its coefficient rows."""
        c_out, c_in, kh, kw = shape
        w = coef if self.transform == "identity" else DCT1D.inverse(coef)
        if self.dct_axis == "flat":
            return w.reshape(c_out, c_in, kh, kw)
        return w.reshape(c_out, kh, kw, c_in).permute(0, 3, 1, 2).contiguous()

    def masks(self, n: int) -> tuple[torch.Tensor, ...]:
        """Index masks for coefficient rows of length n under this rule."""
        return subset_masks(n, self.t_low, self.t_high, self.subset_rule)

    def decompose(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split a kernel into (retained, shared, shared) parts; 1x1 and non-conv tensors are all 'shared'."""
        weight = weight.detach().cpu().float()
        if weight.dim() != 4 or weight.shape[2] * weight.shape[3] <= 1:
            zeros = torch.zeros_like(weight)
            return zeros.clone(), weight.clone(), zeros.clone()
        coef = self.coefficients(weight)
        parts = []
        for mask in self.masks(coef.shape[-1]):
            parts.append(self.from_coefficients(coef * mask.float(), weight.shape))
        return parts[0], parts[1], parts[2]

    @staticmethod
    def recompose(retained: torch.Tensor, shared_a: torch.Tensor, shared_b: torch.Tensor) -> torch.Tensor:
        """Sum of the three parts of a kernel."""
        return retained + shared_a + shared_b

    def n_coefficients(self, shape: tuple) -> int:
        """Length of one coefficient row for a kernel of this shape."""
        c_out, c_in, kh, kw = shape
        return c_in * kh * kw if self.dct_axis == "flat" else c_in

    def band_counts(self, shape: tuple) -> tuple[int, int, int]:
        """Number of (retained, shared, shared) coefficients over a whole kernel of this shape."""
        n = self.n_coefficients(shape)
        per_row = tuple(int(m.sum()) for m in self.masks(n))
        rows = shape[0] if self.dct_axis == "flat" else shape[0] * shape[2] * shape[3]
        return per_row[0] * rows, per_row[1] * rows, per_row[2] * rows

    def personalize(self, global_state: StateDict, client_retained: StateDict,
                    conv_layers: Iterable[str]) -> StateDict:
        """Client's retained part plus the global shared parts for every conv layer."""
        state = OrderedDict((k, v.clone()) for k, v in global_state.items())
        for k in conv_layers:
            _, shared_a, shared_b = self.decompose(global_state[k])
            state[k] = self.recompose(client_retained[k], shared_a, shared_b)
        return state


def conv_layer_names(state: StateDict, pointwise: bool = False) -> list[str]:
    """Conv kernels with an input-channel axis (cin > 1); spatial kernels only unless `pointwise`."""
    names = []
    for k, p in state.items():
        if p.dim() != 4 or p.shape[1] <= 1:
            continue
        if pointwise or p.shape[2] * p.shape[3] > 1:
            names.append(k)
    return names


def select_layers(conv_layers: Iterable[str], retain_layers: str) -> list[str]:
    """Restrict the retained layers to 'all', 'early' (stem, stage1, stage2) or 'late' (stage3, stage4)."""
    groups = {"all": None, "early": ("stem", "stage1", "stage2"), "late": ("stage3", "stage4")}
    if retain_layers not in groups:
        raise ValueError(retain_layers)
    wanted = groups[retain_layers]
    if wanted is None:
        return list(conv_layers)
    return [k for k in conv_layers if k.split(".")[0] in wanted]


def retained_count(state: StateDict, conv_layers: Iterable[str], retention: Retention) -> int:
    """Total number of retained coefficients over the given conv layers."""
    return sum(retention.band_counts(tuple(state[k].shape))[0] for k in conv_layers)


def upload_size_mb(n_params: int, n_retained: int) -> tuple[float, float]:
    """(MB per client per round for a full upload, MB when the retained part stays local), fp32."""
    return n_params * 4 / 2 ** 20, (n_params - n_retained) * 4 / 2 ** 20


def retained_percent(state: StateDict, conv_layers: Iterable[str], retention: Retention) -> tuple[float, float]:
    """(% of conv coefficients kept local, % of all parameters kept local)."""
    counts = [retention.band_counts(tuple(state[k].shape)) for k in conv_layers]
    kept = sum(c[0] for c in counts)
    conv_total = sum(sum(c) for c in counts)
    n_params = sum(p.numel() for p in state.values())
    return 100.0 * kept / conv_total, 100.0 * kept / n_params
