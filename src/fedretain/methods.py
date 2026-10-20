"""One small class per federated method with client_start / on_client_done / aggregate / eval_state hooks."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from fedretain.config import Config
from fedretain.model import bn_keys
from fedretain.retention import Retention, StateDict

METHOD_ALIASES = {"fedfreq_train": "fedretain"}
RETAIN_RUN_PREFIXES = ("fedfreq", "fedretain")
PERSONAL_KEY_METHODS = ("fedbn", "fedper", "fedper4")
METHOD_NAMES = ("fedavg", "fedprox", "fedbn", "fedper", "fedper4", "fedretain")

# file names inside a run directory (legacy names, kept so old result directories load unchanged)
CLIENT_RETAINED_FILE = "client_low.pt"
CLIENT_RETAINED_BEST_FILE = "client_low_best_by_val.pt"
SERVER_RETAINED_FILE = "server_low.pt"
CLIENT_PERSONAL_FILE = "client_bn.pt"


def canonical_method(name: str) -> str:
    """Map a legacy method name (fedfreq_train) to the new one (fedretain); other names pass through."""
    return METHOD_ALIASES.get(name, name)


def is_retain_method(name: str) -> bool:
    """True for the FedRetain method under either of its names."""
    return canonical_method(name) == "fedretain"


def is_retain_run(name: str) -> bool:
    """True for run names or tags that hold a FedRetain-style run (legacy fedfreq* or fedretain*)."""
    return name.startswith(RETAIN_RUN_PREFIXES)


def personal_keys(model: nn.Module, method: str) -> list[str]:
    """State-dict keys that never leave the client: BN tensors, the classifier, or classifier + last stage."""
    if method == "fedbn":
        return sorted(bn_keys(model))
    if method == "fedper":
        return [k for k in model.state_dict() if k.startswith("fc.")]
    if method == "fedper4":
        return [k for k in model.state_dict() if k.startswith(("fc.", "stage4."))]
    return []


def weighted_average(states: Sequence[StateDict], weights: Sequence[float]) -> StateDict:
    """Weighted mean of state dicts; integer tensors (num_batches_tracked) are copied from the first."""
    total = float(sum(weights))
    out: StateDict = OrderedDict()
    for k in states[0]:
        if states[0][k].dtype in (torch.int64, torch.int32, torch.long):
            out[k] = states[0][k].clone()
        else:
            out[k] = sum(s[k].float() * (w / total) for s, w in zip(states, weights))
    return out


def with_personal(global_state: StateDict, personal: StateDict) -> StateDict:
    """Copy of the global state with a client's private tensors written over it."""
    state = OrderedDict((k, v.clone()) for k, v in global_state.items())
    state.update(personal)
    return state


def _load_optional(path: str) -> dict | None:
    """torch.load of a file that may not exist yet."""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


class Method:
    """FedAvg, and the base of every other method; subclasses override the hooks they need."""

    name = "fedavg"
    prox_mu = 0.0

    def __init__(
        self, cfg: Config, retention: Retention, conv_layers: Iterable[str], model: nn.Module
    ) -> None:
        self.cfg = cfg
        self.retention = retention
        self.conv_layers = list(conv_layers)
        self.personal = personal_keys(model, self.name)
        self.client_retained: dict[int, StateDict] = {}
        self.server_retained: StateDict | None = None
        self.client_personal: dict[int, StateDict] = {}

    def retaining(self, rnd: int) -> bool:
        """True while the server keeps its retained part frozen (never for the baselines)."""
        return False

    def client_start(self, c: int, rnd: int, global_state: StateDict) -> StateDict:
        """Weights a client starts round `rnd` from."""
        return global_state

    def on_client_done(self, c: int, state: StateDict) -> None:
        """Record what the client keeps: its retained conv subset (every method, for fedfreq_eval)."""
        self.client_retained[c] = {k: self.retention.decompose(state[k])[0] for k in self.conv_layers}

    def aggregate(
        self, global_state: StateDict, states: Sequence[StateDict], weights: Sequence[float], rnd: int
    ) -> StateDict:
        """New global weights from the client states."""
        return weighted_average(states, weights)

    def eval_state(self, c: int, global_state: StateDict) -> StateDict:
        """Weights scored on client c's own validation split."""
        return global_state

    def save(self, run_dir: str, best: bool = False) -> None:
        """Write the per-client and server tensors next to the global checkpoint (legacy file names)."""
        if best:
            torch.save(self.client_retained, os.path.join(run_dir, CLIENT_RETAINED_BEST_FILE))
            return
        torch.save(self.client_retained, os.path.join(run_dir, CLIENT_RETAINED_FILE))
        if self.server_retained is not None:
            torch.save(self.server_retained, os.path.join(run_dir, SERVER_RETAINED_FILE))
        if self.client_personal:
            torch.save(self.client_personal, os.path.join(run_dir, CLIENT_PERSONAL_FILE))

    def load(self, run_dir: str) -> None:
        """Restore the tensors written by save() when resuming a run."""
        self.client_retained = _load_optional(os.path.join(run_dir, CLIENT_RETAINED_FILE)) or {}
        self.server_retained = _load_optional(os.path.join(run_dir, SERVER_RETAINED_FILE))
        self.client_personal = _load_optional(os.path.join(run_dir, CLIENT_PERSONAL_FILE)) or {}


class FedAvg(Method):
    """Plain FedAvg with sample-proportional (or equal) weights."""

    name = "fedavg"


class FedProx(Method):
    """FedAvg with the proximal term (cfg.prox_mu) in local training."""

    name = "fedprox"

    def __init__(
        self, cfg: Config, retention: Retention, conv_layers: Iterable[str], model: nn.Module
    ) -> None:
        super().__init__(cfg, retention, conv_layers, model)
        self.prox_mu = cfg.prox_mu


class PersonalKeyMethod(Method):
    """Base of the methods that keep a fixed set of state-dict keys private on each client."""

    def client_start(self, c: int, rnd: int, global_state: StateDict) -> StateDict:
        if c in self.client_personal:
            return with_personal(global_state, self.client_personal[c])
        return global_state

    def on_client_done(self, c: int, state: StateDict) -> None:
        super().on_client_done(c, state)
        self.client_personal[c] = {k: state[k].clone() for k in self.personal}

    def eval_state(self, c: int, global_state: StateDict) -> StateDict:
        if c in self.client_personal:
            return with_personal(global_state, self.client_personal[c])
        return global_state


class FedBN(PersonalKeyMethod):
    """BatchNorm affine parameters and running statistics stay on each client."""

    name = "fedbn"


class FedPer(PersonalKeyMethod):
    """The classifier head stays on each client from round 1; the server averages the body."""

    name = "fedper"


class FedPer4(PersonalKeyMethod):
    """FedPer variant keeping the last stage and the head local."""

    name = "fedper4"


class FedRetain(Method):
    """After warm_start FedAvg rounds each client keeps its conv subset; the server freezes its own copy."""

    name = "fedretain"

    def retaining(self, rnd: int) -> bool:
        return rnd >= self.cfg.warm_start

    def client_start(self, c: int, rnd: int, global_state: StateDict) -> StateDict:
        if c in self.client_retained and rnd >= self.cfg.warm_start:
            return self.retention.personalize(global_state, self.client_retained[c], self.conv_layers)
        return global_state

    def aggregate(
        self, global_state: StateDict, states: Sequence[StateDict], weights: Sequence[float], rnd: int
    ) -> StateDict:
        new_state = weighted_average(states, weights)
        if not self.retaining(rnd) or self.cfg.server_low != "freeze":
            return new_state
        if self.server_retained is None:
            self.server_retained = {k: self.retention.decompose(global_state[k])[0] for k in self.conv_layers}
        total = float(sum(weights))
        for k in self.conv_layers:
            parts = [self.retention.decompose(st[k]) for st in states]
            shared_a = sum(p[1] * (w / total) for p, w in zip(parts, weights))
            shared_b = sum(p[2] * (w / total) for p, w in zip(parts, weights))
            new_state[k] = self.retention.recompose(self.server_retained[k], shared_a, shared_b)
        return new_state

    def eval_state(self, c: int, global_state: StateDict) -> StateDict:
        if c in self.client_retained:
            return self.retention.personalize(global_state, self.client_retained[c], self.conv_layers)
        return global_state


METHOD_CLASSES = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedbn": FedBN,
    "fedper": FedPer,
    "fedper4": FedPer4,
    "fedretain": FedRetain,
}


def make_method(
    name: str, cfg: Config, retention: Retention, conv_layers: Iterable[str], model: nn.Module
) -> Method:
    """Instantiate the method class for a (possibly legacy) method name."""
    key = canonical_method(name)
    if key not in METHOD_CLASSES:
        raise ValueError(f"unknown method {name!r}; choose from {METHOD_NAMES} or fedfreq_train")
    return METHOD_CLASSES[key](cfg, retention, conv_layers, model)
