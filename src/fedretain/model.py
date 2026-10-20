"""HistoFreqNet (module names and init identical to the legacy code) and the pretrained backbone factories."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_DEFAULTS = {
    "histofreqnet": {"rounds": 25, "local_epochs": 3, "lr": 1e-3, "optimizer": "adamw"},
    "mobilenetv3": {"rounds": 20, "local_epochs": 1, "lr": 1e-2, "optimizer": "sgd"},
    "efficientnetb2": {"rounds": 20, "local_epochs": 1, "lr": 1e-2, "optimizer": "sgd"},
}


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gating."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(w))))
        return x * w.view(b, c, 1, 1)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm, SE gating and a (possibly strided) shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return F.silu(out + self.shortcut(x))


class HistoFreqNet(nn.Module):
    """The paper's CNN for 96x96 patches; construction order and init are the legacy ones so seeds match."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(inplace=True)
        )
        self.stage1 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64, 1))
        self.stage2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128, 1))
        self.stage3 = nn.Sequential(ResBlock(128, 192, 2), ResBlock(192, 192, 1))
        self.stage4 = nn.Sequential(ResBlock(192, 256, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(256, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        return self.fc(self.dropout(self.pool(x).flatten(1)))


def build_model(arch: str = "histofreqnet", pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """Model factory: HistoFreqNet, or an ImageNet backbone with a two-class classifier head."""
    if arch == "histofreqnet":
        return HistoFreqNet(num_classes)
    import torchvision

    weights = "IMAGENET1K_V1" if pretrained else None
    if arch == "mobilenetv3":
        model = torchvision.models.mobilenet_v3_large(weights=weights)
    elif arch == "efficientnetb2":
        model = torchvision.models.efficientnet_b2(weights=weights)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def bn_keys(model: nn.Module) -> set[str]:
    """State-dict keys of every BatchNorm2d tensor (affine parameters and running statistics)."""
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                keys.add(f"{name}.{k}")
    return keys


def feature_pool(model: nn.Module) -> nn.Module:
    """The global pooling module whose output is the penultimate feature vector."""
    pool = getattr(model, "pool", None)
    return pool if pool is not None else model.avgpool
