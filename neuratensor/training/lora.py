"""LoRA (Low-Rank Adaptation) for NeuraTensor models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["in_proj", "out_proj"])


class LoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with low-rank adaptation."""

    def __init__(self, original: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.original = original
        self.r = r
        self.scaling = alpha / r

        in_features = original.in_features
        out_features = original.out_features

        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        base = self.original(x)
        lora = F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)
        return base + lora * self.scaling

    def extra_repr(self) -> str:
        return (f"in={self.original.in_features}, out={self.original.out_features}, "
                f"r={self.r}, scaling={self.scaling:.2f}")


def _set_submodule(parent: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace target linear layers with LoRALinear. Freezes all non-LoRA params."""
    replacements: list[tuple[str, LoRALinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in config.target_modules:
                replacements.append((
                    name,
                    LoRALinear(module, r=config.r, alpha=config.alpha, dropout=config.dropout),
                ))

    for name, lora_mod in replacements:
        _set_submodule(model, name, lora_mod)

    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base model for deployment (irreversible)."""
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.original.weight.add_(
                    (module.lora_B @ module.lora_A) * module.scaling
                )
            module.original.weight.requires_grad_(True)
            if module.original.bias is not None:
                module.original.bias.requires_grad_(True)
            replacements.append((name, module.original))

    for name, linear in replacements:
        _set_submodule(model, name, linear)

    return model


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Extract only LoRA parameters for lightweight checkpointing."""
    return {
        name: param
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def lora_param_count(model: nn.Module) -> int:
    """Count trainable LoRA parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
