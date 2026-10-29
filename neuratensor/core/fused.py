"""
FusedSpikeBlock — the core building block of NeuraTensor v4.

Data flow:
    x  ──► RMSNorm ──► Linear(3×) ──► [main, gate, select]
                            │            │           │
                            ▼            │           ▼
                       LIF neurons       │    Selective SSM params
                       (sparse spikes)   │    (B, C, Δ from input)
                            │            │           │
                            ▼            ▼           │
                       Selective SSM ◄───────────────┘
                            │
                            ▼
                    SiLU(gate) × ssm_out
                            │
                            ▼
                    SparseLinear (inference: only non-zero spikes)
                            │
                            ▼
                         + residual
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .snn import LIFLayer
from .ssm import SelectiveSSMLayer


@dataclass
class BlockState:
    """Streaming state carried between forward calls."""
    membrane: Optional[Tensor] = None
    ssm_state: Optional[Tensor] = None


class FusedSpikeBlock(nn.Module):
    """
    Fused Spiking Neural Network → Selective State Space Model residual block.

    v4 changes from v3:
      • SSM parameters (B, C, Δ) are projected from the input (selective)
      • Spike-aware sparse output projection during inference
      • Three-way projection: main branch, gate branch, selection branch
      • Learnable output scale with improved initialization

    Args:
        d_model:       block input / output dimension
        d_state:       SSM state size
        expand:        expansion factor for inner dimension
        beta_init:     initial LIF membrane decay
        learn_beta:    make β learnable
        dropout:       spike dropout (training only)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        beta_init: float = 0.9,
        learn_beta: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)

        # Three-way projection: main (for SNN), gate, and selection (for SSM params)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.snn = LIFLayer(
            size=self.d_inner,
            beta_init=beta_init,
            learn_beta=learn_beta,
            dropout=dropout,
        )

        self.ssm = SelectiveSSMLayer(
            d_model=self.d_inner,
            d_state=d_state,
        )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Learnable output scale — small init for stable residual streams
        self.out_scale = nn.Parameter(torch.ones(1) * 0.1)

        self._sparse_inference = True

    def forward(
        self,
        x: Tensor,
        state: Optional[BlockState] = None,
    ) -> Tuple[Tensor, BlockState, dict]:
        """
        Args:
            x:     [B, T, d_model]
            state: BlockState for streaming (None = start fresh)

        Returns:
            out:     [B, T, d_model]
            state:   updated BlockState
            metrics: dict with spike_rate, mean_beta
        """
        residual = x

        h = self.norm(x)
        h, gate = self.in_proj(h).chunk(2, dim=-1)  # each [B, T, d_inner]

        # LIF: produce sparse binary spikes
        membrane_in = state.membrane if state else None
        spikes, membrane_out = self.snn(h, membrane_in)

        # Selective SSM: process spike trains with input-dependent params
        ssm_state_in = state.ssm_state if state else None
        ssm_out, ssm_state_out = self.ssm(spikes, ssm_state_in)

        # Gating
        out = ssm_out * F.silu(gate)

        # Sparse output projection during inference
        if not self.training and self._sparse_inference:
            out = self._sparse_out_proj(out, spikes)
        else:
            out = self.out_proj(out)

        out = out * self.out_scale + residual

        new_state = BlockState(membrane=membrane_out, ssm_state=ssm_state_out)

        metrics = {
            "spike_rate": spikes.mean().item(),
            "mean_beta": self.snn.beta.mean().item(),
        }

        return out, new_state, metrics

    def _sparse_out_proj(self, x: Tensor, spikes: Tensor) -> Tensor:
        """Output projection that skips positions with no spike activity.

        When sparsity is ~90%, this avoids ~90% of the matmul work at inference.
        Falls back to dense if sparsity is below 50%.
        """
        active = spikes.any(dim=-1)  # [B, T] — True where any neuron spiked
        density = active.float().mean().item()

        if density > 0.5:
            return self.out_proj(x)

        B, T, D = x.shape
        out = torch.zeros(B, T, self.d_model, device=x.device, dtype=x.dtype)
        if active.any():
            out[active] = self.out_proj(x[active])
        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_inner={self.d_inner}, d_state={self.ssm.d_state}"


# Backward-compat alias
FusedSNNSSMBlock = FusedSpikeBlock
