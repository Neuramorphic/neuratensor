"""
Selective State Space Model — input-dependent B, C, Δ with HiPPO initialization.

The recurrence (now with time-varying parameters):
    h_t = Ā_t · h_{t-1} + B̄_t · u_t       (state update)
    y_t =  C_t · h_t    +  D   · u_t       (readout)

Key differences from v3:
  • B, C, Δ are projected from the input (selective / Mamba-style)
  • A is initialized with HiPPO-inspired log-spaced eigenvalues
  • The SSM "selects" what to remember based on the current input
  • in_proj / out_proj removed — handled by FusedSpikeBlock

A is diagonal and parameterized as A = -exp(log_A) for guaranteed stability.

ZOH discretization converts continuous (A, B) → discrete (Ā, B̄):
    Ā = exp(Δ·A)
    B̄ = (Ā − I) / A · B

Works on CPU, CUDA, and MPS. No compiled extensions required.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _hippo_init_log_A(d_model: int, d_state: int) -> Tensor:
    """HiPPO-inspired diagonal initialization: eigenvalue n = -(n+1).

    Each state dimension captures a different temporal scale.
    Stored as log(|A|) so that A = -exp(log_A) stays negative.
    """
    A = torch.zeros(d_model, d_state)
    for n in range(d_state):
        A[:, n] = -(n + 1)
    return A.abs().log()


class SelectiveSSMLayer(nn.Module):
    """
    Selective diagonal State Space Model with input-dependent parameters.

    B, C, and Δ are linear projections of the input — the model learns to
    select what information enters and leaves the recurrent state based on
    the current token.

    Args:
        d_model:       feature dimension (number of independent SSM channels)
        d_state:       SSM state size per channel
        dt_min:        minimum timescale Δ (for bias initialization)
        dt_max:        maximum timescale Δ (for bias initialization)
        bidirectional: run a second scan in reverse (disables streaming)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional

        # A: diagonal, HiPPO-initialized, parameterized as -exp(log_A)
        self.log_A = nn.Parameter(_hippo_init_log_A(d_model, d_state))

        # D: learnable skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # Selective projections: B, C, Δ are functions of the input
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        # dt_proj bias initialized log-uniform in [dt_min, dt_max]
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            # softplus^{-1}(dt_init) so that softplus(bias) ≈ dt_init at start
            # Numerically stable: softplus^{-1}(x) = log(expm1(log(x))) = log(x - 1 + eps)
            inv_softplus = (dt_init - 1.0).clamp(min=1e-8).log()
            self.dt_proj.bias.copy_(inv_softplus)

        # Bidirectional: separate B, C projections for the reverse scan
        if bidirectional:
            self.B_proj_rev = nn.Linear(d_model, d_state, bias=False)
            self.C_proj_rev = nn.Linear(d_model, d_state, bias=False)

    # ------------------------------------------------------------------ #
    #  Discretization                                                      #
    # ------------------------------------------------------------------ #

    def _discretize(
        self,
        dt: Tensor,
        B: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Discretize continuous (A, B) → (Ā, B̄) with ZOH.

        Args:
            dt: [B, T, d_model]   per-token timescales
            B:  [B, T, d_state]   per-token input projection

        Returns:
            A_bar: [B, T, d_model, d_state]
            B_bar: [B, T, d_model, d_state]
        """
        A = -self.log_A.exp()  # [d_model, d_state]

        # dt: [B, T, d_model] → [B, T, d_model, 1] for broadcasting
        dt_4d = dt.unsqueeze(-1)
        A_bar = torch.exp(A * dt_4d)  # [B, T, d_model, d_state]

        # Numerically stable ZOH B_bar
        # When A → 0: (e^{ΔA} - 1) / A → Δ, so B_bar → Δ·B
        use_limit = A.abs() < 1e-6
        safe_A = torch.where(use_limit, torch.ones_like(A), A)

        # B: [B, T, d_state] → [B, T, 1, d_state] for broadcasting with d_model
        B_4d = B.unsqueeze(-2)  # [B, T, 1, d_state]

        B_bar_general = (A_bar - 1.0) / safe_A * B_4d
        B_bar_limit = dt_4d * B_4d
        B_bar = torch.where(use_limit, B_bar_limit, B_bar_general)

        return A_bar, B_bar

    # ------------------------------------------------------------------ #
    #  Scan implementations                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _scan_single_step(
        u: Tensor,
        A_bar: Tensor,
        B_bar: Tensor,
        C: Tensor,
        D: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Fast path for T=1 (streaming / autoregressive).

        Args:
            u:     [B, 1, d_model]
            A_bar: [B, 1, d_model, d_state]
            B_bar: [B, 1, d_model, d_state]
            C:     [B, 1, d_state]
            D:     [d_model]
            state: [B, d_model, d_state] or None
        """
        B_sz = u.shape[0]
        d_model = u.shape[2]
        d_state = A_bar.shape[-1]
        device, dtype = u.device, u.dtype

        if state is None:
            state = torch.zeros(B_sz, d_model, d_state, device=device, dtype=dtype)

        # Squeeze time dim
        A_t = A_bar[:, 0]  # [B, d_model, d_state]
        B_t = B_bar[:, 0]  # [B, d_model, d_state]
        C_t = C[:, 0]      # [B, d_state]
        u_t = u[:, 0]      # [B, d_model]

        state = A_t * state + B_t * u_t.unsqueeze(-1)
        # C_t: [B, d_state] → [B, 1, d_state] for broadcast with state [B, d_model, d_state]
        y_t = (C_t.unsqueeze(1) * state).sum(-1) + D * u_t  # [B, d_model]

        return y_t.unsqueeze(1), state

    @staticmethod
    def _scan_sequential(
        u: Tensor,
        A_bar: Tensor,
        B_bar: Tensor,
        C: Tensor,
        D: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """O(T) sequential scan — reference implementation and fallback.

        Args:
            u:     [B, T, d_model]
            A_bar: [B, T, d_model, d_state]
            B_bar: [B, T, d_model, d_state]
            C:     [B, T, d_state]
            D:     [d_model]
            state: [B, d_model, d_state] or None
        """
        B_sz, T, d_model = u.shape
        d_state = A_bar.shape[-1]
        device, dtype = u.device, u.dtype

        if state is None:
            state = torch.zeros(B_sz, d_model, d_state, device=device, dtype=dtype)

        outputs = []
        for t in range(T):
            A_t = A_bar[:, t]  # [B, d_model, d_state]
            B_t = B_bar[:, t]  # [B, d_model, d_state]
            C_t = C[:, t]      # [B, d_state]
            u_t = u[:, t]      # [B, d_model]

            state = A_t * state + B_t * u_t.unsqueeze(-1)
            y_t = (C_t.unsqueeze(1) * state).sum(-1) + D * u_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1), state

    @staticmethod
    def _scan_chunked(
        u: Tensor,
        A_bar: Tensor,
        B_bar: Tensor,
        C: Tensor,
        D: Tensor,
        state: Optional[Tensor] = None,
        chunk_size: int = 32,
    ) -> Tuple[Tensor, Tensor]:
        """Chunked scan for selective SSM with time-varying B, C.

        A_bar has a static base (from log_A) modulated by per-token dt, so within
        a chunk the A component is nearly constant when dt varies slowly. We use
        the per-step A_bar values directly — no constant-A assumption needed.

        Float32 accumulation avoids bfloat16 precision loss in exponentiation.

        Args:
            u:     [B, T, d_model]
            A_bar: [B, T, d_model, d_state]
            B_bar: [B, T, d_model, d_state]
            C:     [B, T, d_state]
            D:     [d_model]
            state: [B, d_model, d_state] or None
        """
        B_sz, T, d_model = u.shape
        d_state = A_bar.shape[-1]
        device, dtype = u.device, u.dtype

        if state is None:
            state = torch.zeros(B_sz, d_model, d_state, device=device, dtype=dtype)

        n_chunks = math.ceil(T / chunk_size)
        outputs: list[Tensor] = []

        for ci in range(n_chunks):
            s = ci * chunk_size
            e = min(s + chunk_size, T)
            csz = e - s

            # Slice chunk tensors
            u_c = u[:, s:e]          # [B, csz, d_model]
            A_c = A_bar[:, s:e]      # [B, csz, d_model, d_state]
            B_c = B_bar[:, s:e]      # [B, csz, d_model, d_state]
            C_c = C[:, s:e]          # [B, csz, d_state]

            # Float32 accumulation for numerical stability
            A_f = A_c.float()
            B_f = B_c.float()
            state_f = state.float()

            # Compute cumulative product of A within the chunk for prefix propagation
            # log_A_cum[j] = sum_{k=0}^{j} log(A_c[k])
            log_A_c = A_f.clamp(min=1e-20).log()            # [B, csz, d_model, d_state]
            log_A_cum = log_A_c.cumsum(dim=1)                # [B, csz, d_model, d_state]
            A_cum = log_A_cum.exp()                          # [B, csz, d_model, d_state]

            # For state propagation: A_cum_full[j] = prod_{k=0}^{j} A[k]
            # Contribution of initial state at position j: A_cum[j] * state
            h_from_state = A_cum * state_f.unsqueeze(1)      # [B, csz, d_model, d_state]

            # Contribution of inputs: h_from_input[j] = sum_{k=0}^{j} (prod_{i=k+1}^{j} A[i]) * B[k] * u[k]
            # = sum_{k=0}^{j} A_cum[j] / A_cum[k] * B[k] * u[k]
            # = A_cum[j] * sum_{k=0}^{j} (B[k] * u[k] / A_cum[k])
            Bu = B_f * u_c.float().unsqueeze(-1)             # [B, csz, d_model, d_state]

            # Shift A_cum for division: at position k, divide by A_cum[k]
            # A_cum[0] = A[0], so item at k=0 is divided by A[0]
            inv_A_cum = (-log_A_cum).exp()                   # [B, csz, d_model, d_state]
            scaled_Bu = Bu * inv_A_cum                       # [B, csz, d_model, d_state]
            prefix_sum = scaled_Bu.cumsum(dim=1)             # [B, csz, d_model, d_state]
            h_from_input = A_cum * prefix_sum                # [B, csz, d_model, d_state]

            h_chunk = h_from_state + h_from_input            # [B, csz, d_model, d_state]

            # Readout: y[t] = C[t] · h[t] + D · u[t]
            # C_c: [B, csz, d_state] → [B, csz, 1, d_state]
            y_c = (C_c.unsqueeze(2).float() * h_chunk).sum(-1)  # [B, csz, d_model]
            y_c = y_c.to(dtype) + D.unsqueeze(0).unsqueeze(0) * u_c

            outputs.append(y_c)
            state = h_chunk[:, -1].to(dtype)                 # [B, d_model, d_state]

        return torch.cat(outputs, dim=1), state

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:     [B, T, d_model]
            state: [B, d_model, d_state] or None

        Returns:
            y:     [B, T, d_model]
            state: [B, d_model, d_state]
        """
        # Selective parameters from input
        B = self.B_proj(x)                        # [B, T, d_state]
        C = self.C_proj(x)                        # [B, T, d_state]
        dt = F.softplus(self.dt_proj(x))          # [B, T, d_model]

        A_bar, B_bar = self._discretize(dt, B)

        T = x.shape[1]
        scan_fn = self._scan_single_step if T == 1 else self._scan_sequential

        if self.bidirectional:
            B_rev = self.B_proj_rev(x)
            C_rev = self.C_proj_rev(x)
            A_bar_rev, B_bar_rev = self._discretize(dt.flip(1), B_rev.flip(1))

            y_fwd, state_fwd = scan_fn(x, A_bar, B_bar, C, self.D, state)
            y_bwd, _ = scan_fn(x.flip(1), A_bar_rev, B_bar_rev, C_rev.flip(1), self.D)
            y = y_fwd + y_bwd.flip(1)
            state = state_fwd
        else:
            y, state = scan_fn(x, A_bar, B_bar, C, self.D, state)

        return y, state

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, d_state={self.d_state}, "
                f"selective=True, bidirectional={self.bidirectional}")


# Backward-compat alias
SSMLayer = SelectiveSSMLayer
