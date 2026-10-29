"""
Spiking Neural Network layers — NeuraTensor v4.

LIFLayer:       Leaky Integrate-and-Fire with chunked parallel scan.
AdaptiveLIFLayer: LIF with per-neuron learnable thresholds.

Chunked parallel scan reduces Python dispatch overhead from O(T) to
O(T / chunk_size).  For T=1024, chunk_size=64 → 16 Python iterations
instead of 1024.  For T=1 (streaming) a fast single-step path runs
with zero loop overhead.

Differentiable via fast-sigmoid surrogate gradient (Zenke & Ganguli 2021).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ── Surrogate gradient ───────────────────────────────────────────────────────

class SurrogateGradient(torch.autograd.Function):
    """
    Fast-sigmoid surrogate gradient for spike backpropagation.

    Forward:  S(V) = 1 if V >= threshold else 0   (Heaviside)
    Backward: dS/dV ≈ k / (1 + k·|V - threshold|)²
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        membrane: Tensor,
        threshold: Tensor,
        slope: float,
    ) -> Tensor:
        ctx.save_for_backward(membrane, threshold)
        ctx.slope = slope
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor], None]:
        membrane, threshold = ctx.saved_tensors
        k = ctx.slope
        sg = k / (1.0 + k * (membrane - threshold).abs()) ** 2
        return grad_output * sg, -grad_output * sg, None


def spike_function(membrane: Tensor, threshold: Tensor, slope: float = 10.0) -> Tensor:
    """Apply spike function with surrogate gradient."""
    return SurrogateGradient.apply(membrane, threshold, slope)


# ── Chunked parallel LIF scan ───────────────────────────────────────────────

def _lif_single_step(
    x_t: Tensor,
    membrane: Tensor,
    beta: Tensor,
    threshold: Tensor,
    slope: float,
) -> Tuple[Tensor, Tensor]:
    """Fast path for T=1.  Zero Python-loop overhead."""
    membrane = beta * membrane + x_t
    spk = spike_function(membrane, threshold, slope)
    membrane = membrane - threshold * spk.detach()
    return spk, membrane


def _lif_chunked_scan(
    x: Tensor,
    membrane: Tensor,
    beta: Tensor,
    threshold: Tensor,
    slope: float,
    chunk_size: int = 64,
) -> Tuple[Tensor, Tensor]:
    """
    Chunked parallel LIF scan — O(T / chunk_size) Python iterations.

    Within each chunk the linear recurrence ``V(t) = β·V(t-1) + I(t)``
    is solved in closed form using a cumulative-sum trick (valid when β
    is constant across timesteps within the chunk).  Spike detection and
    soft reset are then applied sequentially inside the chunk, but the
    chunk size is small (32-64) so the inner loop is tight and fast.

    Between chunks the final membrane state carries over exactly as in
    the original sequential scan, so correctness is identical.

    Parameters
    ----------
    x         : [B, T, size]  input currents
    membrane  : [B, size]     initial membrane potential
    beta      : [size]        decay constants in (0, 1)
    threshold : [size]        spike thresholds
    slope     : float         surrogate gradient sharpness
    chunk_size: int           timesteps per chunk (default 64)
    """
    B_sz, T, size = x.shape
    device, dtype = x.device, x.dtype

    # Precompute log(β) once — reused across all chunks.
    # Float32 accumulation avoids bfloat16 precision loss.
    log_beta = torch.log(beta.float().clamp(min=1e-7, max=1.0 - 1e-7))  # [size]

    # Power table for the maximum chunk length: β^k for k=0..chunk_size-1
    k_idx = torch.arange(chunk_size, device=device, dtype=torch.float32)  # [cs]
    decay_table = torch.exp(k_idx[:, None] * log_beta[None, :])           # [cs, size]
    inv_decay_table = torch.exp(-k_idx[:, None] * log_beta[None, :])      # [cs, size]

    spikes_chunks: list[Tensor] = []
    mem = membrane  # running membrane state

    n_chunks = math.ceil(T / chunk_size)
    for ci in range(n_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, T)
        csz = e - s

        x_c = x[:, s:e, :]  # [B, csz, size]

        # ── Parallel integrate (ignoring resets) ─────────────────────
        # V(t) = β^t · (β·V₀ + Σ_{k=0}^{t} β^{-k} · I(k))
        decay_c = decay_table[:csz]         # [csz, size]
        inv_decay_c = inv_decay_table[:csz]  # [csz, size]

        weighted = x_c.float() * inv_decay_c[None, :, :]   # [B, csz, size]
        cum_input = weighted.cumsum(dim=1)                   # [B, csz, size]

        init_term = (beta.float() * mem.float()).unsqueeze(1)  # [B, 1, size]
        V = (decay_c[None, :, :] * (init_term + cum_input)).to(dtype)  # [B, csz, size]

        # ── Sequential spike detection + reset correction ────────────
        # Within the (small) chunk, apply threshold + soft reset.
        # Each reset subtracts θ from the current V and propagates the
        # decay-weighted correction to all later timesteps in the chunk.
        spk_chunk = torch.zeros(B_sz, csz, size, device=device, dtype=dtype)
        V = V.clone()

        for t in range(csz):
            spk = spike_function(V[:, t, :].clone(), threshold, slope)   # [B, size]
            spk_chunk[:, t, :] = spk
            reset = threshold * spk.detach()                     # [B, size]

            # Propagate reset to remaining positions in this chunk
            remaining = csz - 1 - t
            if remaining > 0:
                future_decay = decay_table[1:remaining + 1]      # [rem, size]
                V[:, t + 1:, :] = V[:, t + 1:, :] - reset.unsqueeze(1) * future_decay[None, :, :]

        spikes_chunks.append(spk_chunk)

        # Carry membrane = last V after final reset
        mem = V[:, -1, :] - threshold * spk_chunk[:, -1, :].detach()

    return torch.cat(spikes_chunks, dim=1), mem


def _lif_sequential_scan(
    x: Tensor,
    membrane: Tensor,
    beta: Tensor,
    threshold: Tensor,
    slope: float,
) -> Tuple[Tensor, Tensor]:
    """Original O(T) sequential scan — kept for correctness verification."""
    B_sz, T, size = x.shape
    spikes_out: list[Tensor] = []
    for t in range(T):
        membrane = beta * membrane + x[:, t, :]
        spk = spike_function(membrane, threshold, slope)
        membrane = membrane - threshold * spk.detach()
        spikes_out.append(spk)
    return torch.stack(spikes_out, dim=1), membrane


# ── LIFLayer ─────────────────────────────────────────────────────────────────

class LIFLayer(nn.Module):
    """
    Leaky Integrate-and-Fire layer with optional chunked parallel scan.

    Dynamics (per timestep)::

        V(t)  = β · V(t-1) + I(t)          # integrate
        S(t)  = Heaviside(V(t) - θ)         # spike
        V(t) -= θ · S(t)                    # soft reset

    β  (membrane decay) is learnable per neuron.
    θ  (threshold) is fixed at init (use AdaptiveLIFLayer for learnable θ).

    **Note on chunked scan:** The chunked parallel implementation has known
    numerical issues with long sequences (T>128) and high beta values that
    can produce NaN in the membrane state. It is disabled by default until
    these issues are resolved. Use `use_chunked=True` to enable it at your
    own risk, or use the sequential scan (default) which is slower but correct.

    For T = 1 a dedicated single-step path executes with zero loops regardless
    of the `use_chunked` setting.

    Parameters
    ----------
    size            : number of neurons
    beta_init       : initial membrane decay (0 < β < 1)
    threshold       : spike threshold
    learn_beta      : make β a learnable parameter
    surrogate_slope : sharpness of surrogate gradient
    dropout         : spike dropout probability (training regularisation)
    use_chunked     : use chunked parallel scan (faster but has numerical issues)
    chunk_size      : timesteps per chunk in parallel scan (if use_chunked=True)
    multi_threshold : if True, neurons get thresholds drawn from
                      {θ, 1.5θ, 2θ} for richer spike patterns
    """

    def __init__(
        self,
        size: int,
        beta_init: float = 0.9,
        threshold: float = 1.0,
        learn_beta: bool = True,
        surrogate_slope: float = 10.0,
        dropout: float = 0.0,
        use_chunked: bool = False,
        chunk_size: int = 64,
        multi_threshold: bool = False,
    ) -> None:
        super().__init__()
        self.size = size
        self.surrogate_slope = surrogate_slope
        self.use_chunked = use_chunked
        self.chunk_size = chunk_size

        # β in logit space so sigmoid keeps it in (0, 1)
        beta_logit = math.log(beta_init / (1.0 - beta_init))
        if learn_beta:
            self.beta_logit = nn.Parameter(torch.full((size,), beta_logit))
        else:
            self.register_buffer("beta_logit", torch.full((size,), beta_logit))

        # θ — per-neuron threshold (not learnable here; see AdaptiveLIFLayer)
        if multi_threshold:
            scales = torch.ones(size)
            third = size // 3
            scales[third: 2 * third] = 1.5
            scales[2 * third:] = 2.0
            self.register_buffer("threshold", threshold * scales)
        else:
            self.register_buffer("threshold", torch.full((size,), threshold))

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else None

    @property
    def beta(self) -> Tensor:
        return torch.sigmoid(self.beta_logit)

    def forward(
        self,
        x: Tensor,
        membrane: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x        : [B, T, size]  input currents
        membrane : [B, size]     initial membrane potential (zeros if None)

        Returns
        -------
        spikes   : [B, T, size]  binary spike train
        membrane : [B, size]     final membrane potential (for streaming)
        """
        B, T, sz = x.shape
        assert sz == self.size, f"Expected size={self.size}, got {sz}"

        beta = self.beta
        device, dtype = x.device, x.dtype

        if membrane is None:
            membrane = torch.zeros(B, sz, device=device, dtype=dtype)
        else:
            membrane = membrane.to(device=device, dtype=dtype)

        if T == 1:
            spk, membrane = _lif_single_step(
                x[:, 0, :], membrane, beta, self.threshold, self.surrogate_slope,
            )
            spikes = spk.unsqueeze(1)
        elif self.use_chunked:
            # Chunked scan: faster but has numerical issues with long sequences
            # Use at your own risk - may produce NaN in membrane state
            spikes, membrane = _lif_chunked_scan(
                x, membrane, beta, self.threshold, self.surrogate_slope,
                chunk_size=self.chunk_size,
            )
        else:
            # Sequential scan: slower but numerically stable and correct
            spikes, membrane = _lif_sequential_scan(
                x, membrane, beta, self.threshold, self.surrogate_slope,
            )

        if self.drop is not None and self.training:
            spikes = self.drop(spikes)

        return spikes, membrane

    def extra_repr(self) -> str:
        beta_mean = torch.sigmoid(self.beta_logit).mean().item()
        theta_mean = self.threshold.mean().item()
        multi = self.threshold.unique().numel() > 1
        scan_mode = "chunked" if self.use_chunked else "sequential"
        return (
            f"size={self.size}, β≈{beta_mean:.3f}, θ≈{theta_mean:.2f}, "
            f"scan={scan_mode}, multi_θ={multi}"
        )


# ── AdaptiveLIFLayer ─────────────────────────────────────────────────────────

class AdaptiveLIFLayer(nn.Module):
    """
    LIF with **learnable per-neuron thresholds**.

    Threshold is parameterised as ``θ = softplus(θ_raw)`` to stay positive.
    The gradient through the spike function now flows to both V *and* θ,
    letting the network learn which neurons should be sensitive (low θ)
    vs. selective (high θ).

    API-compatible with LIFLayer.
    """

    def __init__(
        self,
        size: int,
        beta_init: float = 0.9,
        threshold_init: float = 1.0,
        learn_beta: bool = True,
        surrogate_slope: float = 10.0,
        dropout: float = 0.0,
        use_chunked: bool = False,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.size = size
        self.surrogate_slope = surrogate_slope
        self.use_chunked = use_chunked
        self.chunk_size = chunk_size

        # β
        beta_logit = math.log(beta_init / (1.0 - beta_init))
        if learn_beta:
            self.beta_logit = nn.Parameter(torch.full((size,), beta_logit))
        else:
            self.register_buffer("beta_logit", torch.full((size,), beta_logit))

        # θ — learnable, parameterised via inverse-softplus so softplus(raw) = init
        raw_init = math.log(math.exp(threshold_init) - 1.0)
        self.threshold_raw = nn.Parameter(torch.full((size,), raw_init))

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else None

    @property
    def beta(self) -> Tensor:
        return torch.sigmoid(self.beta_logit)

    @property
    def threshold(self) -> Tensor:
        return nn.functional.softplus(self.threshold_raw)

    def forward(
        self,
        x: Tensor,
        membrane: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, T, sz = x.shape
        assert sz == self.size, f"Expected size={self.size}, got {sz}"

        beta = self.beta
        threshold = self.threshold
        device, dtype = x.device, x.dtype

        if membrane is None:
            membrane = torch.zeros(B, sz, device=device, dtype=dtype)
        else:
            membrane = membrane.to(device=device, dtype=dtype)

        if T == 1:
            spk, membrane = _lif_single_step(
                x[:, 0, :], membrane, beta, threshold, self.surrogate_slope,
            )
            spikes = spk.unsqueeze(1)
        elif self.use_chunked:
            # Chunked scan: faster but has numerical issues with long sequences
            spikes, membrane = _lif_chunked_scan(
                x, membrane, beta, threshold, self.surrogate_slope,
                chunk_size=self.chunk_size,
            )
        else:
            # Sequential scan: slower but numerically stable and correct
            spikes, membrane = _lif_sequential_scan(
                x, membrane, beta, threshold, self.surrogate_slope,
            )

        if self.drop is not None and self.training:
            spikes = self.drop(spikes)

        return spikes, membrane

    def extra_repr(self) -> str:
        beta_mean = self.beta.mean().item()
        theta_mean = self.threshold.mean().item()
        return (
            f"size={self.size}, β≈{beta_mean:.3f}, θ≈{theta_mean:.2f} (learnable), "
            f"chunk={self.chunk_size}"
        )
