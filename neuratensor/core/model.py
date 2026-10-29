"""
NeuraTensorModel v4 — autoregressive language model built from FusedSpikeBlocks.

Architecture:
    TokenEmbedding(vocab, d_model)
    ──► N × FusedSpikeBlock(d_model, d_state, expand)
    ──► [Optional: SlidingWindowAttention every attn_every layers]
    ──► RMSNorm(d_model)
    ──► Linear(d_model, vocab)   [weight-tied to embedding]

v4 changes:
  • Rotary position embeddings (RoPE) applied in attention layers
  • Optional sliding-window attention blocks for long-range retrieval
  • Improved weight initialization
  • Configurable attention layer placement
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .fused import FusedSpikeBlock, BlockState


@dataclass
class ModelConfig:
    """Full model configuration."""
    vocab_size: int = 32_000
    d_model: int = 512
    n_layers: int = 8
    d_state: int = 16
    expand: int = 2
    beta_init: float = 0.9
    learn_beta: bool = True
    dropout: float = 0.0
    tie_weights: bool = True
    max_seq_len: int = 2048

    # Hybrid attention
    attn_every: int = 0
    attn_heads: int = 4
    attn_window: int = 256

    # Metadata
    model_name: str = "custom"

    @property
    def n_attn_layers(self) -> int:
        if self.attn_every <= 0:
            return 0
        return self.n_layers // self.attn_every

    def n_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        emb = self.vocab_size * self.d_model
        d_inner = self.d_model * self.expand

        per_spike_block = (
            self.d_model * 2                             # RMSNorm
            + self.d_model * d_inner * 2 * 2             # in_proj (2× for gate), out_proj
            + d_inner                                    # snn beta_logit
            + d_inner * self.d_state * 2                 # ssm B_proj, C_proj
            + d_inner * d_inner + d_inner                # ssm dt_proj (weight + bias)
            + d_inner * self.d_state                     # ssm log_A
            + d_inner                                    # ssm D
            + 1                                          # out_scale
        )

        per_attn = (
            self.d_model * 2                             # norm
            + self.d_model * self.d_model * 4            # q, k, v, o projections
        ) if self.attn_every > 0 else 0

        n_spike = self.n_layers
        n_attn = self.n_attn_layers

        head = 0 if self.tie_weights else self.vocab_size * self.d_model
        return emb + n_spike * per_spike_block + n_attn * per_attn + head


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self._max_cached:
            return
        self._max_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        """Returns (cos, sin) for positions [offset, offset + T)."""
        T = x.shape[-2]
        self._build_cache(offset + T, x.device, x.dtype)
        return (
            self.cos_cached[offset:offset + T],
            self.sin_cached[offset:offset + T],
        )


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to x: [B, n_heads, T, head_dim]."""
    return x * cos + _rotate_half(x) * sin


class SlidingWindowAttention(nn.Module):
    """Multi-head attention with sliding window for bounded memory.

    Used every `attn_every` layers for long-range retrieval that pure SSM cannot do.
    Window size bounds the O(T·W) cost instead of O(T²).
    """

    def __init__(self, d_model: int, n_heads: int = 4, window: int = 256, max_seq_len: int = 8192) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window

        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        B, T, _ = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, n_heads, head_dim]
        q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Sliding window: use causal mask with window limit
        if T <= self.window:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Build sliding window causal mask
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            mask = mask & (torch.arange(T, device=x.device).unsqueeze(0)
                          - torch.arange(T, device=x.device).unsqueeze(1)).abs().le(self.window)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.o_proj(attn_out) + residual


class NeuraTensorModel(nn.Module):
    """
    NeuraTensor v4 — neuromorphic language model.

    Hybrid architecture: FusedSpikeBlocks (SNN→SSM) with optional
    sliding-window attention every K layers for long-range retrieval.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Build layer stack: SpikeBlocks with optional attention interleaved
        self.blocks = nn.ModuleList()
        self.attn_layers = nn.ModuleDict()

        for i in range(config.n_layers):
            self.blocks.append(
                FusedSpikeBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    expand=config.expand,
                    beta_init=config.beta_init,
                    learn_beta=config.learn_beta,
                    dropout=config.dropout,
                )
            )
            # Insert attention after every attn_every blocks
            if config.attn_every > 0 and (i + 1) % config.attn_every == 0:
                self.attn_layers[str(i)] = SlidingWindowAttention(
                    d_model=config.d_model,
                    n_heads=config.attn_heads,
                    window=config.attn_window,
                    max_seq_len=config.max_seq_len,
                )

        self.norm = nn.RMSNorm(config.d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.in_proj.weight)
            nn.init.xavier_uniform_(block.out_proj.weight)
        for attn in self.attn_layers.values():
            nn.init.xavier_uniform_(attn.qkv.weight)
            nn.init.xavier_uniform_(attn.o_proj.weight)

    def forward(
        self,
        input_ids: Tensor,
        states: Optional[List[Optional[BlockState]]] = None,
    ) -> Tuple[Tensor, List[BlockState], Dict]:
        """
        Args:
            input_ids: [B, T]  token IDs
            states:    list of per-block BlockState (None = fresh start)

        Returns:
            logits:  [B, T, vocab_size]
            states:  updated list of BlockState
            metrics: spike rates per layer and global average
        """
        if states is None:
            states = [None] * len(self.blocks)

        x = self.embedding(input_ids)

        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_seq_len:
            import warnings
            warnings.warn(
                f"Input sequence length {seq_len} exceeds max_seq_len "
                f"{self.config.max_seq_len}; truncating.",
                UserWarning, stacklevel=2,
            )
            input_ids = input_ids[:, :self.config.max_seq_len]
            x = self.embedding(input_ids)

        new_states: List[BlockState] = []
        all_metrics: List[dict] = []

        for i, (block, state) in enumerate(zip(self.blocks, states)):
            x, new_state, m = block(x, state)
            new_states.append(new_state)
            all_metrics.append(m)

            # Apply attention after this block if configured
            if str(i) in self.attn_layers:
                x = self.attn_layers[str(i)](x)

        x = self.norm(x)
        logits = self.lm_head(x)

        mean_spike_rate = sum(m["spike_rate"] for m in all_metrics) / len(all_metrics)
        metrics = {
            "spike_rate": mean_spike_rate,
            "per_layer_spike_rate": [m["spike_rate"] for m in all_metrics],
            "estimated_sparsity_pct": (1.0 - mean_spike_rate) * 100,
        }

        return logits, new_states, metrics

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        states: Optional[List[Optional[BlockState]]] = None,
    ) -> Tuple[Tensor, List[BlockState]]:
        """
        Autoregressive token generation with streaming state.

        Args:
            input_ids:          [B, T]  prompt token IDs
            max_new_tokens:     how many tokens to generate
            temperature:        sampling temperature
            top_k:              top-k sampling (0 = greedy)
            top_p:              nucleus sampling threshold (1.0 = disabled)
            repetition_penalty: penalize repeated tokens (1.0 = disabled)
            stop_tokens:        list of token IDs to stop generation
            states:             initial streaming states
        """
        self.eval()
        generated = input_ids
        stop_set = set(stop_tokens or [])

        # Process prompt in one shot
        logits, states, _ = self(generated, states)
        next_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    for token_id in generated[b].tolist():
                        if next_logits[b, token_id] > 0:
                            next_logits[b, token_id] /= repetition_penalty
                        else:
                            next_logits[b, token_id] *= repetition_penalty

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                cumul = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumul - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                next_logits = torch.zeros_like(next_logits).scatter_(-1, sorted_idx, sorted_logits)

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Check stop tokens
            if stop_set and next_token.item() in stop_set:
                break

            # Step one token, carry state
            logits, states, _ = self(next_token, states)
            next_logits = logits[:, -1, :]

        return generated, states

    def info(self) -> str:
        c = self.config
        lines = [
            "=" * 60,
            f"  NeuraTensor v4 — {c.model_name.upper()}",
            "=" * 60,
            f"  Parameters:    {self.count_parameters():>14,}",
            f"  Layers:        {c.n_layers:>14}",
            f"  d_model:       {c.d_model:>14}",
            f"  d_state:       {c.d_state:>14}",
            f"  expand:        {c.expand:>14}",
            f"  vocab:         {c.vocab_size:>14,}",
            f"  max_seq_len:   {c.max_seq_len:>14}",
        ]
        if c.attn_every > 0:
            lines.append(f"  attn_every:    {c.attn_every:>14}")
            lines.append(f"  attn_heads:    {c.attn_heads:>14}")
            lines.append(f"  attn_window:   {c.attn_window:>14}")
            lines.append(f"  attn_layers:   {c.n_attn_layers:>14}")
        lines.append("=" * 60)
        return "\n".join(lines)


# Convenience alias
NeuraTensor = NeuraTensorModel
