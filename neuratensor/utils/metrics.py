"""Spike activity meter — tracks sparsity across inference calls."""
from __future__ import annotations
from collections import deque
from typing import Deque


class SpikeMeter:
    """Running average of spike rate over a sliding window."""

    def __init__(self, window: int = 100) -> None:
        self._window: Deque[float] = deque(maxlen=window)

    def update(self, spike_rate: float) -> None:
        self._window.append(spike_rate)

    @property
    def mean(self) -> float:
        return sum(self._window) / len(self._window) if self._window else 0.0

    @property
    def sparsity_pct(self) -> float:
        return (1.0 - self.mean) * 100.0

    @property
    def estimated_energy_saving_pct(self) -> float:
        """Approximate energy saving vs dense activations (linear estimate)."""
        return self.sparsity_pct * 0.85

    def __repr__(self) -> str:
        return (f"SpikeMeter(rate={self.mean:.3f}, "
                f"sparsity={self.sparsity_pct:.1f}%, "
                f"energy_saved≈{self.estimated_energy_saving_pct:.1f}%)")
