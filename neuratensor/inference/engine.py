"""
High-performance inference engine for NeuraTensorModel v4.

Features:
  • Auto device placement (CUDA > MPS > CPU)
  • FP16 / BF16 autocast
  • torch.compile (when torch >= 2.0)
  • Streaming state — carry BlockState between calls
  • Latency benchmarking with warmup
  • Spike-rate and energy-efficiency reporting
  • from_pretrained() for model hub downloads
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from neuratensor.core.model import NeuraTensorModel, ModelConfig
from neuratensor.core.fused import BlockState
from neuratensor.config.presets import get_preset
from neuratensor import telemetry as _telemetry


@dataclass
class InferenceResult:
    """Single inference output."""
    logits: Tensor
    states: List[Optional[BlockState]]
    latency_ms: float
    metrics: Dict
    device: str


class InferenceEngine:
    """
    Wraps NeuraTensorModel with device management, precision, and compilation.

    Example::

        engine = InferenceEngine.from_preset("small")
        result = engine.run(input_ids)
        print(f"Latency: {result.latency_ms:.2f} ms")
        print(f"Spike rate: {result.metrics['spike_rate']:.3f}")
    """

    def __init__(
        self,
        model: NeuraTensorModel,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
    ) -> None:
        self.device = device or _best_device()
        self.dtype = dtype or _best_dtype(self.device)
        self._states: List[Optional[BlockState]] = [None] * len(model.blocks)

        model = model.to(self.device).to(self.dtype)
        model.eval()

        if compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)

        self.model = model

    # ------------------------------------------------------------------ #
    #  Factory methods                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_preset(
        cls,
        name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
    ) -> "InferenceEngine":
        config = get_preset(name)
        model = NeuraTensorModel(config)
        return cls(model, device=device, dtype=dtype, compile_model=compile_model)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "InferenceEngine":
        import torch.serialization as _ser
        _ser.add_safe_globals([ModelConfig])
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = ckpt["config"]
        model = NeuraTensorModel(config)
        model.load_state_dict(ckpt["model"])
        return cls(model, device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "InferenceEngine":
        """Download and load a pre-trained model from the Neuramorphic model hub.

        Args:
            name:      model identifier (e.g. "neuramorphic/spike-small")
            cache_dir: where to cache downloaded checkpoints
            device:    override device
            dtype:     override dtype
        """
        cache = Path(cache_dir or os.path.expanduser("~/.neuratensor/models"))
        cache.mkdir(parents=True, exist_ok=True)

        slug = name.replace("/", "--")
        ckpt_path = cache / f"{slug}.pt"

        if not ckpt_path.exists():
            url = f"https://models.neuramorphic.ai/{name}/model.pt"
            print(f"Downloading {name} from {url} ...")
            import urllib.request
            try:
                urllib.request.urlretrieve(url, str(ckpt_path))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download model '{name}'. "
                    f"Check the model name or download manually to {ckpt_path}"
                ) from e

        return cls.from_checkpoint(str(ckpt_path), device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def run(
        self,
        input_ids: Tensor,
        reset_state: bool = False,
    ) -> InferenceResult:
        if reset_state:
            self.reset_state()

        ids = input_ids.to(self.device)

        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype, enabled=self.dtype != torch.float32):
            t0 = _timer(self.device)
            logits, new_states, metrics = self.model(ids, self._states)
            latency_ms = _elapsed_ms(t0, self.device)

        self._states = new_states

        _telemetry.track_inference(
            preset=self.model.config.model_name,
            latency_ms=latency_ms,
            spike_rate=metrics.get("spike_rate", 0.0),
            device=self.device,
            batch_size=int(ids.shape[0]),
            seq_length=int(ids.shape[1]),
        )

        return InferenceResult(
            logits=logits,
            states=new_states,
            latency_ms=latency_ms,
            metrics=metrics,
            device=self.device,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        reset_state: bool = True,
    ) -> Tensor:
        """Generate tokens autoregressively. Returns [B, T + max_new_tokens]."""
        if reset_state:
            self.reset_state()

        ids = input_ids.to(self.device)
        generated, final_states = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
            states=self._states,
        )
        self._states = final_states
        return generated

    # ------------------------------------------------------------------ #
    #  Benchmarking                                                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def benchmark(
        self,
        batch_size: int = 1,
        seq_length: int = 64,
        n_iter: int = 50,
        n_warmup: int = 10,
    ) -> Dict:
        vocab = self.model.config.vocab_size
        ids = torch.randint(0, vocab, (batch_size, seq_length), device=self.device)

        for _ in range(n_warmup):
            self.run(ids, reset_state=True)

        latencies = []
        spike_rates = []
        for _ in range(n_iter):
            result = self.run(ids, reset_state=True)
            latencies.append(result.latency_ms)
            spike_rates.append(result.metrics["spike_rate"])

        mean_ms = statistics.mean(latencies)
        throughput = batch_size * 1000.0 / mean_ms

        result_stats = {
            "mean_ms": mean_ms,
            "std_ms": statistics.stdev(latencies) if n_iter > 1 else 0.0,
            "min_ms": min(latencies),
            "p99_ms": sorted(latencies)[int(n_iter * 0.99)],
            "throughput_seq_per_s": throughput,
            "spike_rate": statistics.mean(spike_rates),
            "sparsity_pct": (1.0 - statistics.mean(spike_rates)) * 100,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "device": self.device,
            "dtype": str(self.dtype),
            "n_params": self.model.count_parameters(),
            "preset": self.model.config.model_name,
        }
        _telemetry.track_benchmark(result_stats)
        return result_stats

    # ------------------------------------------------------------------ #
    #  State management                                                    #
    # ------------------------------------------------------------------ #

    def reset_state(self) -> None:
        self._states = [None] * len(self.model.blocks)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save({
            "config": self.model.config,
            "model": self.model.state_dict(),
            "version": "4.0.0",
        }, path)

    def print_info(self) -> None:
        print(self.model.info())
        print(f"  Device:  {self.device}")
        print(f"  Dtype:   {self.dtype}")
        compiled = hasattr(self.model, "_orig_mod")
        print(f"  Compiled:{compiled}")
        print()


# ──────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                  #
# ──────────────────────────────────────────────────────────────────────── #

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _best_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        return torch.bfloat16 if props.major >= 8 else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def _timer(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def _elapsed_ms(t0: float, device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0
