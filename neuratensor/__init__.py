"""
NeuraTensor v4 — Production Neuromorphic Inference SDK
========================================================

Real SNN-SSM hybrid architecture:
  • Leaky Integrate-and-Fire neurons (chunked parallel scan, surrogate gradient)
  • Selective State Space Model (input-dependent B, C, Δ — Mamba-style)
  • HiPPO-inspired A initialization for multi-scale temporal memory
  • Fused SNN→SSM residual blocks with spike-aware sparse inference
  • Optional sliding-window attention for long-range retrieval
  • Streaming inference with persistent state
  • CPU / CUDA / MPS — no compiled extensions required
  • torch.compile ready

Quick start::

    from neuratensor import NeuraTensor

    model = NeuraTensor.from_preset("small")       # 80M params
    logits, states, metrics = model(input_ids)
    print(f"Spike rate: {metrics['spike_rate']:.3f}")
    print(f"Sparsity:   {metrics['estimated_sparsity_pct']:.1f}%")

Inference engine::

    from neuratensor import InferenceEngine

    engine = InferenceEngine.from_preset("small")
    result = engine.run(input_ids)
    stats  = engine.benchmark()

Training::

    from neuratensor.training import Trainer, TrainConfig, LoRAConfig, apply_lora

    model = NeuraTensor.from_preset("small")
    apply_lora(model, LoRAConfig(r=16))
    trainer = Trainer(model, TrainConfig(max_steps=1000), train_data="data.jsonl")
    trainer.train()

© 2024-2026 Neuramorphic, Inc. — Apache 2.0
"""

__version__ = "4.0.4"
__author__ = "Neuramorphic, Inc."
__email__ = "info@neuramorphic.ai"
__license__ = "Apache-2.0"

from neuratensor.core.model import NeuraTensorModel, ModelConfig
from neuratensor.core.snn import LIFLayer, AdaptiveLIFLayer
from neuratensor.core.ssm import SelectiveSSMLayer, SSMLayer
from neuratensor.core.fused import FusedSpikeBlock, FusedSNNSSMBlock, BlockState
from neuratensor.core.model import SlidingWindowAttention, RotaryEmbedding
from neuratensor.config.presets import get_preset, list_presets, PRESETS
from neuratensor.inference.engine import InferenceEngine, InferenceResult
from neuratensor.backends.detect import detect_backend
from neuratensor import telemetry as _telemetry

# Fire import event
_telemetry._VERSION = __version__
_telemetry.track_import(__version__)

# Top-level convenience alias
NeuraTensor = NeuraTensorModel

__all__ = [
    # Model
    "NeuraTensor",
    "NeuraTensorModel",
    "ModelConfig",
    # Layers
    "LIFLayer",
    "AdaptiveLIFLayer",
    "SelectiveSSMLayer",
    "SSMLayer",
    "FusedSpikeBlock",
    "FusedSNNSSMBlock",
    "BlockState",
    # Attention
    "SlidingWindowAttention",
    "RotaryEmbedding",
    # Config
    "get_preset",
    "list_presets",
    "PRESETS",
    # Inference
    "InferenceEngine",
    "InferenceResult",
    # Backend
    "detect_backend",
    # Version
    "__version__",
]
