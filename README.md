# NeuraTensor

Fast neuromorphic inference for Python. Spiking neurons meet selective state space models — runs on CPU, CUDA, or Apple Silicon without any compilation hassles.

```bash
pip install neuratensor
```

Requires Python 3.9+ and PyTorch 2.0+.

---

## What this is

Most neuromorphic packages are either academic toys or hardware-locked C++ binaries. NeuraTensor is pure PyTorch — leaky integrate-and-fire neurons with surrogate gradients, a Mamba-style selective SSM with HiPPO initialization, and real spike sparsity (75-90% activations are zero, depending on task complexity and training). It works wherever PyTorch works, compiles with `torch.compile`, and lets you fine-tune with standard optimizers.

I built this because existing options were either impractical for production or didn't deliver on the efficiency promises. This does both.

**Sparsity note:** Random-weight models show ~90% sparsity. Trained models typically converge to 70-80% sparsity — the network learns to use more neurons for better accuracy. Use `spike_regularization` during training to trade accuracy for efficiency.

---

## Quickstart

```python
from neuratensor import InferenceEngine
import torch

engine = InferenceEngine.from_preset("small")   # 80M params, auto device
input_ids = torch.randint(0, 32000, (1, 64))
result = engine.run(input_ids)

print(f"Latency:  {result.latency_ms:.1f} ms")
print(f"Sparsity: {result.metrics['estimated_sparsity_pct']:.1f}%")
```

No config files, no CUDA extensions to compile, no surprises.

---

## Presets

| Preset   | Params | d_model | Layers | d_state | Attention | Max seq |
|----------|--------|---------|--------|---------|-----------|---------|
| `pico`   | 3.5M   | 96      | 4      | 8       | None      | 512     |
| `nano`   | 14.7M  | 288     | 6      | 16      | Every 6th | 1024    |
| `small`  | 74.4M  | 640     | 12     | 32      | Every 6th | 2048    |
| `medium` | 333M   | 1152    | 20     | 64      | Every 5th | 4096    |
| `large`  | 1.35B  | 2048    | 28     | 128     | Every 7th | 8192    |

```python
from neuratensor import list_presets
list_presets()
```

---

## Architecture (v4)

Each layer is a `FusedSpikeBlock`:

```
input → RMSNorm → expand(2×) → [LIF] → [Selective SSM] → sparse gate → project → + residual
                                                     ↓
                              Optional sliding-window attention every N layers
```

**LIF (Leaky Integrate-and-Fire):** Membrane potential accumulates input current, fires a binary spike when crossing threshold, then soft-resets. Differentiable via fast-sigmoid surrogate gradient. Decay constant β is learned per neuron. Chunked parallel scan reduces Python overhead from O(T) to O(T/chunk_size).

**Selective SSM:** Input-dependent B, C, Δ parameters (Mamba-style selection mechanism). HiPPO-inspired A matrix initialization for multi-scale temporal memory. Zero-order-hold discretization. State carries across calls for streaming inference.

**Hybrid attention:** Optional sliding-window attention with RoPE every N layers for long-range retrieval. Most layers are pure SNN-SSM (faster, sparser); attention is added only where needed.

**Sparse inference:** Output projection skips computation for positions with no spike activity during inference (~90% speedup when sparsity > 50%).

All pure PyTorch — no compiled extensions. Works with `torch.compile` out of the box.

---

## Custom model

```python
from neuratensor import NeuraTensorModel, ModelConfig

config = ModelConfig(
    vocab_size=50000,
    d_model=768,
    n_layers=16,
    d_state=32,
    expand=2,
    attn_every=4,        # attention every 4th layer
    attn_heads=8,
    attn_window=512,
    dropout=0.1,
)

model = NeuraTensorModel(config)
print(f"{model.count_parameters():,} parameters")

input_ids = torch.randint(0, 50000, (4, 128))
logits, states, metrics = model(input_ids)
# metrics: spike_rate, estimated_sparsity_pct, per_layer_spike_rate
```

---

## Streaming inference

Carry state between calls — useful for real-time or autoregressive generation:

```python
engine = InferenceEngine.from_preset("small")

# Process chunk by chunk, state flows through
for chunk in token_stream:
    result = engine.run(chunk, reset_state=False)
    yield result.logits

# Or use built-in generate
output, metrics = engine.generate(
    prompt_ids, 
    max_new_tokens=128, 
    temperature=0.8,
    top_k=50,
    repetition_penalty=1.1,
)
```

---

## Training and fine-tuning

```python
from neuratensor import NeuraTensorModel
from neuratensor.training import Trainer, TrainConfig, LoRAConfig, apply_lora

model = NeuraTensorModel.from_preset("small")

# Apply LoRA for efficient fine-tuning
apply_lora(model, LoRAConfig(r=16, alpha=32, target_modules=["in_proj", "out_proj"]))

config = TrainConfig(
    max_steps=10000,
    batch_size=8,
    learning_rate=3e-4,
    warmup_steps=500,
    gradient_accumulation_steps=4,
    spike_regularization=0.05,  # Penalize high spike rates (0.01-0.1 typical)
)

trainer = Trainer(model, config, train_data="data.jsonl")
trainer.train()

# Training will report spike rate — aim for 70-85% sparsity for good accuracy/efficiency balance

# Merge LoRA weights back into base model
from neuratensor.training import merge_lora
merge_lora(model)
```

---

## Benchmark

Run benchmarks on your own hardware:

```bash
neuratensor benchmark small --seq-length 64 --iters 100
```

**Reproducible results:** We publish automated benchmark runs from GitHub Actions CI in [`benchmarks/`](benchmarks/) (CPU verified weekly; GPU results require self-hosted runner). Results vary significantly by hardware — run locally for your deployment target.

Example format (hardware-specific, not a performance claim):

```
================================================================
  NeuraTensor SMALL | Benchmark Results
================================================================
  Device:         cuda / cpu / mps
  Dtype:          torch.bfloat16 / torch.float32
  Parameters:     [actual count from model.count_parameters()]
  Batch size:     1
  Seq length:     64
----------------------------------------------------------------
  Latency mean:   [your hardware] ms
  Latency p99:    [your hardware] ms
  Throughput:     [your hardware] seq/s
----------------------------------------------------------------
  Spike rate:     [varies by input]
  Sparsity:       [varies by input and training]
================================================================
```

**Sparsity note:** Pre-trained models show 75-90% sparsity depending on task complexity. Use `spike_regularization` during training to control the accuracy/efficiency trade-off.

---

## Inference server

```bash
pip install 'neuratensor[server]'
neuratensor serve --preset small --port 8080
```

Endpoints: `GET /health`, `GET /info`, `POST /infer`, `POST /generate`, `GET /metrics` (Prometheus format).

Docker:

```bash
docker build -t neuratensor:4.0.0 .
docker run -p 8080:8080 -e NEURATENSOR_PRESET=small neuratensor:4.0.0
```

For ECS/EKS, see [`deploy/ecs_task_definition.json`](deploy/ecs_task_definition.json).

---

## Save and load

```python
engine.save("checkpoint.pt")

from neuratensor import InferenceEngine
engine = InferenceEngine.from_checkpoint("checkpoint.pt")
```

---

## CLI

```bash
neuratensor info              # device + backend info
neuratensor presets           # list all presets with param counts
neuratensor benchmark small   # run benchmark
neuratensor serve --preset medium --port 8080
```

---

## Telemetry

NeuraTensor collects anonymous usage data (SDK version, Python version, OS, preset name, latency) to help prioritize development. No personally identifiable information is collected. Opt out anytime:

```bash
export NEURATENSOR_TELEMETRY=0
```

Or in code:

```python
import os
os.environ["NEURATENSOR_TELEMETRY"] = "0"
import neuratensor
```

---

## Support

Questions or issues? Reach us at **support@neuramorphic.ai**

Built by Peter Fullé (Lead Architect) and the team at [Neuramorphic, Inc.](https://neuramorphic.ai)

---

## License

Apache 2.0. See [LICENSE](LICENSE).
