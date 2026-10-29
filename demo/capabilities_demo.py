#!/usr/bin/env python3
"""
NeuraTensor v3 — Capabilities Demo
=====================================
Sections: Architecture Overview | Spike Raster | Throughput | Linear Scaling
          torch.compile Speedup | InferenceEngine End-to-End

Run from repo root (no pip install required):
    python demo/capabilities_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# ── Utility helpers ──────────────────────────────────────────────────────── #

def _detect_device() -> str:
    """Return the best available device string: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _best_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        return torch.bfloat16 if props.major >= 8 else torch.float16
    return torch.float32


def _timer(device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def _elapsed_ms(t0: float, device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _box(title: str, width: int = 62) -> None:
    """Print a section header with box-drawing characters."""
    inner = f"  {title}  "
    pad   = max(0, width - len(inner) - 2)
    left  = pad // 2
    right = pad - left
    print()
    print("┌" + "─" * (width - 2) + "┐")
    print("│" + " " * left + inner + " " * right + "│")
    print("└" + "─" * (width - 2) + "┘")


def _bar(value: float, max_value: float, width: int = 30) -> str:
    """Return a filled-block progress bar string."""
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


# ── Import neuratensor ───────────────────────────────────────────────────── #

try:
    from neuratensor import (
        NeuraTensor,
        InferenceEngine,
        get_preset,
        __version__ as NT_VERSION,
    )
except ImportError as exc:
    print(f"\n[ERROR] Could not import neuratensor: {exc}")
    print("        Run from the repo root:  python demo/capabilities_demo.py")
    print("        Or install first:        pip install -e .")
    sys.exit(1)



# ── Main demo ────────────────────────────────────────────────────────────── #

def main() -> None:

    DEVICE = _detect_device()
    DTYPE  = _best_dtype(DEVICE)

    if DEVICE == "cuda":
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif DEVICE == "mps":
        device_label = "MPS (Apple Silicon)"
    else:
        device_label = "CPU"

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          NeuraTensor v3 — Capabilities Demo                  ║")
    print("║     Spiking Neural Network  ×  State Space Model             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  NeuraTensor : v{NT_VERSION}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Device      : {device_label}")
    print(f"  Dtype       : {DTYPE}")

    # ────────────────────────────────────────────────────────────────────── #
    #  SECTION 1 — Architecture Overview                                      #
    # ────────────────────────────────────────────────────────────────────── #
    _box("SECTION 1 — Architecture Overview")

    cfg   = get_preset("nano")
    model = NeuraTensor(cfg)
    n_params = model.count_parameters()

    print(f"  Preset      : {cfg.model_name}")
    print(f"  d_model     : {cfg.d_model}")
    print(f"  n_layers    : {cfg.n_layers}")
    print(f"  d_state     : {cfg.d_state}  (SSM state size)")
    print(f"  expand      : {cfg.expand}  → d_inner = {cfg.d_model * cfg.expand}")
    print(f"  vocab_size  : {cfg.vocab_size:,}")
    print(f"  max_seq_len : {cfg.max_seq_len}")
    print(f"  Parameters  : {n_params:,}")
    print(f"  Device      : {DEVICE}")
    print(f"  Dtype       : {DTYPE}")
    print()
    print("  Stack:")
    print(f"    TokenEmbedding({cfg.vocab_size:,}, {cfg.d_model})")
    print(f"    x{cfg.n_layers}  FusedSNNSSMBlock: LIFLayer -> SSMLayer -> gate -> residual")
    print(f"    RMSNorm({cfg.d_model})")
    print(f"    Linear({cfg.d_model}, {cfg.vocab_size:,})  [weight-tied to embedding]")

    # ────────────────────────────────────────────────────────────────────── #
    #  SECTION 2 — Spike Pattern Visualization                                #
    # ────────────────────────────────────────────────────────────────────── #
    _box("SECTION 2 — Spike Pattern Visualization")

    N_COLS = 20   # timesteps shown
    N_ROWS = 16   # neurons shown

    # Register a forward hook on blocks[0].snn to capture the raw spike tensor.
    # LIFLayer.forward returns (spikes [B, T, d_inner], membrane [B, d_inner]).
    _captured: list[torch.Tensor] = []

    def _spike_hook(_module, _inputs, outputs):  # type: ignore[override]
        _captured.clear()
        _captured.append(outputs[0].detach().cpu())  # spikes: [B, T, d_inner]

    hook = model.blocks[0].snn.register_forward_hook(_spike_hook)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        logits, states, metrics = model(input_ids)

    hook.remove()

    if not _captured:
        print("  [WARN] Spike hook did not fire — skipping raster plot.")
    else:
        spikes = _captured[0].squeeze(0)  # [T, d_inner]
        t_avail = spikes.shape[0]
        n_avail = spikes.shape[1]
        cols = min(N_COLS, t_avail)
        rows = min(N_ROWS, n_avail)

        print("  First FusedSNNSSMBlock — LIF neuron spike raster")
        print(f"  (showing {cols} timesteps × {rows} neurons)")
        print()
        # Header: timestep indices
        header_nums = "".join(f"{t % 10}" for t in range(cols))
        print(f"  Neuron  t→  {header_nums}")
        print("  " + "─" * (cols + 16))

        total_spikes = 0
        total_cells  = 0
        for n in range(rows):
            cells = []
            for t in range(cols):
                fired = spikes[t, n].item() > 0.5
                cells.append("█" if fired else "·")
                if fired:
                    total_spikes += 1
                total_cells += 1
            print(f"  [{n:>2}]        {''.join(cells)}")

        win_rate    = total_spikes / total_cells if total_cells > 0 else 0.0
        win_sparsity = (1.0 - win_rate) * 100.0

        print()
        print(f"  Window spike rate  : {win_rate:.3f}  ({win_sparsity:.1f}% silent)")
        print(f"  Global spike rate  : {metrics['spike_rate']:.3f}  "
              f"({metrics['estimated_sparsity_pct']:.1f}% sparsity, all layers)")
        print()
        print("  █ = neuron fired    · = silent")

    # ────────────────────────────────────────────────────────────────────── #
    #  SECTION 3 — Throughput Benchmark                                       #
    # ────────────────────────────────────────────────────────────────────── #
    _box("SECTION 3 — Throughput Benchmark  (seq_len=64)")

    BENCH_SEQ    = 64
    BENCH_WARMUP = 20
    BENCH_TIMED  = 50
    bench_tok_s: dict[str, float] = {}

    for preset_name in ("nano", "small"):
        print(f"  Running '{preset_name}' ...", end="", flush=True)
        try:
            engine = InferenceEngine.from_preset(preset_name, device=DEVICE)
            ids    = torch.randint(0, 32_000, (1, BENCH_SEQ))

            for _ in range(BENCH_WARMUP):
                engine.run(ids, reset_state=True)

            latencies: list[float] = []
            for _ in range(BENCH_TIMED):
                t0 = _timer(DEVICE)
                engine.run(ids, reset_state=True)
                latencies.append(_elapsed_ms(t0, DEVICE))

            mean_ms = sum(latencies) / len(latencies)
            tok_s   = BENCH_SEQ * 1000.0 / mean_ms
            bench_tok_s[preset_name] = tok_s
            print(f"  mean={mean_ms:.2f} ms  →  {tok_s:,.0f} tok/s")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            bench_tok_s[preset_name] = 0.0

    # ASCII bar chart
    max_tok = max(bench_tok_s.values()) if bench_tok_s else 1.0
    print()
    print("  Throughput (tokens / second):")
    for pname, tok_s in bench_tok_s.items():
        bar = _bar(tok_s, max_tok, width=30)
        print(f"  {pname:<6}  {bar}  {tok_s:>9,.0f} tok/s")

    # ────────────────────────────────────────────────────────────────────── #
    #  SECTION 4 — Linear Scaling  (SNN is O(T), not O(T²))                  #
    # ────────────────────────────────────────────────────────────────────── #
    _box("SECTION 4 — Linear Scaling  (SNN is O(T), not O(T²))")

    SEQ_LENS   = [16, 32, 64, 128, 256]
    SCALE_WARM = 5
    SCALE_ITER = 20
    baseline_ms: float | None = None

    print("  Model: nano    Warmup:", SCALE_WARM, "   Timed runs:", SCALE_ITER)
    print()
    print(f"  {'seq_len':>8}  {'latency_ms':>12}  {'tok/s':>10}  ratio vs T=16")
    print("  " + "─" * 58)

    try:
        scale_engine = InferenceEngine.from_preset("nano", device=DEVICE)

        for seq_len in SEQ_LENS:
            ids = torch.randint(0, 32_000, (1, seq_len))

            for _ in range(SCALE_WARM):
                scale_engine.run(ids, reset_state=True)

            lats: list[float] = []
            for _ in range(SCALE_ITER):
                t0 = _timer(DEVICE)
                scale_engine.run(ids, reset_state=True)
                lats.append(_elapsed_ms(t0, DEVICE))

            mean_ms = sum(lats) / len(lats)
            tok_s   = seq_len * 1000.0 / mean_ms

            if baseline_ms is None:
                baseline_ms = mean_ms
                ratio_col   = "1.00x  (baseline)"
            else:
                observed   = mean_ms / baseline_ms
                linear_exp = seq_len / SEQ_LENS[0]
                ratio_col  = f"{observed:.2f}x  (linear ideal: {linear_exp:.1f}x)"

            print(f"  {seq_len:>8}  {mean_ms:>12.2f}  {tok_s:>10,.0f}  {ratio_col}")

    except Exception as exc:
        print(f"  ERROR during scaling test: {exc}")

    print()
    print("  Transformer (attention) scales as O(T²): doubling T → 4× slower.")
    print("  SNN-SSM scales as O(T): doubling T → ~2× slower (linear).")
    print("  This makes long-context inference dramatically cheaper.")

    # ────────────────────────────────────────────────────────────────────── #
    #  SECTION 5 — torch.compile Speedup                                      #
    # ────────────────────────────────────────────────────────────────────── #
    _box("SECTION 5 — torch.compile Speedup")

    COMP_SEQ   = 64
    COMP_WARM  = 10
    COMP_ITER  = 30

    try:
        eager_engine = InferenceEngine.from_preset("nano", device=DEVICE)
        ids_c = torch.randint(0, 32_000, (1, COMP_SEQ))

        for _ in range(COMP_WARM):
            eager_engine.run(ids_c, reset_state=True)

        eager_lats: list[float] = []
        for _ in range(COMP_ITER):
            t0 = _timer(DEVICE)
            eager_engine.run(ids_c, reset_state=True)
            eager_lats.append(_elapsed_ms(t0, DEVICE))

        eager_ms  = sum(eager_lats) / len(eager_lats)
        eager_tok = COMP_SEQ * 1000.0 / eager_ms
        print(f"  Eager mode   : {eager_ms:>8.3f} ms   {eager_tok:>10,.0f} tok/s")

        if not hasattr(torch, "compile"):
            print("  torch.compile: not available — requires PyTorch >= 2.0")
        else:
            print("  Compiling model (first call triggers tracing) ...", flush=True)
            compiled_engine = InferenceEngine.from_preset(
                "nano", device=DEVICE, compile_model=True
            )

            # Warmup includes compilation tracing time — don't measure it.
            for _ in range(COMP_WARM):
                compiled_engine.run(ids_c, reset_state=True)

            comp_lats: list[float] = []
            for _ in range(COMP_ITER):
                t0 = _timer(DEVICE)
                compiled_engine.run(ids_c, reset_state=True)
                comp_lats.append(_elapsed_ms(t0, DEVICE))

            comp_ms  = sum(comp_lats) / len(comp_lats)
            comp_tok = COMP_SEQ * 1000.0 / comp_ms
            speedup  = eager_ms / comp_ms if comp_ms > 0 else 1.0
            print(f"  Compiled     : {comp_ms:>8.3f} ms   {comp_tok:>10,.0f} tok/s")
            print(f"  Speedup      : {speedup:.2f}x")
            if speedup < 1.05:
                print("  Note: compile gains are modest at small seq_len on CPU;")
                print("        larger benefits appear on GPU with longer sequences.")

    except (RuntimeError, Exception) as exc:
        err_name = type(exc).__name__
        print(f"  torch.compile not available: {err_name}")
        print(f"  Detail : {exc}")

    # ────────────────────────────────────────────────────────────────────── #
    #  SECTION 6 — InferenceEngine End-to-End Demo                            #
    # ────────────────────────────────────────────────────────────────────── #
    _box("SECTION 6 — InferenceEngine End-to-End Demo")

    try:
        demo_engine = InferenceEngine.from_preset("nano", device=DEVICE)
        demo_ids    = torch.randint(0, 256, (1, 16))

        result = demo_engine.run(demo_ids)

        print(f"  Preset       : nano")
        print(f"  Input shape  : {list(demo_ids.shape)}  (batch=1, seq=16)")
        print(f"  Output shape : {list(result.logits.shape)}")
        print(f"  Latency      : {result.latency_ms:.3f} ms")
        print(f"  Device       : {result.device}")
        print(f"  Spike rate   : {result.metrics['spike_rate']:.4f}")
        print(f"  Sparsity     : {result.metrics['estimated_sparsity_pct']:.1f}%")
        print()
        print("  Per-layer spike rates:")
        for i, rate in enumerate(result.metrics["per_layer_spike_rate"]):
            bar = _bar(rate, 1.0, width=20)
            print(f"    Layer {i}: {bar}  {rate:.3f}")
        print()
        print("  [NOTE] No pre-trained weights loaded — output logits are random.")
        print("         This run verifies the architecture is wired correctly.")
        print("         Use InferenceEngine.from_checkpoint(path) for real inference.")

    except Exception as exc:
        print(f"  ERROR: {exc}")

    # ──────────────────────────────────────────────────────────────────── #
    #  Footer                                                               #
    # ──────────────────────────────────────────────────────────────────── #
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    Demo complete.                            ║")
    print("║  NeuraTensor v3 — Apache 2.0 — neuramorphic.ai               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
