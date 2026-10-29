"""
NeuraTensor CLI — neuratensor <command> [options]

Commands:
  info        System + backend information
  benchmark   Latency / throughput benchmark
  serve       Start FastAPI inference server
  presets     List all model presets
  version     Print version
"""

from __future__ import annotations

import argparse
import sys


def cmd_info(args: argparse.Namespace) -> int:
    from neuratensor.backends.detect import detect_backend
    from neuratensor.utils.device import print_device_info
    import neuratensor
    print(f"\nNeuraTensor v{neuratensor.__version__}\n")
    print_device_info()
    print()
    detect_backend().print()
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    from neuratensor.config.presets import list_presets
    list_presets()
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from neuratensor.inference.engine import InferenceEngine

    print(f"\nLoading preset '{args.preset}'…")
    engine = InferenceEngine.from_preset(args.preset, compile_model=args.compile)
    engine.print_info()

    print(f"Benchmarking  batch={args.batch_size}  seq={args.seq_length}  "
          f"iters={args.iters}  warmup={args.warmup}\n")

    stats = engine.benchmark(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        n_iter=args.iters,
        n_warmup=args.warmup,
    )

    print("=" * 56)
    print(f"  NeuraTensor {args.preset.upper()} | Benchmark Results")
    print("=" * 56)
    print(f"  Device:         {stats['device']}")
    print(f"  Dtype:          {stats['dtype']}")
    print(f"  Parameters:     {stats['n_params']:,}")
    print(f"  Batch size:     {stats['batch_size']}")
    print(f"  Seq length:     {stats['seq_length']}")
    print("-" * 56)
    print(f"  Latency mean:   {stats['mean_ms']:.2f} ms ± {stats['std_ms']:.2f} ms")
    print(f"  Latency p99:    {stats['p99_ms']:.2f} ms")
    print(f"  Latency min:    {stats['min_ms']:.2f} ms")
    print(f"  Throughput:     {stats['throughput_seq_per_s']:.1f} seq/s")
    print("-" * 56)
    print(f"  Spike rate:     {stats['spike_rate']:.3f}")
    print(f"  Sparsity:       {stats['sparsity_pct']:.1f}%")
    print("=" * 56)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install 'neuratensor[server]'")
        return 1

    import os
    os.environ["NEURATENSOR_PRESET"] = args.preset
    if args.checkpoint:
        os.environ["NEURATENSOR_CHECKPOINT"] = args.checkpoint
    if args.compile:
        os.environ["NEURATENSOR_COMPILE"] = "1"

    uvicorn.run(
        "neuratensor.inference.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )
    return 0


def main() -> int:
    import neuratensor

    parser = argparse.ArgumentParser(
        prog="neuratensor",
        description="NeuraTensor v4 — neuromorphic inference SDK",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {neuratensor.__version__}")
    sub = parser.add_subparsers(dest="cmd", metavar="command")

    # info
    sub.add_parser("info", help="Show system and backend information")

    # presets
    sub.add_parser("presets", help="List available model presets")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark latency and throughput")
    p_bench.add_argument("preset", default="small", nargs="?", help="Model preset")
    p_bench.add_argument("--batch-size", type=int, default=1)
    p_bench.add_argument("--seq-length", type=int, default=64)
    p_bench.add_argument("--iters", type=int, default=50)
    p_bench.add_argument("--warmup", type=int, default=10)
    p_bench.add_argument("--compile", action="store_true", help="torch.compile the model")

    # serve
    p_serve = sub.add_parser("serve", help="Start FastAPI inference server")
    p_serve.add_argument("--preset", default="small")
    p_serve.add_argument("--checkpoint", default="", help="Path to .pt checkpoint")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--workers", type=int, default=1)
    p_serve.add_argument("--compile", action="store_true")

    args = parser.parse_args()

    dispatch = {
        "info": cmd_info,
        "presets": cmd_presets,
        "benchmark": cmd_benchmark,
        "serve": cmd_serve,
    }

    if args.cmd not in dispatch:
        parser.print_help()
        return 1

    try:
        return dispatch[args.cmd](args) or 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
