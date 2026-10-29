from __future__ import annotations
import torch


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info() -> dict:
    info: dict = {"device": best_device(), "torch": torch.__version__}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": props.name,
            "gpu_memory_gb": round(props.total_memory / 1024 ** 3, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "bf16_supported": props.major >= 8,
        })
    return info


def print_device_info() -> None:
    info = device_info()
    print("=" * 48)
    print("  Device Information")
    print("=" * 48)
    for k, v in info.items():
        print(f"  {k:<22}: {v}")
    print("=" * 48)
