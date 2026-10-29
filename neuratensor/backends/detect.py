from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class BackendInfo:
    name: str
    device: str
    dtype: torch.dtype
    gpu_name: Optional[str] = None
    compute_capability: Optional[str] = None
    memory_gb: Optional[float] = None
    bf16: bool = False
    compile_ready: bool = False
    notes: list = field(default_factory=list)

    def print(self) -> None:
        print("=" * 52)
        print(f"  Backend: {self.name}")
        print(f"  Device:  {self.device}  dtype={self.dtype}")
        if self.gpu_name:
            print(f"  GPU:     {self.gpu_name}")
            print(f"  CC:      {self.compute_capability}")
            print(f"  VRAM:    {self.memory_gb:.1f} GB")
            print(f"  BF16:    {self.bf16}")
        print(f"  Compile: {self.compile_ready}")
        for n in self.notes:
            print(f"  ⚠  {n}")
        print("=" * 52)


def detect_backend() -> BackendInfo:
    compile_ready = hasattr(torch, "compile")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        bf16 = props.major >= 8
        dtype = torch.bfloat16 if bf16 else torch.float16
        name = "jetson_orin" if "Orin" in props.name else (
               "jetson_agx" if "Xavier" in props.name else "cuda")
        return BackendInfo(
            name=name,
            device="cuda",
            dtype=dtype,
            gpu_name=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            memory_gb=props.total_memory / 1024 ** 3,
            bf16=bf16,
            compile_ready=compile_ready,
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return BackendInfo(
            name="apple_silicon",
            device="mps",
            dtype=torch.bfloat16,
            bf16=True,
            compile_ready=False,
            notes=["MPS does not support torch.compile yet"],
        )

    return BackendInfo(
        name="cpu",
        device="cpu",
        dtype=torch.float32,
        compile_ready=compile_ready,
        notes=["No GPU detected — running on CPU"],
    )
