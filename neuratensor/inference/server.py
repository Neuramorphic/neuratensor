"""
FastAPI inference server v4 — thread-safe with request queuing.

Endpoints:
  GET  /health          — liveness + readiness
  GET  /info            — model metadata
  POST /infer           — single inference pass
  POST /generate        — autoregressive generation
  GET  /metrics         — Prometheus-style text metrics

Thread safety: all model access goes through an asyncio lock.
Counters use threading.Lock for safe access across uvicorn workers.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "FastAPI not installed. Run: pip install 'neuratensor[server]'"
    ) from exc

import torch

from neuratensor.inference.engine import InferenceEngine


# ──────────────────────────────────────────────────────────────────────── #
#  Global state — thread-safe                                               #
# ──────────────────────────────────────────────────────────────────────── #

_engine: Optional[InferenceEngine] = None
_model_lock: asyncio.Lock = asyncio.Lock()
_start_time: float = time.time()

_stats_lock = threading.Lock()
_request_count: int = 0
_total_latency_ms: float = 0.0
_error_count: int = 0


def get_engine() -> InferenceEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _engine


def _inc_stats(latency_ms: float) -> None:
    global _request_count, _total_latency_ms
    with _stats_lock:
        _request_count += 1
        _total_latency_ms += latency_ms


def _inc_errors() -> None:
    global _error_count
    with _stats_lock:
        _error_count += 1


# ──────────────────────────────────────────────────────────────────────── #
#  Lifespan                                                                 #
# ──────────────────────────────────────────────────────────────────────── #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _model_lock
    _model_lock = asyncio.Lock()

    preset = os.getenv("NEURATENSOR_PRESET", "small")
    checkpoint = os.getenv("NEURATENSOR_CHECKPOINT", "")
    compile_model = os.getenv("NEURATENSOR_COMPILE", "0") == "1"

    if checkpoint:
        _engine = InferenceEngine.from_checkpoint(checkpoint)
    else:
        _engine = InferenceEngine.from_preset(preset, compile_model=compile_model)

    print(_engine.model.info())
    yield
    _engine = None


app = FastAPI(
    title="NeuraTensor v4 Inference API",
    description="Neuromorphic SNN-SSM inference — Neuramorphic, Inc.",
    version="4.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────── #
#  Schemas                                                                  #
# ──────────────────────────────────────────────────────────────────────── #

class InferRequest(BaseModel):
    input_ids: List[List[int]] = Field(..., description="[batch, seq] token IDs")
    reset_state: bool = Field(True, description="Reset streaming state before inference")

class InferResponse(BaseModel):
    logits_shape: List[int]
    logits_top5: List[List[float]]
    latency_ms: float
    spike_rate: float
    sparsity_pct: float
    device: str

class GenerateRequest(BaseModel):
    input_ids: List[List[int]] = Field(..., description="[batch, prompt_len] token IDs")
    max_new_tokens: int = Field(64, ge=1, le=4096)
    temperature: float = Field(1.0, ge=0.01, le=10.0)
    top_k: int = Field(50, ge=0, le=500)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=5.0)

class GenerateResponse(BaseModel):
    generated_ids: List[List[int]]
    n_new_tokens: int
    latency_ms: float


# ──────────────────────────────────────────────────────────────────────── #
#  Routes                                                                   #
# ──────────────────────────────────────────────────────────────────────── #

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "4.0.0",
        "uptime_s": round(time.time() - _start_time, 1),
        "model_loaded": _engine is not None,
    }


@app.get("/info")
async def info():
    engine = get_engine()
    cfg = engine.model.config
    return {
        "version": "4.0.0",
        "preset": cfg.model_name,
        "n_params": engine.model.count_parameters(),
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "d_state": cfg.d_state,
        "vocab_size": cfg.vocab_size,
        "max_seq_len": cfg.max_seq_len,
        "attn_every": cfg.attn_every,
        "n_attn_layers": cfg.n_attn_layers,
        "device": engine.device,
        "dtype": str(engine.dtype),
    }


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    engine = get_engine()

    try:
        ids = torch.tensor(req.input_ids, dtype=torch.long)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid input_ids: {e}")

    async with _model_lock:
        result = engine.run(ids, reset_state=req.reset_state)

    _inc_stats(result.latency_ms)

    last_logits = result.logits[:, -1, :]
    top5 = torch.topk(last_logits, k=5, dim=-1).values.tolist()

    return InferResponse(
        logits_shape=list(result.logits.shape),
        logits_top5=top5,
        latency_ms=round(result.latency_ms, 3),
        spike_rate=round(result.metrics["spike_rate"], 4),
        sparsity_pct=round(result.metrics["estimated_sparsity_pct"], 2),
        device=result.device,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    engine = get_engine()

    try:
        ids = torch.tensor(req.input_ids, dtype=torch.long)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid input_ids: {e}")

    async with _model_lock:
        t0 = time.perf_counter()
        generated = engine.generate(
            ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            reset_state=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

    _inc_stats(latency_ms)
    prompt_len = ids.shape[1]

    return GenerateResponse(
        generated_ids=generated.tolist(),
        n_new_tokens=generated.shape[1] - prompt_len,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/metrics")
async def metrics():
    engine = get_engine()
    with _stats_lock:
        count = _request_count
        total_lat = _total_latency_ms
        errors = _error_count
    avg_latency = total_lat / count if count else 0
    lines = [
        "# HELP neuratensor_requests_total Total inference requests",
        "# TYPE neuratensor_requests_total counter",
        f"neuratensor_requests_total {count}",
        "# HELP neuratensor_errors_total Total inference errors",
        "# TYPE neuratensor_errors_total counter",
        f"neuratensor_errors_total {errors}",
        "# HELP neuratensor_latency_ms_avg Average inference latency",
        "# TYPE neuratensor_latency_ms_avg gauge",
        f"neuratensor_latency_ms_avg {avg_latency:.3f}",
        "# HELP neuratensor_uptime_seconds Server uptime",
        "# TYPE neuratensor_uptime_seconds gauge",
        f"neuratensor_uptime_seconds {time.time() - _start_time:.1f}",
        "# HELP neuratensor_model_params Total model parameters",
        "# TYPE neuratensor_model_params gauge",
        f"neuratensor_model_params {engine.model.count_parameters()}",
    ]
    return PlainTextResponse("\n".join(lines))
