"""
Anonymous usage telemetry for NeuraTensor SDK.

What is collected:
  - SDK version
  - Python version
  - Operating system and architecture
  - Preset / model name used
  - Inference latency (ms)
  - Spike rate
  - A random anonymous session ID (generated per install, never tied to a person)

What is NOT collected:
  - Input data, prompts, or model outputs
  - IP address (stripped at API Gateway level)
  - Any personally identifiable information

Opt out at any time:
  export NEURATENSOR_TELEMETRY=0
  # or in code:
  import os; os.environ["NEURATENSOR_TELEMETRY"] = "0"

Events are sent in a background daemon thread — they never block inference.
If the endpoint is unreachable, the error is silently ignored.

AWS backend:
  API Gateway → Lambda → DynamoDB (event log) + CloudWatch (metrics dashboard)
  Endpoint: https://telemetry.neuramorphic.ai/v1/event
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

# ── Configuration ───────────────────────────────────────────────────────────

_ENDPOINT = "https://telemetry.neuramorphic.ai/v1/event"
_TIMEOUT_S = 3.0
_CONFIG_DIR = Path.home() / ".neuratensor"
_CONFIG_FILE = _CONFIG_DIR / "config.json"
_LAST_SENT: dict = {}
_RATE_LIMITS = {"import": 0.0, "benchmark": 0.0, "inference": 60.0, "generate": 60.0}

# ── Opt-out check ───────────────────────────────────────────────────────────

def _telemetry_enabled() -> bool:
    val = os.environ.get("NEURATENSOR_TELEMETRY", "1").strip().lower()
    return val not in ("0", "false", "no", "off")

# ── Anonymous session ID ─────────────────────────────────────────────────────
# Generated once per install, stored in ~/.neuratensor/config.json
# It is a random hex string with no link to the user's identity.

def _get_session_id() -> str:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if _CONFIG_FILE.exists():
            data = json.loads(_CONFIG_FILE.read_text())
            sid = data.get("session_id", "")
            if sid and len(sid) == 64:
                return sid
        # Generate new random session ID using os.urandom (not user-derived)
        import secrets
        sid = secrets.token_hex(32)
        _CONFIG_FILE.write_text(json.dumps({"session_id": sid}))
        return sid
    except Exception:
        return "unknown"

_SESSION_ID: Optional[str] = None

def _session_id() -> str:
    global _SESSION_ID
    if _SESSION_ID is None:
        _SESSION_ID = _get_session_id()
    return _SESSION_ID

# ── Platform fingerprint (no PII) ───────────────────────────────────────────

def _platform_info() -> Dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
    }

# ── Fire-and-forget sender ───────────────────────────────────────────────────

def _send(payload: Dict[str, Any]) -> None:
    """POST payload to telemetry endpoint. Runs in a daemon thread."""
    if not _telemetry_enabled():
        return
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _ENDPOINT,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": f"neuratensor/{payload.get('version', '?')}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S):
            pass
    except Exception:
        pass  # Never surface telemetry errors to the user


def _fire(payload: Dict[str, Any]) -> None:
    """Dispatch payload in a non-blocking daemon thread."""
    event_type = payload.get("event", "")
    _rate = _RATE_LIMITS.get(event_type, 60.0)
    if _rate > 0.0:
        _now = time.time()
        if _now - _LAST_SENT.get(event_type, 0.0) < _rate:
            return
        _LAST_SENT[event_type] = _now
    t = threading.Thread(target=_send, args=(payload,), daemon=True)
    t.start()

# ── Public event functions ───────────────────────────────────────────────────

def track_import(version: str) -> None:
    """Called once when neuratensor is first imported."""
    if not _telemetry_enabled():
        return
    _fire({
        "event": "import",
        "version": version,
        "session_id": _session_id(),
        "ts": int(time.time()),
        **_platform_info(),
    })


def track_inference(
    preset: str,
    latency_ms: float,
    spike_rate: float,
    device: str,
    batch_size: int = 1,
    seq_length: int = 0,
) -> None:
    """Called after each InferenceEngine.run() call."""
    if not _telemetry_enabled():
        return
    _fire({
        "event": "inference",
        "version": _VERSION,
        "session_id": _session_id(),
        "ts": int(time.time()),
        "preset": preset,
        "latency_ms": round(latency_ms, 2),
        "spike_rate": round(spike_rate, 4),
        "device": device.split(":")[0],   # "cuda", not "cuda:0"
        "batch_size": batch_size,
        "seq_length": seq_length,
        **_platform_info(),
    })


def track_generate(preset: str, n_tokens: int, latency_ms: float) -> None:
    """Called after generate()."""
    if not _telemetry_enabled():
        return
    _fire({
        "event": "generate",
        "version": _VERSION,
        "session_id": _session_id(),
        "ts": int(time.time()),
        "preset": preset,
        "n_tokens": n_tokens,
        "latency_ms": round(latency_ms, 2),
        **_platform_info(),
    })


def track_benchmark(stats: dict) -> None:
    """Called after InferenceEngine.benchmark()."""
    if not _telemetry_enabled():
        return
    _fire({
        "event": "benchmark",
        "version": _VERSION,
        "session_id": _session_id(),
        "ts": int(time.time()),
        "preset": stats.get("preset", "unknown"),
        "mean_ms": round(stats.get("mean_ms", 0), 2),
        "spike_rate": round(stats.get("spike_rate", 0), 4),
        "device": stats.get("device", "").split(":")[0],
        **_platform_info(),
    })


# Set at package import time by __init__.py
_VERSION = "4.0.3"
