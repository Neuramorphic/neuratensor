"""
Model presets — pico / nano / small / medium / large.

v4 presets include optional hybrid attention layers for long-range retrieval.
All presets use a 32k vocabulary and tie the LM head to the embedding.

Approximate parameter counts (with tie_weights=True):
  pico:   ~3M    (no attention)
  nano:   ~15M   (1 attention layer)
  small:  ~80M   (2 attention layers)
  medium: ~350M  (4 attention layers)
  large:  ~1.3B  (6 attention layers)
"""

from neuratensor.core.model import ModelConfig

PRESETS: dict[str, ModelConfig] = {
    # ── 3M — tiny, for unit tests and edge devices ───────────────────────
    "pico": ModelConfig(
        model_name="pico",
        vocab_size=32_000,
        d_model=96,
        n_layers=4,
        d_state=8,
        expand=2,
        beta_init=0.9,
        learn_beta=True,
        dropout=0.0,
        max_seq_len=512,
        attn_every=0,
    ),
    # ── 15M — development and prototyping ────────────────────────────────
    "nano": ModelConfig(
        model_name="nano",
        vocab_size=32_000,
        d_model=288,
        n_layers=6,
        d_state=16,
        expand=2,
        beta_init=0.9,
        learn_beta=True,
        dropout=0.0,
        max_seq_len=1024,
        attn_every=6,
        attn_heads=6,
        attn_window=256,
    ),
    # ── 80M — the default production preset ──────────────────────────────
    "small": ModelConfig(
        model_name="small",
        vocab_size=32_000,
        d_model=640,
        n_layers=12,
        d_state=32,
        expand=2,
        beta_init=0.9,
        learn_beta=True,
        dropout=0.0,
        max_seq_len=2048,
        attn_every=6,
        attn_heads=10,
        attn_window=256,
    ),
    # ── 350M — for serious applications ──────────────────────────────────
    "medium": ModelConfig(
        model_name="medium",
        vocab_size=32_000,
        d_model=1152,
        n_layers=20,
        d_state=64,
        expand=2,
        beta_init=0.9,
        learn_beta=True,
        dropout=0.0,
        max_seq_len=4096,
        attn_every=5,
        attn_heads=16,
        attn_window=512,
    ),
    # ── 1.3B — flagship ─────────────────────────────────────────────────
    "large": ModelConfig(
        model_name="large",
        vocab_size=32_000,
        d_model=2048,
        n_layers=28,
        d_state=128,
        expand=2,
        beta_init=0.9,
        learn_beta=True,
        dropout=0.0,
        max_seq_len=8192,
        attn_every=7,
        attn_heads=16,
        attn_window=512,
    ),
}

# Back-compat aliases
PRESETS["64m"] = PRESETS["small"]
PRESETS["256m"] = PRESETS["medium"]
PRESETS["1b"] = PRESETS["large"]


def get_preset(name: str) -> ModelConfig:
    """Return a ModelConfig by preset name."""
    key = name.lower().strip()
    if key not in PRESETS:
        available = ", ".join(sorted(k for k in PRESETS.keys() if not k[-1].isdigit()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[key]


def list_presets() -> None:
    """Print a table of all available presets."""
    from neuratensor.core.model import NeuraTensorModel
    print(f"\n{'Preset':<10} {'d_model':>8} {'layers':>7} {'d_state':>8} {'attn':>5} {'max_seq':>8}  Est. params")
    print("-" * 72)
    for name in ("pico", "nano", "small", "medium", "large"):
        cfg = PRESETS[name]
        model = NeuraTensorModel(cfg)
        n = model.count_parameters()
        attn_str = f"{cfg.n_attn_layers}" if cfg.attn_every > 0 else "0"
        print(f"{name:<10} {cfg.d_model:>8} {cfg.n_layers:>7} {cfg.d_state:>8} {attn_str:>5} {cfg.max_seq_len:>8}  {n:>12,}")
    print()
