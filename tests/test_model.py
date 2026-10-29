"""
Core test suite for NeuraTensor v4.

Tests:
  • LIF layer dynamics (chunked scan, spikes, membrane, surrogate gradient, multi-threshold)
  • AdaptiveLIF (learnable threshold)
  • Selective SSM (input-dependent B, C, Δ; state update; streaming continuity)
  • FusedSpikeBlock (forward, state carry, sparse inference)
  • SlidingWindowAttention (RoPE, windowed mask)
  • NeuraTensorModel (all presets, generate, save/load, hybrid attention)
  • InferenceEngine (run, benchmark, checkpoint)
  • Training (LoRA apply/merge, Trainer smoke test)
"""

from __future__ import annotations

import tempfile
import pytest
import torch

from neuratensor import (
    NeuraTensorModel, ModelConfig,
    LIFLayer, AdaptiveLIFLayer,
    SelectiveSSMLayer,
    FusedSpikeBlock, BlockState,
    InferenceEngine, get_preset,
    SlidingWindowAttention,
)


# ──────────────────────────────────────────────────────────────────────── #
#  Fixtures                                                                 #
# ──────────────────────────────────────────────────────────────────────── #

@pytest.fixture(scope="module")
def pico_config():
    return ModelConfig(
        vocab_size=1000, d_model=64, n_layers=4, d_state=8,
        max_seq_len=128, attn_every=4, attn_heads=2, attn_window=32,
    )

@pytest.fixture(scope="module")
def pico_model(pico_config):
    return NeuraTensorModel(pico_config).eval()

@pytest.fixture(scope="module")
def pico_engine(pico_config):
    model = NeuraTensorModel(pico_config)
    return InferenceEngine(model, device="cpu", dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────── #
#  LIF                                                                      #
# ──────────────────────────────────────────────────────────────────────── #

class TestLIF:
    def test_output_shape(self):
        lif = LIFLayer(size=16)
        x = torch.randn(2, 10, 16)
        spikes, membrane = lif(x)
        assert spikes.shape == (2, 10, 16)
        assert membrane.shape == (2, 16)

    def test_spikes_binary(self):
        lif = LIFLayer(size=8)
        x = torch.ones(1, 20, 8) * 2.0
        spikes, _ = lif(x)
        unique = spikes.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0}), "Spikes must be 0 or 1"

    def test_beta_clamped(self):
        lif = LIFLayer(size=4, beta_init=0.95)
        assert (lif.beta >= 0).all() and (lif.beta <= 1).all()

    def test_membrane_carry(self):
        lif = LIFLayer(size=4, learn_beta=False)
        x = torch.randn(1, 5, 4)
        _, mem1 = lif(x)
        _, mem2 = lif(x, membrane=mem1)
        assert mem2.shape == (1, 4)

    def test_surrogate_grad_flows(self):
        lif = LIFLayer(size=8, learn_beta=True)
        x = torch.randn(1, 5, 8, requires_grad=True)
        spikes, _ = lif(x)
        loss = spikes.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_single_step_path(self):
        lif = LIFLayer(size=8)
        x = torch.randn(2, 1, 8)
        spikes, mem = lif(x)
        assert spikes.shape == (2, 1, 8)

    def test_multi_threshold(self):
        lif = LIFLayer(size=12, multi_threshold=True)
        x = torch.randn(1, 10, 12)
        spikes, _ = lif(x)
        assert spikes.shape == (1, 10, 12)


# ──────────────────────────────────────────────────────────────────────── #
#  AdaptiveLIF                                                              #
# ──────────────────────────────────────────────────────────────────────── #

class TestAdaptiveLIF:
    def test_output_shape(self):
        alif = AdaptiveLIFLayer(size=16)
        x = torch.randn(2, 10, 16)
        spikes, membrane = alif(x)
        assert spikes.shape == (2, 10, 16)
        assert membrane.shape == (2, 16)

    def test_threshold_positive(self):
        alif = AdaptiveLIFLayer(size=8)
        assert (alif.threshold > 0).all()

    def test_threshold_grad_flows(self):
        alif = AdaptiveLIFLayer(size=8)
        x = torch.randn(1, 5, 8, requires_grad=True)
        spikes, _ = alif(x)
        loss = spikes.sum()
        loss.backward()
        assert alif.threshold_raw.grad is not None


# ──────────────────────────────────────────────────────────────────────── #
#  Selective SSM                                                            #
# ──────────────────────────────────────────────────────────────────────── #

class TestSelectiveSSM:
    def test_output_shape(self):
        ssm = SelectiveSSMLayer(d_model=16, d_state=4)
        x = torch.randn(2, 10, 16)
        y, state = ssm(x)
        assert y.shape == (2, 10, 16)
        assert state.shape == (2, 16, 4)

    def test_state_streaming(self):
        ssm = SelectiveSSMLayer(d_model=8, d_state=4)
        x_full = torch.randn(1, 6, 8)
        y_full, _ = ssm(x_full)
        y1, state = ssm(x_full[:, :3, :])
        y2, _ = ssm(x_full[:, 3:, :], state=state)
        y_concat = torch.cat([y1, y2], dim=1)
        assert torch.allclose(y_full, y_concat, atol=1e-4)

    def test_grad_flows(self):
        ssm = SelectiveSSMLayer(d_model=8, d_state=4)
        x = torch.randn(1, 5, 8, requires_grad=True)
        y, _ = ssm(x)
        y.sum().backward()
        assert x.grad is not None

    def test_hippo_init_non_uniform(self):
        ssm = SelectiveSSMLayer(d_model=4, d_state=8)
        log_A = ssm.log_A.data
        assert not torch.allclose(log_A[:, 0], log_A[:, 1])

    def test_single_step(self):
        ssm = SelectiveSSMLayer(d_model=8, d_state=4)
        x = torch.randn(2, 1, 8)
        y, state = ssm(x)
        assert y.shape == (2, 1, 8)


# ──────────────────────────────────────────────────────────────────────── #
#  FusedSpikeBlock                                                          #
# ──────────────────────────────────────────────────────────────────────── #

class TestFusedBlock:
    def test_output_shape(self):
        block = FusedSpikeBlock(d_model=16, d_state=4, expand=2)
        x = torch.randn(2, 8, 16)
        out, state, metrics = block(x)
        assert out.shape == (2, 8, 16)
        assert isinstance(state, BlockState)
        assert 0.0 <= metrics["spike_rate"] <= 1.0

    def test_residual_preserves_shape(self):
        block = FusedSpikeBlock(d_model=32, d_state=8)
        x = torch.randn(3, 4, 32)
        out, _, _ = block(x)
        assert out.shape == x.shape

    def test_state_carry(self):
        block = FusedSpikeBlock(d_model=16, d_state=4)
        x = torch.randn(1, 5, 16)
        out1, state1, _ = block(x)
        out2, state2, _ = block(x, state=state1)
        assert state2.membrane is not None
        assert state2.ssm_state is not None


# ──────────────────────────────────────────────────────────────────────── #
#  SlidingWindowAttention                                                   #
# ──────────────────────────────────────────────────────────────────────── #

class TestAttention:
    def test_output_shape(self):
        attn = SlidingWindowAttention(d_model=32, n_heads=4, window=16)
        x = torch.randn(2, 20, 32)
        out = attn(x)
        assert out.shape == (2, 20, 32)

    def test_residual(self):
        attn = SlidingWindowAttention(d_model=32, n_heads=4, window=16)
        x = torch.randn(1, 8, 32)
        out = attn(x)
        assert not torch.allclose(out, x), "Output should differ from input"

    def test_long_sequence_windowed(self):
        attn = SlidingWindowAttention(d_model=32, n_heads=4, window=8)
        x = torch.randn(1, 32, 32)
        out = attn(x)
        assert out.shape == (1, 32, 32)


# ──────────────────────────────────────────────────────────────────────── #
#  NeuraTensorModel                                                         #
# ──────────────────────────────────────────────────────────────────────── #

class TestModel:
    def test_forward_shape(self, pico_model, pico_config):
        ids = torch.randint(0, pico_config.vocab_size, (2, 8))
        logits, states, metrics = pico_model(ids)
        assert logits.shape == (2, 8, pico_config.vocab_size)
        assert len(states) == pico_config.n_layers
        assert "spike_rate" in metrics

    def test_parameter_count_positive(self, pico_model):
        assert pico_model.count_parameters() > 0

    def test_weight_tying(self, pico_model):
        assert pico_model.lm_head.weight.data_ptr() == pico_model.embedding.weight.data_ptr()

    def test_generate(self, pico_model, pico_config):
        ids = torch.randint(0, pico_config.vocab_size, (1, 4))
        generated, _ = pico_model.generate(ids, max_new_tokens=5, top_k=10)
        assert generated.shape == (1, 9)

    def test_generate_with_stop_token(self, pico_config):
        model = NeuraTensorModel(pico_config).eval()
        ids = torch.randint(0, pico_config.vocab_size, (1, 4))
        generated, _ = model.generate(ids, max_new_tokens=100, top_k=10, stop_tokens=[0])
        assert generated.shape[1] <= 104

    def test_save_load(self, pico_config):
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            engine = InferenceEngine(
                NeuraTensorModel(pico_config),
                device="cpu",
                dtype=torch.float32,
            )
            engine.save(f.name)
            loaded = InferenceEngine.from_checkpoint(f.name, device="cpu")
            assert loaded.model.count_parameters() == engine.model.count_parameters()

    @pytest.mark.parametrize("preset", ["pico", "nano"])
    def test_presets_run(self, preset):
        cfg = get_preset(preset)
        model = NeuraTensorModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, _, _ = model(ids)
        assert logits.shape[0] == 1
        assert logits.shape[-1] == cfg.vocab_size

    def test_model_with_attention(self):
        cfg = ModelConfig(
            vocab_size=500, d_model=32, n_layers=4, d_state=4,
            attn_every=2, attn_heads=2, attn_window=16, max_seq_len=64,
        )
        model = NeuraTensorModel(cfg).eval()
        ids = torch.randint(0, 500, (1, 16))
        logits, states, metrics = model(ids)
        assert logits.shape == (1, 16, 500)
        assert len(model.attn_layers) == 2


# ──────────────────────────────────────────────────────────────────────── #
#  InferenceEngine                                                          #
# ──────────────────────────────────────────────────────────────────────── #

class TestEngine:
    def test_run_returns_result(self, pico_engine, pico_config):
        ids = torch.randint(0, pico_config.vocab_size, (1, 8))
        result = pico_engine.run(ids)
        assert result.latency_ms > 0
        assert result.logits is not None
        assert "spike_rate" in result.metrics

    def test_benchmark_keys(self, pico_engine):
        stats = pico_engine.benchmark(batch_size=1, seq_length=8, n_iter=5, n_warmup=2)
        for key in ("mean_ms", "std_ms", "p99_ms", "throughput_seq_per_s",
                    "spike_rate", "sparsity_pct"):
            assert key in stats

    def test_reset_state(self, pico_engine, pico_config):
        ids = torch.randint(0, pico_config.vocab_size, (1, 4))
        pico_engine.run(ids)
        assert any(s is not None for s in pico_engine._states)
        pico_engine.reset_state()
        assert all(s is None for s in pico_engine._states)


# ──────────────────────────────────────────────────────────────────────── #
#  Training                                                                 #
# ──────────────────────────────────────────────────────────────────────── #

class TestLoRA:
    def test_apply_and_merge(self):
        from neuratensor.training import LoRAConfig, apply_lora, merge_lora

        cfg = ModelConfig(vocab_size=100, d_model=32, n_layers=2, d_state=4, max_seq_len=32)
        model = NeuraTensorModel(cfg)

        n_total_before = model.count_parameters()
        apply_lora(model, LoRAConfig(r=4, target_modules=["in_proj", "out_proj"]))

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_trainable < n_total_before

        ids = torch.randint(0, 100, (1, 8))
        logits, _, _ = model(ids)
        assert logits.shape == (1, 8, 100)

        merge_lora(model)

        logits2, _, _ = model(ids)
        assert logits2.shape == (1, 8, 100)
