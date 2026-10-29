"""Training loop for NeuraTensor models."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .data import InfiniteDataLoader


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    beta_lr_multiplier: float = 3.0
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    batch_size: int = 4
    grad_accum_steps: int = 1
    max_seq_len: int = 512
    eval_every: int = 100
    save_every: int = 500
    log_every: int = 10
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    output_dir: str = "checkpoints"
    seed: int = 42
    spike_regularization: float = 0.0  # L1 penalty on spike rate (0.0 = disabled, 0.01-0.1 typical)


class Trainer:
    """Simple, correct training loop for NeuraTensor models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_data: Dataset,
        eval_data: Optional[Dataset] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.eval_data = eval_data

        torch.manual_seed(config.seed)

        self.device = self._detect_device()
        self.model.to(self.device)

        self.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if self.distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = f"cuda:{local_rank}"
            self.model.to(self.device)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank]
            )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        use_amp = config.mixed_precision and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.autocast_ctx = lambda: torch.autocast(
            device_type=self.device.split(":")[0],
            dtype=torch.bfloat16 if use_amp else torch.float32,
            enabled=use_amp,
        )

        self.train_loader = InfiniteDataLoader(
            train_data,
            batch_size=config.batch_size,
            distributed=self.distributed,
            seed=config.seed,
        )
        self.train_iter = iter(self.train_loader)

        self.global_step = 0
        self._log_header_printed = False

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self) -> Dict[str, Any]:
        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self.model.train()

        accum_loss = 0.0
        accum_spike = 0.0
        accum_count = 0
        t_start = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)

        for micro_step in range(cfg.max_steps * cfg.grad_accum_steps):
            batch = next(self.train_iter)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with self.autocast_ctx():
                logits, _states, metrics = self._forward(input_ids)
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

                # Spike regularization: penalize high spike rates
                spike_loss = 0.0
                if cfg.spike_regularization > 0.0:
                    spike_rate = metrics.get("spike_rate", 0.0)
                    spike_loss = cfg.spike_regularization * spike_rate

                loss = (ce_loss + spike_loss) / cfg.grad_accum_steps

            self.scaler.scale(loss).backward()

            accum_loss += loss.item() * cfg.grad_accum_steps
            accum_spike += metrics.get("spike_rate", 0.0)
            accum_count += 1

            if (micro_step + 1) % cfg.grad_accum_steps != 0:
                continue

            # Optimizer step
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self._trainable_params(), cfg.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.global_step += 1

            # Logging
            if self.global_step % cfg.log_every == 0:
                avg_loss = accum_loss / accum_count
                avg_spike = accum_spike / accum_count
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t_start
                tok_per_s = (
                    accum_count * cfg.batch_size * cfg.max_seq_len / elapsed
                )
                self._log(
                    step=self.global_step,
                    loss=avg_loss,
                    spike_rate=avg_spike,
                    lr=lr,
                    grad_norm=grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
                    tok_s=tok_per_s,
                )
                accum_loss = 0.0
                accum_spike = 0.0
                accum_count = 0
                t_start = time.perf_counter()

            # Eval
            if cfg.eval_every > 0 and self.global_step % cfg.eval_every == 0:
                if self.eval_data is not None:
                    eval_metrics = self.evaluate()
                    self._log_eval(self.global_step, eval_metrics)
                    self.model.train()

            # Save
            if cfg.save_every > 0 and self.global_step % cfg.save_every == 0:
                self.save_checkpoint(
                    str(Path(cfg.output_dir) / f"step_{self.global_step}.pt")
                )

            if self.global_step >= cfg.max_steps:
                break

        final_path = str(Path(cfg.output_dir) / "final.pt")
        self.save_checkpoint(final_path)
        return {"final_checkpoint": final_path, "steps": self.global_step}

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> Dict[str, float]:
        self.model.eval()
        if self.eval_data is None:
            return {}

        loader = InfiniteDataLoader(
            self.eval_data, batch_size=self.config.batch_size, seed=0
        )
        total_loss = 0.0
        total_tokens = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            with self.autocast_ctx():
                logits, _, _ = self._forward(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="sum",
                )
            total_loss += loss.item()
            total_tokens += labels.numel()

        avg_loss = total_loss / max(total_tokens, 1)
        return {"eval_loss": avg_loss, "eval_ppl": math.exp(min(avg_loss, 20.0))}

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: str) -> None:
        raw_model = self.model.module if self.distributed else self.model
        torch.save(
            {
                "model": raw_model.state_dict(),
                "config": getattr(raw_model, "config", None),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.global_step,
                "train_config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw_model = self.model.module if self.distributed else self.model
        raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["step"]

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _forward(self, input_ids: Tensor):
        raw = self.model.module if self.distributed else self.model
        return raw(input_ids)

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        beta_params = []
        other_params = []
        no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "beta_logit" in name:
                beta_params.append(param)
            elif param.ndim < 2 or "norm" in name or "bias" in name:
                no_decay.append(param)
            else:
                other_params.append(param)

        groups = [
            {"params": other_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {
                "params": beta_params,
                "lr": cfg.learning_rate * cfg.beta_lr_multiplier,
                "weight_decay": 0.0,
            },
        ]
        groups = [g for g in groups if g["params"]]
        return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.95))

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        cfg = self.config

        def _lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return step / max(cfg.warmup_steps, 1)
            progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _log(
        self,
        step: int,
        loss: float,
        spike_rate: float,
        lr: float,
        grad_norm: float,
        tok_s: float,
    ) -> None:
        if not self._log_header_printed:
            print(
                f"{'step':>7} | {'loss':>8} | {'spike':>6} | "
                f"{'lr':>10} | {'gnorm':>7} | {'tok/s':>9}"
            )
            print("-" * 62)
            self._log_header_printed = True
        print(
            f"{step:>7d} | {loss:>8.4f} | {spike_rate:>6.3f} | "
            f"{lr:>10.2e} | {grad_norm:>7.2f} | {tok_s:>9,.0f}"
        )

    @staticmethod
    def _log_eval(step: int, metrics: Dict[str, float]) -> None:
        parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  [eval @ step {step}] {parts}")
