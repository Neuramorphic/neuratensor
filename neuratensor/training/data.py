"""Data utilities for NeuraTensor training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class TextDataset(Dataset):
    """Language-model dataset from .jsonl or .txt files.

    .jsonl — expects {"text": "..."} per line.
    .txt   — each non-empty line is a document.

    If no tokenizer is provided, falls back to UTF-8 byte encoding (vocab 256).
    """

    def __init__(
        self,
        path: str,
        max_seq_len: int = 512,
        tokenizer: Optional[Callable[[str], List[int]]] = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self._encode = tokenizer or _byte_encode

        raw_texts = self._load(path)
        all_ids: List[int] = []
        for text in raw_texts:
            all_ids.extend(self._encode(text))

        stride = max_seq_len + 1
        n_chunks = max(1, len(all_ids) // stride)
        trimmed = all_ids[: n_chunks * stride]
        self.chunks = torch.tensor(trimmed, dtype=torch.long).view(n_chunks, stride)

    def __len__(self) -> int:
        return self.chunks.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        seq = self.chunks[idx]
        return {"input_ids": seq[:-1], "labels": seq[1:]}

    @staticmethod
    def _load(path: str) -> List[str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        raw = p.read_text(encoding="utf-8")
        if p.suffix == ".jsonl":
            texts: List[str] = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(obj["text"])
            return texts
        return [line for line in raw.splitlines() if line.strip()]


def _byte_encode(text: str) -> List[int]:
    return list(text.encode("utf-8"))


class InfiniteDataLoader:
    """Wraps a DataLoader to cycle indefinitely for step-based training."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        num_workers: int = 0,
        distributed: bool = False,
        seed: int = 42,
    ) -> None:
        self._seed = seed
        self._epoch = 0

        sampler: Optional[DistributedSampler] = None
        shuffle = True
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
            shuffle = False

        self._sampler = sampler
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        self._iter: Optional[Iterator] = None

    def __iter__(self) -> "InfiniteDataLoader":
        return self

    def __next__(self) -> Dict[str, Tensor]:
        if self._iter is None:
            self._new_epoch()
        try:
            return next(self._iter)  # type: ignore[arg-type]
        except StopIteration:
            self._new_epoch()
            return next(self._iter)  # type: ignore[arg-type]

    def _new_epoch(self) -> None:
        self._epoch += 1
        if self._sampler is not None:
            self._sampler.set_epoch(self._epoch)
        self._iter = iter(self._loader)
