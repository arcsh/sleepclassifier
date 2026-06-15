"""PyTorch datasets and data modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from sleepstage.features.preprocessing import ProcessedRecord, load_processed_npz


@dataclass
class NightRecord:
    """One night of cached epochs."""

    path: Path
    subject_id: str
    night_id: str
    n_epochs: int


class EpochDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    """Dataset of single epochs from cached NPZ files."""

    def __init__(self, manifest: pd.DataFrame, indices: list[int]) -> None:
        """Build dataset from manifest rows.

        Args:
            manifest: Preprocessing manifest DataFrame.
            indices: Row indices to include.
        """
        self.records: list[NightRecord] = []
        self.epoch_offsets: list[tuple[int, int]] = []
        offset = 0
        for idx in indices:
            row = manifest.iloc[idx]
            n_epochs = int(row["n_epochs"])
            self.records.append(
                NightRecord(
                    path=Path(row["path"]),
                    subject_id=str(row["subject_id"]),
                    night_id=str(row["night_id"]),
                    n_epochs=n_epochs,
                )
            )
            self.epoch_offsets.append((offset, offset + n_epochs))
            offset += n_epochs
        self._total = offset
        self._cache: dict[int, ProcessedRecord] = {}

    def __len__(self) -> int:
        return self._total

    def _locate(self, index: int) -> tuple[int, int]:
        for rec_idx, (start, end) in enumerate(self.epoch_offsets):
            if start <= index < end:
                return rec_idx, index - start
        raise IndexError(f"Index {index} out of range")

    def _load_record(self, rec_idx: int) -> ProcessedRecord:
        if rec_idx not in self._cache:
            self._cache[rec_idx] = load_processed_npz(self.records[rec_idx].path)
        return self._cache[rec_idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        rec_idx, ep_idx = self._locate(index)
        data = self._load_record(rec_idx)
        x = torch.from_numpy(data.epochs[ep_idx].astype(np.float32))
        y = torch.tensor(int(data.labels[ep_idx]), dtype=torch.long)
        rec = self.records[rec_idx]
        return x, y, rec.subject_id, rec.night_id


class SequenceEpochDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    """Sliding-window dataset for sequence models (never crosses nights)."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        indices: list[int],
        sequence_length: int,
    ) -> None:
        """Build sequence windows from manifest rows.

        Args:
            manifest: Preprocessing manifest.
            indices: Row indices to include.
            sequence_length: Window length L.
        """
        self.sequence_length = sequence_length
        self.windows: list[tuple[int, int, str, str]] = []
        self._paths: list[Path] = []
        self._cache: dict[int, ProcessedRecord] = {}

        for local_idx, manifest_idx in enumerate(indices):
            row = manifest.iloc[manifest_idx]
            path = Path(row["path"])
            self._paths.append(path)
            n_epochs = int(row["n_epochs"])
            half = sequence_length // 2
            for center in range(half, n_epochs - (sequence_length - half - 1)):
                start = center - half
                self.windows.append(
                    (local_idx, start, str(row["subject_id"]), str(row["night_id"]))
                )

    def __len__(self) -> int:
        return len(self.windows)

    def _load(self, local_idx: int) -> ProcessedRecord:
        if local_idx not in self._cache:
            self._cache[local_idx] = load_processed_npz(self._paths[local_idx])
        return self._cache[local_idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        local_idx, start, subject_id, night_id = self.windows[index]
        data = self._load(local_idx)
        end = start + self.sequence_length
        window = data.epochs[start:end]
        center = self.sequence_length // 2
        label = int(data.labels[start + center])
        x = torch.from_numpy(window.astype(np.float32))
        y = torch.tensor(label, dtype=torch.long)
        return x, y, subject_id, night_id


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, str, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Collate function preserving metadata.

    Args:
        batch: List of dataset items.

    Returns:
        Batched tensors and metadata lists.
    """
    xs, ys, subjects, nights = zip(*batch, strict=True)
    return torch.stack(xs), torch.stack(ys), list(subjects), list(nights)


def make_dataloader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor, str, str]],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, list[str], list[str]]]:
    """Create a DataLoader with standard collate.

    Args:
        dataset: Source dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Worker processes.

    Returns:
        Configured DataLoader.
    """
    pin = torch.cuda.is_available()
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
