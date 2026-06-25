"""PyTorch Lightning training module."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from sleepstage.constants import STAGE_NAMES
from sleepstage.evaluation.metrics import compute_metrics
from sleepstage.models.base import SleepStageModel
from sleepstage.training.losses import (
    build_loss,
    compute_class_weights,
    compute_class_weights_from_counts,
)


class SleepStageLightningModule(LightningModule):
    """Generic Lightning wrapper for any SleepStageModel."""

    def __init__(
        self,
        model: SleepStageModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_name: str = "weighted_ce",
        focal_gamma: float = 2.0,
        scheduler_name: str = "cosine",
        max_epochs: int = 50,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])
        self.model = model
        self.loss_fn = build_loss(loss_name, class_weights, focal_gamma)
        self.scheduler_name = scheduler_name
        self.max_epochs = max_epochs
        self._val_preds: list[int] = []
        self._val_targets: list[int] = []
        self._train_loss_epoch: float | None = None
        self.epoch_history: list[dict[str, float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_end(self) -> None:
        loss = self.trainer.callback_metrics.get("train_loss_epoch")
        if loss is not None:
            self._train_loss_epoch = float(loss)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, list[str], list[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, list[str], list[str]], batch_idx: int
    ) -> None:
        x, y, _, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=-1)
        self._val_preds.extend(preds.cpu().tolist())
        self._val_targets.extend(y.cpu().tolist())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        if not self._val_targets:
            return
        metrics = compute_metrics(self._val_targets, self._val_preds)
        val_loss = self.trainer.callback_metrics.get("val_loss")
        self.log("val_macro_f1", metrics["macro_f1"], prog_bar=True)
        self.log("val_accuracy", metrics["accuracy"])
        self.log("val_kappa", metrics["kappa"])
        self.epoch_history.append(
            {
                "epoch": float(self.current_epoch),
                "train_loss": float(self._train_loss_epoch or 0.0),
                "val_loss": float(val_loss) if val_loss is not None else 0.0,
                "val_macro_f1": float(metrics["macro_f1"]),
            }
        )
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self) -> Any:
        lr = float(self.hparams["lr"])
        weight_decay = float(self.hparams["weight_decay"])
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        if self.scheduler_name == "plateau":
            scheduler: ReduceLROnPlateau | CosineAnnealingLR = ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_macro_f1"},
            }
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def class_weights_from_manifest(
        manifest: pd.DataFrame,
        train_indices: list[int],
        n_classes: int = 5,
    ) -> torch.Tensor:
        """Compute class weights from manifest count_* columns (no loader scan)."""
        counts = torch.zeros(n_classes)
        count_cols = [f"count_{name}" for name in STAGE_NAMES]
        for idx in train_indices:
            row = manifest.iloc[idx]
            for i, col in enumerate(count_cols):
                counts[i] += float(row[col])
        return compute_class_weights_from_counts(counts, n_classes)

    @staticmethod
    def class_weights_from_loader(
        loader: torch.utils.data.DataLoader,
        n_classes: int = 5,
    ) -> torch.Tensor:
        """Compute class weights from a training DataLoader."""
        labels: list[int] = []
        for _, y, _, _ in loader:
            labels.extend(y.tolist())
        return compute_class_weights(torch.tensor(labels), n_classes)
