"""Tests for model forward passes."""

from __future__ import annotations

import pytest
import torch

import sleepstage.models  # noqa: F401 — register models
from sleepstage.models.attnsleep import AttnSleep
from sleepstage.models.base import MODEL_REGISTRY
from sleepstage.models.cnn1d import CNN1D


@pytest.mark.parametrize("name", ["cnn1d", "cnn_bilstm", "attnsleep"])
def test_model_registry_forward(name: str) -> None:
    cls = MODEL_REGISTRY[name]
    model = cls()
    if getattr(model, "is_sequence_model", False):
        x = torch.randn(4, 20, 1, 3000)
    else:
        x = torch.randn(4, 1, 3000)
    logits = model(x)
    assert logits.shape == (4, 5)
    loss = logits.sum()
    loss.backward()
    assert not any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)


def test_cnn1d_sequence_input_uses_center() -> None:
    model = CNN1D()
    x = torch.randn(2, 10, 1, 3000)
    out = model(x)
    assert out.shape == (2, 5)


def test_attnsleep_attention_weights() -> None:
    model = AttnSleep(sequence_length=10)
    x = torch.randn(2, 10, 1, 3000)
    model(x)
    weights = model.get_attention_weights()
    assert weights is not None
