# Model Card — CNN-BiLSTM

- **Architecture:** Epoch CNN encoder + 2-layer BiLSTM + residual skip (DeepSleepNet-style)
- **Input:** (batch, L, 1, 3000) with center-epoch label
- **Parameters:** ~1–2M
- **Portfolio status:** **Not evaluated in portfolio run** — implemented for completeness; skipped due to compute/impact trade-off.
- **Failure modes:** Slow training on CPU; border nights lose edge windows
