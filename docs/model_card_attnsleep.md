# Model Card — AttnSleep

- **Architecture:** Multi-resolution CNN branches + multi-head self-attention
- **Input:** (batch, L, 1, 3000) with sequence length L=20
- **Parameters:** ~500K–1M
- **Portfolio status:** **Not evaluated in portfolio run** — MPS OOM at batch_size=64; CPU fallback estimated ~18 h/epoch on M1 MacBook Air. Training killed manually after ~9 h with epoch 0 incomplete.
- **Retry path:** `training.batch_size=16 training.accelerator=auto` may fit MPS; see `logs/FINAL_STATUS.md`
- **Failure modes:** With few epochs of training, can collapse to majority class; attention maps need sufficient training to be interpretable
