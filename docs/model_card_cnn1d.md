# Model Card — CNN1D

- **Architecture:** 3-block 1D CNN + global average pool
- **Input:** (batch, 1, 3000) single epoch
- **Parameters:** ~400K–600K depending on config
- **Portfolio results:** 2-fold, seed 42 on Sleep-EDF SC — **macro-F1 0.697 ± 0.002**, κ 0.664 ± 0.003
- **Per-class (mean across folds):** N1 F1 ~0.42; W ~0.90; N2 ~0.77; N3 ~0.74; REM ~0.65
- **Lift over RF:** +0.09 macro-F1 vs RF baseline (0.609)
- **Failure modes:** Treats epochs independently; N1 still weakest class; W/N1 boundary confusion persists
