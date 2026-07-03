STATUS: PARTIAL

_Updated: 2026-07-11 09:28 UTC_

## Results (mean ± std macro-F1 / κ)

Completed combos: 17 / 18 (DL folds: 2/3)

| Model | Scope | macro-F1 | κ | N1 F1 (mean) | combos |
|-------|-------|----------|---|--------------|--------|
| rf_baseline | 5-fold × 3 seeds | 0.609 ± 0.009 | 0.580 ± 0.012 | 0.279 | 15 |
| cnn1d | 2-fold, seed 42 | 0.697 ± 0.002 | 0.664 ± 0.003 | 0.421 | 2 |
| attnsleep | 1-fold, seed 42 | — | — | — | 0 |

## Sanity band (reference)

Published Sleep-EDF SC macro-F1 is often **0.75–0.82** (chance floor ≈ 0.20). Portfolio RF ~0.61; DL should beat RF clearly.

Run: `python scripts/run_portfolio_sweep.py`

## Failures / skipped

- **poc_attnsleep_sleep_edf_sc seed=42 fold=0**: MPS OOM on batch_size=64; CPU fallback ~18h/epoch. User killed after ~9h with epoch 0 incomplete.
