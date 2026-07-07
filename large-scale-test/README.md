# Large-Scale Test Archive

This folder holds the **original research-reproduction scope** (60-combo full sweep: 4 models × 5 folds × 3 seeds) and related tooling. It is **not** the active project goal — the root README describes the portfolio-focused pipeline.

## Contents

| Path | Purpose |
|------|---------|
| `scripts/run_full_sweep.py` | 60-combo fault-tolerant sweep orchestrator |
| `scripts/run_unattended.py` | Download → preprocess → sanity → background sweep |
| `scripts/watchdog.py` | Long-run supervisor with caffeinate + circuit breaker |
| `tests/test_watchdog.py` | Watchdog unit tests |

Optional local artifacts (gitignored): `mac-logs/`, `logs/`, `experiments/`, `outputs/`.

## Why archived

On an M1 MacBook Air, the full sweep takes **~5–7 days** of continuous compute. The active goal favors a shorter baseline → deep learning story over full paper reproduction.

## Running (if you really want the full sweep)

From repo root, on AC power, with `caffeinate`:

```bash
caffeinate -dims python large-scale-test/scripts/run_unattended.py
# or resume sweep only:
caffeinate -dims python large-scale-test/scripts/run_full_sweep.py
```

**Known issues on M1:**

- Use `model.classifier=random_forest` for RF (LightGBM segfaults on full-fold RAM concat).
- PyTorch `accelerator=auto` → MPS; may hang — CPU fallback is implemented in `run_full_sweep.py`.
- RF baseline results already complete in `experiments/rf_baseline_sleep_edf_sc/` (repo root, not moved).

## RF results preserved

The completed Random Forest 5-fold × 3-seed run remains at:

```
experiments/rf_baseline_sleep_edf_sc/
```

Macro-F1 **0.609 ± 0.009**, κ **0.580 ± 0.012** — reused by the portfolio pipeline.
