# Portfolio Run — Final Status

**Date:** 2026-07-11  
**Outcome:** PARTIAL PASS — RF + CNN1D complete; AttnSleep skipped  
**Wall clock:** ~14 h (started 2026-07-10 19:45 UTC)

---

## What ran

| Step | Status | Wall time | Notes |
|------|--------|-----------|-------|
| Phase 1 — code + gates | **PASS** | ~1 h | `run_portfolio_sweep.py`, tests, lint |
| RF baseline (pre-done) | **PASS** | — | 15/15 combos (5-fold × 3 seeds) |
| CNN1D fold 0 | **PASS** | ~2.5 h | MPS, seed 42 |
| CNN1D fold 1 | **PASS** | ~2.3 h | MPS, seed 42 |
| AttnSleep fold 0 | **SKIPPED** | ~9 h (killed) | MPS OOM → CPU; epoch 0 incomplete at step 899/1804 |
| Phase 3 — polish | **PASS** | ~30 min | docs, figures, this file |

---

## Final metrics

| Model | Scope | Macro-F1 | κ | N1 F1 | Combos |
|-------|-------|----------|---|-------|--------|
| RF baseline | 5-fold × 3 seeds | 0.609 ± 0.009 | 0.580 ± 0.012 | 0.279 | 15/15 |
| CNN1D | 2-fold, seed 42 | **0.697 ± 0.002** | **0.664 ± 0.003** | 0.421 | 2/2 |
| AttnSleep | 1-fold planned | — | — | — | 0/1 (skipped) |

**Key result:** CNN1D beats RF by **+0.09 macro-F1** — clear baseline → DL narrative.

Artifacts: `experiments/rf_baseline_sleep_edf_sc/`, `experiments/poc_cnn1d_sleep_edf_sc/`

---

## Figures produced

`python scripts/make_figures.py` — **10 figures** in `reports/figures/`:

- `01_example_epochs.png`
- `02_psd_per_class.png`
- `03_class_distribution.png`
- `04_hypnogram_overlay.png`
- `05_confusion_poc_cnn1d_sleep_edf_sc.png`
- `05_confusion_rf_baseline_sleep_edf_sc.png`
- `06_training_curves.png`
- `07_tsne_embeddings.png`
- `08_per_subject_f1.png`
- `10_model_comparison.png` (README hero)

Skipped: `09_attention_weights.png` (no AttnSleep checkpoint)

---

## Test / lint status

```
ruff check .     — PASS
black --check .  — PASS
mypy src/        — PASS
pytest tests/    — 30 passed
```

---

## Definition of done checklist

| Criterion | Status |
|-----------|--------|
| Quality gates green | **PASS** |
| `run_portfolio_sweep.py` exists and ran | **PASS** (partial) |
| RF 15/15 | **PASS** |
| CNN1D ≥1 fold with metrics.json | **PASS** (2 folds) |
| AttnSleep fold 0 | **SKIP** — documented below |
| ≥5 figures incl. `10_model_comparison.png` | **PASS** |
| RESULTS.md + README + writeup updated | **PASS** |
| No root README primary path to full sweep | **PASS** |

---

## Skipped: AttnSleep

**Root cause:** MPS OOM at `batch_size=64` (7.6 GiB allocated, tried +937 MiB).

**Fallback:** CPU training started automatically. After ~9 h, only step 899/1804 of epoch 0 completed (~35 s/step → ~18 h/epoch estimated).

**Decision:** User killed process. Portfolio ships with RF + CNN1D.

**Retry path (optional):**
```bash
python scripts/train.py model=attnsleep training=portfolio fold_only=0 seed=42 \
  force_seed_subdir=true training.n_seeds=1 \
  experiment_name=poc_attnsleep_sleep_edf_sc \
  output_dir=experiments/poc_attnsleep_sleep_edf_sc \
  training.batch_size=16 training.accelerator=auto
```

---

## Updated docs

| File | Status |
|------|--------|
| `RESULTS.md` | Regenerated with real numbers |
| `README.md` | Results table + honest AttnSleep skip |
| `reports/writeup.md` | Full writeup with N1 failure analysis |
| `docs/model_card_rf.md` | Already filled |
| `docs/model_card_cnn1d.md` | Filled with portfolio metrics |
| `docs/model_card_attnsleep.md` | Skip note + retry path |
| `docs/model_card_cnn_bilstm.md` | Not evaluated note |
| `experiments/portfolio_run_status.json` | AttnSleep marked skipped |

---

## Bottom line

Portfolio is **hire-ready** with RF (rigorous CV) + CNN1D (clear DL lift). AttnSleep is optional stretch — not required for the baseline → DL story. Total compute exceeded the 8 h budget due to AttnSleep CPU trap; CNN1D portion alone was ~5 h as planned.
