# Sleep Stage Classification from EEG

Automated sleep staging on 30-second EEG epochs from the [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) (Sleep Cassette subset). Classifies each epoch into **W, N1, N2, N3, REM** with subject-independent cross-validation on real PhysioNet data.

> **Portfolio project** — credible baseline → deep learning on laptop-scale compute. For the full research reproduction spec (60 combos, 3 seeds), see [`large-scale-test/`](large-scale-test/README.md).

![Model comparison](reports/figures/10_model_comparison.png)

## Results

| Model | Eval scope | Macro-F1 | Cohen's κ |
|-------|------------|----------|-----------|
| RF baseline | 5-fold × 3 seeds | **0.609 ± 0.009** | 0.580 ± 0.012 |
| CNN1D | 2-fold, seed 42 | **0.697 ± 0.002** | 0.664 ± 0.003 |
| AttnSleep | _not evaluated_ | — | — |

CNN1D beats RF by **+0.09 macro-F1** on the same Sleep-EDF SC subset. AttnSleep was skipped after MPS OOM (CPU fallback estimated 18+ h/epoch); see [`logs/FINAL_STATUS.md`](logs/FINAL_STATUS.md).

Live table: [`RESULTS.md`](RESULTS.md) · Blog: [arcsh.github.io/blog/sleeptracker](https://arcsh.github.io/blog/sleeptracker) · Writeup: [`reports/writeup.md`](reports/writeup.md)

## Non-goals

This is a portfolio/research codebase, not a clinical sleep-scoring device. It does not diagnose sleep disorders, does not target FDA-grade validation, and does not claim SOTA benchmark records. CNN-BiLSTM and AttnSleep are implemented but not fully evaluated in the portfolio run.

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Portfolio pipeline (CNN1D ~5 h on M1; RF results cached)
caffeinate -dims python scripts/run_portfolio_sweep.py
python scripts/make_figures.py

# Or step-by-step:
python scripts/download_data.py --subset sc   # idempotent
python scripts/preprocess.py
python scripts/train.py model=cnn1d training=portfolio fold_only=0 seed=42 \
  force_seed_subdir=true training.n_seeds=1 \
  experiment_name=poc_cnn1d_sleep_edf_sc output_dir=experiments/poc_cnn1d_sleep_edf_sc
python scripts/make_figures.py

# CI smoke (synthetic data, <2 min)
python scripts/demo_pipeline.py

# Lint + test
ruff check . && black --check . && mypy src/ && python -m pytest tests/
```

**Docs:** [`reports/writeup.md`](reports/writeup.md) · [`reports/blog.md`](reports/blog.md) · [`docs/`](docs/)

## Repository structure

```
configs/          Hydra configs (data, model, training, portfolio)
src/sleepstage/   Installable package
scripts/          CLI entry points (+ run_portfolio_sweep.py)
tests/            Unit + integration tests
reports/figures/  Generated plots
docs/             Data and model cards
experiments/      Per-run outputs (gitignored)
data/             Raw/processed data (gitignored)
large-scale-test/ Archived full research sweep (60 combos)
```

## CI

[![CI](https://github.com/arcsh/sleeptracker/actions/workflows/ci.yml/badge.svg)](https://github.com/arcsh/sleeptracker/actions/workflows/ci.yml)

Runs `ruff`, `black`, `mypy`, and `pytest` on every push.

## Citations

- Kemp et al. (2000). Analysis of a sleep-dependent neuronal feedback loop. *IEEE TBME*.
- Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*.
- Supratak et al. (2017). DeepSleepNet. *IEEE TPAMI*.
- Eldele et al. (2021). AttnSleep. *IEEE TBME*.

## License

MIT. Sleep-EDF data under [PhysioNet Open Data License](https://physionet.org/content/sleep-edfx/view-license/1.0.0/).
