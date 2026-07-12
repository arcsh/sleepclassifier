# Sleep Stage Classification from EEG

[![CI](https://github.com/arcsh/sleepclassifier/actions/workflows/ci.yml/badge.svg)](https://github.com/arcsh/sleepclassifier/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end deep learning for **automated sleep staging** on real overnight EEG.

Trains on the [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) Sleep Cassette subset (~153 recordings, ~195k epochs), labels each 30-second window as **W / N1 / N2 / N3 / REM**, and evaluates with **subject-independent** cross-validation — no leakage across people.

![Model comparison](reports/figures/10_model_comparison.png)

## Highlights

- **Baseline → DL lift:** Random Forest (handcrafted spectra) → CNN1D (**+0.09** macro-F1)
- **Honest eval:** subject-wise GroupKFold, macro-F1 + Cohen's κ, N1 failure analysis
- **Full pipeline:** download → preprocess → train → figures (Hydra + PyTorch Lightning)
- **Reproducible:** installable package, model/data cards, CI (`ruff` / `black` / `mypy` / `pytest`)

## Results

| Model | Eval scope | Macro-F1 | Cohen's κ |
|-------|------------|----------|-----------|
| RF baseline | 5-fold × 3 seeds | **0.609 ± 0.009** | 0.580 ± 0.012 |
| CNN1D | 2-fold, seed 42 | **0.697 ± 0.002** | 0.664 ± 0.003 |
| AttnSleep | _not evaluated_ | — | — |

CNN1D beats RF by **+0.09 macro-F1** on the same Sleep-EDF SC subset. AttnSleep skipped after MPS OOM (CPU fallback ~18+ h/epoch); see [`logs/FINAL_STATUS.md`](logs/FINAL_STATUS.md).

Live table: [`RESULTS.md`](RESULTS.md) · Blog: [arcsh.github.io/blog/sleepclassifier](https://arcsh.github.io/blog/sleepclassifier) · Writeup: [`reports/writeup.md`](reports/writeup.md)

## Quickstart

```bash
pip install -e ".[dev]"

# Full portfolio run (CNN1D ~5 h on M1; RF results cached)
caffeinate -dims python scripts/run_portfolio_sweep.py
python scripts/make_figures.py

# Or step-by-step
python scripts/download_data.py --subset sc
python scripts/preprocess.py
python scripts/train.py model=cnn1d training=portfolio fold_only=0 seed=42 \
  force_seed_subdir=true training.n_seeds=1 \
  experiment_name=poc_cnn1d_sleep_edf_sc output_dir=experiments/poc_cnn1d_sleep_edf_sc
python scripts/make_figures.py

# CI smoke (<2 min, synthetic data)
python scripts/demo_pipeline.py
ruff check . && black --check . && mypy src/ && python -m pytest tests/
```

## Layout

```
configs/          Hydra (data, model, training)
src/sleepstage/   Installable package
scripts/          CLI entrypoints
tests/            Unit + integration
reports/figures/  Generated plots
docs/             Data + model cards
large-scale-test/ Archived 60-combo research sweep
```

More: [`reports/writeup.md`](reports/writeup.md) · [`reports/blog.md`](reports/blog.md) · [`docs/`](docs/)

## Non-goals

Research / portfolio codebase — **not** a clinical scorer. No FDA claims, no SOTA chase. CNN-BiLSTM + AttnSleep implemented; not fully evaluated in the portfolio run. Full 5-fold × 3-seed sweep lives under [`large-scale-test/`](large-scale-test/).

## Citations

- Kemp et al. (2000). Analysis of a sleep-dependent neuronal feedback loop. *IEEE TBME*.
- Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*.
- Supratak et al. (2017). DeepSleepNet. *IEEE TPAMI*.
- Eldele et al. (2021). AttnSleep. *IEEE TBME*.

## License

MIT. Sleep-EDF under [PhysioNet Open Data License](https://physionet.org/content/sleep-edfx/view-license/1.0.0/).
