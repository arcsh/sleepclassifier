# Data Card — Sleep-EDF Expanded (Sleep Cassette)

## Overview

Primary dataset: **Sleep Cassette (SC)** from Sleep-EDF Expanded v1.0.0 (PhysioNet). ~153 whole-night PSG recordings from 78 healthy subjects (ages 25–101), mostly two nights per subject.

## Provenance

- Fetched via `mne.datasets.sleep_physionet.age.fetch_data()`
- Raw EDFs cached under `data/raw/sc/` (not committed)
- Preprocessed epochs cached under `data/processed/`

## Signals used

- **Single-channel default:** `EEG Fpz-Cz` (read from EDF header, not hardcoded rate)
- **Multi-channel variant:** Fpz-Cz + Pz-Oz + EOG horizontal

## Preprocessing

1. Band-pass 0.3–35 Hz
2. 30 s non-overlapping epochs aligned to hypnogram
3. R&K stages 3+4 → N3; drop `?` and Movement time
4. Wake trimming: 30 min wake padding before/after sleep period (configurable)
5. Per-recording z-score normalization per channel
6. Subject-wise GroupKFold splits (persisted to `data/processed/splits/`)

## Class distribution

Wake trimming substantially reduces W dominance. See `reports/figures/03_class_distribution.png`. N1 remains underrepresented (~5–7%); handled via class-weighted loss at training time.

## Limitations

- Healthy subjects only (SC); not clinically validated
- Modest subject count; high variance across folds expected
- Age skew in SC cohort
- ST subset reserved for cross-dataset generalization (not mixed into primary training)

## License

PhysioNet Open Data License. Cite Kemp et al. (2000) and Goldberger et al. (2000).
