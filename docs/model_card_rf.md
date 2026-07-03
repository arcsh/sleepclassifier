# Model Card — RF Baseline

See shared evaluation protocol in [model_card.md](model_card.md).

- **Architecture:** sklearn Random Forest on handcrafted spectral features (LightGBM available but not used on M1 full-data runs due to segfault)
- **Parameters:** ~0 (tree ensemble, not neural)
- **Portfolio results:** 5-fold × 3 seeds on Sleep-EDF SC — **macro-F1 0.609 ± 0.009**, κ 0.580 ± 0.012
- **Per-class (typical):** N1 F1 lowest (~0.35–0.40); N2 strongest (~0.75+)
- **Known failure modes:** No temporal context; W/N1 confusion; random-label sanity ~0.20 macro-F1 on synthetic data
