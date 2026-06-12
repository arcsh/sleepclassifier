# Model Cards — Sleep Stage Classifiers

## RF Baseline (`rf_baseline`)

- **Type:** LightGBM on handcrafted features (band power δ/θ/α/σ/β, spectral entropy, Hjorth, ZCR)
- **Input:** Single 30 s epoch, single EEG channel
- **Purpose:** Classical ceiling; same subject-wise splits as deep models
- **Failure modes:** Weak on N1; limited temporal context

## CNN1D (`cnn1d`)

- **Type:** Multi-layer 1D CNN on raw waveform (3000 samples @ 100 Hz)
- **Input:** Single epoch, no temporal context
- **Parameters:** ~500K (varies by config)
- **Failure modes:** N1↔W confusion; no transition modeling

## CNN-BiLSTM (`cnn_bilstm`)

- **Type:** DeepSleepNet-style CNN encoder + bidirectional LSTM + residual skip
- **Input:** Sequence of L=20 epochs (center-epoch prediction)
- **Failure modes:** Border epochs at night boundaries excluded; can over-predict N2

## AttnSleep (`attnsleep`)

- **Type:** Multi-resolution CNN branches + multi-head self-attention
- **Input:** Sequence of L=20 epochs
- **Failure modes:** Attention may focus on neighboring N2; N1 recall often lowest

## Evaluation protocol

All models: subject-independent k-fold CV, early stopping on validation macro-F1, class-weighted CE loss (default). Report accuracy, macro-F1, per-class F1, Cohen's κ, confusion matrices.

## Intended use

Research and education only. Not for clinical sleep scoring or diagnosis.
