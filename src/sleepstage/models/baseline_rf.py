"""Random Forest / LightGBM classical baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sleepstage.features.handcrafted import extract_features_batch


@dataclass
class BaselineResult:
    """Training result for classical baseline."""

    model: object
    n_features: int


class RandomForestBaseline:
    """Handcrafted-feature classifier (not nn.Module)."""

    is_sequence_model = False
    n_classes: int = 5

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        classifier: str = "random_forest",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.classifier = classifier
        self._model: RandomForestClassifier | lgb.LGBMClassifier | None = None

    def fit(self, epochs: np.ndarray, labels: np.ndarray, sfreq: float) -> BaselineResult:
        """Train on handcrafted features.

        Args:
            epochs: Shape (N, C, T).
            labels: Integer labels.
            sfreq: Sampling rate.

        Returns:
            BaselineResult with fitted model.
        """
        X = extract_features_batch(epochs, sfreq)
        if self.classifier == "lightgbm":
            kwargs: dict[str, Any] = {
                "n_estimators": self.n_estimators,
                "class_weight": "balanced",
                "verbose": -1,
            }
            if self.max_depth is not None:
                kwargs["max_depth"] = self.max_depth
            self._model = lgb.LGBMClassifier(**kwargs)
        else:
            self._model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                n_jobs=-1,
            )
        self._model.fit(X, labels)
        return BaselineResult(model=self._model, n_features=X.shape[1])

    def predict(self, epochs: np.ndarray, sfreq: float) -> np.ndarray:
        """Predict stage labels.

        Args:
            epochs: Input epochs.
            sfreq: Sampling rate.

        Returns:
            Predicted label array.
        """
        assert self._model is not None, "Model not fitted"
        X = extract_features_batch(epochs, sfreq)
        return self._model.predict(X)

    def predict_proba(self, epochs: np.ndarray, sfreq: float) -> np.ndarray:
        """Predict class probabilities.

        Args:
            epochs: Input epochs.
            sfreq: Sampling rate.

        Returns:
            Probability array (N, n_classes).
        """
        assert self._model is not None, "Model not fitted"
        X = extract_features_batch(epochs, sfreq)
        return self._model.predict_proba(X)

    def count_parameters(self) -> int:
        return 0
