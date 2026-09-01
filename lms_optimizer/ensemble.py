"""Training-only calibrated logistic stacking of market, DC, and Elo probabilities."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression

@dataclass
class EnsemblePrediction:
    probabilities: np.ndarray
    model: str = "logistic_stack"

class MarketEnsemble:
    def __init__(self) -> None:
        self.classifier: LogisticRegression | None = None

    def _features(self, components: np.ndarray) -> np.ndarray:
        values = np.asarray(components, dtype=float)
        if values.ndim != 2 or values.shape[1] != 9: raise ValueError("expected n x 9 component probabilities")
        return np.log(np.clip(values, 1e-8, 1.0))

    def fit(self, components: np.ndarray, outcomes: np.ndarray) -> "MarketEnsemble":
        x, y = self._features(components), np.asarray(outcomes, dtype=int)
        if len(x) != len(y) or len(np.unique(y)) < 2: raise ValueError("training needs matching rows and at least two outcome classes")
        self.classifier = LogisticRegression(max_iter=2000).fit(x, y)
        return self

    def predict(self, components: np.ndarray) -> EnsemblePrediction:
        if self.classifier is None: raise ValueError("fit the ensemble on training data first")
        probabilities = self.classifier.predict_proba(self._features(np.asarray(components)))
        full = np.zeros((len(probabilities), 3))
        for col, label in enumerate(self.classifier.classes_): full[:, label] = probabilities[:, col]
        return EnsemblePrediction(full)
