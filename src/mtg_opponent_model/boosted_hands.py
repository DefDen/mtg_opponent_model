"""Shared histogram-boosted card-presence model, with separate Platt calibration."""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class BoostedHandModel:
    seed: int = 42
    iterations: int = 80

    def fit(self, features: np.ndarray, labels: np.ndarray) -> 'BoostedHandModel':
        """All candidate rows included; no class reweighting or sampled negatives."""
        self.estimator = HistGradientBoostingClassifier(
            max_iter=self.iterations, learning_rate=0.08, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=10, early_stopping=False,
            random_state=self.seed)
        self.estimator.fit(features, labels)
        self.calibrator = None
        return self

    def calibrate(self, validation_features: np.ndarray, validation_labels: np.ndarray) -> None:
        """Fit one sigmoid using event-disjoint validation data, never test labels."""
        scores = self.estimator.decision_function(validation_features).reshape(-1,1)
        self.calibrator = LogisticRegression(C=100, max_iter=300, random_state=self.seed)
        self.calibrator.fit(scores,validation_labels)

    def predict(self, features: np.ndarray, calibrated: bool = True) -> np.ndarray:
        if calibrated and self.calibrator is not None:
            scores = self.estimator.decision_function(features).reshape(-1,1)
            probabilities = self.calibrator.predict_proba(scores)[:,1]
        else:
            probabilities = self.estimator.predict_proba(features)[:,1]
        # A zero generative probability here is a finite-slot/empty-hand
        # constraint, not merely an absent historical card combination.
        return np.where(features[:,0] > 0, probabilities, 0.0)
