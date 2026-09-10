"""
Unit tests for src/train.py
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.train import (
    get_candidate_models,
    train_all_models,
    train_single_model,
)


@pytest.fixture
def synthetic_training_data():
    """Provides synthetic sparse training features and binary labels."""
    # 10 samples, 20 features
    rng = np.random.RandomState(42)
    dense = rng.binomial(1, 0.3, size=(10, 20)).astype(float)
    X_train = csr_matrix(dense)
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X_train, y_train


def test_get_candidate_models():
    """Verify that all required candidate models are registered."""
    models = get_candidate_models()
    required = ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"]
    for name in required:
        assert name in models
        # Check estimator methods exist
        assert hasattr(models[name], "fit")
        assert hasattr(models[name], "predict")


def test_train_all_models(synthetic_training_data):
    """Verify fitting across all candidate models and probability support."""
    X_train, y_train = synthetic_training_data
    results = train_all_models(X_train, y_train, random_state=42)

    assert len(results) == 3
    for name, res in results.items():
        assert "model" in res
        assert "train_time" in res
        assert res["train_time"] >= 0.0

        model = res["model"]
        # All models must support predict and predict_proba
        assert hasattr(model, "predict_proba"), f"{name} does not support predict_proba"
        
        preds = model.predict(X_train)
        probs = model.predict_proba(X_train)

        assert len(preds) == X_train.shape[0]
        assert probs.shape == (X_train.shape[0], 2)
        # Probabilities must sum to 1.0 across classes
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
