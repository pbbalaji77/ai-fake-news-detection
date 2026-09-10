"""
Unit tests for src/explainability.py
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.explainability import (
    explain_prediction,
    get_global_top_features,
    get_model_feature_weights,
)
from src.feature_extractor import TextFeatureExtractor


@pytest.fixture
def fitted_explainability_setup():
    """Provides a fitted vectorizer and models for explainability tests."""
    corpus = [
        "reuters washington federal reserve economic growth report",
        "central bank interest rates inflation monetary policy meeting",
        "secret leaked documents alien tap water mind control",
        "miracle herbal root cancer cure conspiracy big pharma",
    ]
    labels = np.array([0, 0, 1, 1])

    extractor = TextFeatureExtractor(min_df=1)
    X = extractor.fit_transform(corpus)

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X, labels)

    nb_model = MultinomialNB()
    nb_model.fit(X, labels)

    return extractor, lr_model, nb_model


def test_get_model_feature_weights(fitted_explainability_setup):
    """Verify weight vector length matches vocabulary size."""
    extractor, lr_model, nb_model = fitted_explainability_setup
    vocab_len = len(extractor.get_feature_names())

    lr_weights = get_model_feature_weights(lr_model, extractor)
    assert len(lr_weights) == vocab_len
    assert isinstance(lr_weights, np.ndarray)

    nb_weights = get_model_feature_weights(nb_model, extractor)
    assert len(nb_weights) == vocab_len


def test_get_global_top_features(fitted_explainability_setup):
    """Verify global top features extraction."""
    extractor, lr_model, _ = fitted_explainability_setup
    global_feats = get_global_top_features(lr_model, extractor, top_n=3)

    assert "indicative_of_fake" in global_feats
    assert "indicative_of_real" in global_feats
    assert len(global_feats["indicative_of_fake"]) > 0
    assert len(global_feats["indicative_of_real"]) > 0


def test_explain_prediction_local(fitted_explainability_setup):
    """Verify document-level explanation generation."""
    extractor, lr_model, _ = fitted_explainability_setup
    sample_fake = "shocking leaked documents show secret tap water"

    exp = explain_prediction(
        text=sample_fake,
        model=lr_model,
        feature_extractor=extractor,
        predicted_label="FAKE",
        confidence=95.0,
        top_n=3,
    )

    assert "fake_signals" in exp
    assert "real_signals" in exp
    assert "summary_text" in exp
    assert "disclaimer" in exp
    assert len(exp["fake_signals"]) > 0


def test_explain_prediction_empty_and_oov(fitted_explainability_setup):
    """Verify explainability handles empty and OOV strings safely."""
    extractor, lr_model, _ = fitted_explainability_setup

    empty_exp = explain_prediction(
        text="",
        model=lr_model,
        feature_extractor=extractor,
        predicted_label="REAL",
        confidence=50.0,
    )
    assert "insufficient text" in empty_exp["summary_text"]

    oov_exp = explain_prediction(
        text="quantum quasar galaxy supernova",
        model=lr_model,
        feature_extractor=extractor,
        predicted_label="REAL",
        confidence=50.0,
    )
    assert "training vocabulary" in oov_exp["summary_text"]
