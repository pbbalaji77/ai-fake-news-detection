"""
Unit tests for src/predictor.py
"""

import pytest

from src.config import LABEL_FAKE, LABEL_REAL
from src.predictor import FakeNewsPredictor


@pytest.fixture(scope="module")
def predictor():
    """Initializes a shared predictor instance using serialized model artifacts."""
    return FakeNewsPredictor()


def test_predictor_initialization(predictor):
    """Verify predictor loads model and metadata."""
    assert predictor.model is not None
    assert predictor.extractor is not None
    assert predictor.metadata is not None
    assert isinstance(predictor.model_name, str)


def test_prediction_real_sample(predictor):
    """Verify prediction output on a credible news report."""
    real_text = (
        "WASHINGTON (Reuters) - The Federal Reserve kept its benchmark interest rate "
        "unchanged on Wednesday following a two-day monetary policy meeting. "
        "Policymakers noted that economic activity has continued to expand."
    )
    result = predictor.predict(real_text)
    
    assert result["is_valid"] is True
    assert result["label"] in ["REAL", "FAKE"]
    assert 0.0 <= result["confidence"] <= 100.0
    assert abs((result["probability_real"] + result["probability_fake"]) - 100.0) < 0.5
    assert result["metrics"]["word_count"] > 20
    assert len(result["top_keywords"]) > 0


def test_prediction_fake_sample(predictor):
    """Verify prediction output on a sensationalist fake news text."""
    fake_text = (
        "SHOCKING LEAK: Secret underground lab admits tap water is secretly laced "
        "with alien frequency chemicals to control human thoughts and suppress consciousness."
    )
    result = predictor.predict(fake_text)
    
    assert result["is_valid"] is True
    assert result["label"] in ["REAL", "FAKE"]
    assert 0.0 <= result["confidence"] <= 100.0
    assert len(result["top_keywords"]) > 0


def test_prediction_empty_and_whitespace(predictor):
    """Verify graceful handling of empty or blank inputs."""
    empty_res = predictor.predict("")
    assert empty_res["is_valid"] is False
    assert "error_message" in empty_res
    assert empty_res["metrics"]["word_count"] == 0

    blank_res = predictor.predict("       \n\t   ")
    assert blank_res["is_valid"] is False
    assert "error_message" in blank_res


def test_prediction_invalid_characters_only(predictor):
    """Verify handling of text with only symbols or numbers."""
    symbols_res = predictor.predict("$$$ !!! ??? 12345 67890 @#&")
    assert symbols_res["is_valid"] is False
    assert "error_message" in symbols_res
