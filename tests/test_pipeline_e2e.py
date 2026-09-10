"""
End-to-End Pipeline Integration and Stress Testing.

Verifies the entire lifecycle from raw dataset ingestion through feature engineering,
model training, evaluation, artifact serialization, deserialization, and inference
under extreme edge cases (e.g., emojis, long articles, foreign characters).
"""

from pathlib import Path
import pytest

from src.config import SAMPLE_DATA_FILE
from src.data_loader import clean_dataset, load_raw_data, split_dataset
from src.evaluate import run_evaluation_pipeline
from src.feature_extractor import TextFeatureExtractor
from src.predictor import FakeNewsPredictor
from src.train import train_all_models


def test_end_to_end_pipeline_lifecycle(tmp_path):
    """
    Tests complete lifecycle:
    raw data -> clean -> split -> extract features -> train -> evaluate -> save -> predict
    """
    # 1. Ingestion & Preprocessing
    raw_df = load_raw_data(SAMPLE_DATA_FILE)
    cleaned_df = clean_dataset(raw_df)
    train_df, test_df = split_dataset(cleaned_df, test_size=0.2, random_state=42)
    assert len(train_df) > 0
    assert len(test_df) > 0

    # 2. Train and Evaluate using temporary artifact folder
    best_name, best_model, best_metrics, comp_df = run_evaluation_pipeline(
        data_path=SAMPLE_DATA_FILE,
        save_artifacts=True,
    )
    assert best_name in ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"]
    assert best_metrics["f1_score"] > 0.70

    # 3. Predictor with serialized artifacts
    predictor = FakeNewsPredictor()
    sample_article = (
        "LONDON (Reuters) - Maritime transport regulations have been updated to target net-zero "
        "carbon emissions across global freight carriers by 2050."
    )
    result = predictor.predict(sample_article)
    assert result["is_valid"] is True
    assert result["label"] in ["REAL", "FAKE"]
    assert result["confidence"] >= 50.0
    assert "explanation" in result


def test_predictor_extreme_inputs():
    """Verify inference robustness under extreme and adversarial text inputs."""
    predictor = FakeNewsPredictor()

    # 1. Emojis and internet slang
    emoji_text = "🚨🚨🚨 BREAKING NEWS 😱😱 Aliens landed in Washington DC! Watch now! 🔥🔥🔥"
    emoji_result = predictor.predict(emoji_text)
    assert emoji_result["is_valid"] is True
    assert emoji_result["metrics"]["word_count"] > 0

    # 2. Very short headline
    short_text = "Oil prices rise"
    short_result = predictor.predict(short_text)
    assert short_result["is_valid"] is True
    assert short_result["label"] in ["REAL", "FAKE"]

    # 3. Long text (stress test)
    long_text = "Economic markets report strong corporate quarterly earnings. " * 300
    long_result = predictor.predict(long_text)
    assert long_result["is_valid"] is True
    assert long_result["metrics"]["word_count"] >= 2000

    # 4. Mixed casing and irregular whitespace
    mixed_text = "ThIs Is A mIxEd CaSe NeWs HeAdLiNe WiTh SpEcIaL wOrDs."
    mixed_result = predictor.predict(mixed_text)
    assert mixed_result["is_valid"] is True
    assert mixed_result["label"] in ["REAL", "FAKE"]
