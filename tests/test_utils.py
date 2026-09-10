"""
Unit tests for src/utils.py and src/config.py
"""

import tempfile
from pathlib import Path
import pytest

from src.config import (
    PROJECT_ROOT,
    SAMPLE_DATA_FILE,
    REQUIRED_COLUMNS,
    LABEL_REAL,
    LABEL_FAKE,
    LABEL_MAP,
)
from src.utils import (
    compute_basic_text_metrics,
    compute_sentiment_and_sensationalism,
    ensure_directory,
    save_json,
    load_json,
)


def test_config_paths_exist():
    """Verify project root and sample data paths are valid."""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()
    assert SAMPLE_DATA_FILE.exists()
    assert SAMPLE_DATA_FILE.is_file()


def test_config_label_mappings():
    """Verify label constants and bidirectional mapping."""
    assert LABEL_REAL == 0
    assert LABEL_FAKE == 1
    assert LABEL_MAP[0] == "REAL"
    assert LABEL_MAP[1] == "FAKE"


def test_compute_basic_text_metrics():
    """Verify text statistics calculation."""
    text = "The quick brown fox jumps over the lazy dog.\nSecond line here."
    metrics = compute_basic_text_metrics(text)
    
    assert metrics["word_count"] == 12
    assert metrics["character_count"] == len(text)
    assert metrics["line_count"] == 2


def test_compute_basic_text_metrics_empty():
    """Verify edge case handling for empty or None text."""
    empty_metrics = compute_basic_text_metrics("")
    assert empty_metrics["word_count"] == 0
    assert empty_metrics["character_count"] == 0
    assert empty_metrics["line_count"] == 0

    none_metrics = compute_basic_text_metrics(None)
    assert none_metrics["word_count"] == 0


def test_json_roundtrip():
    """Verify save_json and load_json operate consistently."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test_data.json"
        data = {"model_name": "LogisticRegression", "accuracy": 0.95}
        
        save_json(data, test_file)
        assert test_file.exists()
        
        loaded = load_json(test_file)
        assert loaded == data


def test_compute_sentiment_and_sensationalism():
    """Verify sensationalism and sentiment metric calculations."""
    sensational_text = "SHOCKING BOMBSHELL: Secret leaked documents prove miracle cancer cure!!!"
    sens_result = compute_sentiment_and_sensationalism(sensational_text)
    assert sens_result["uppercase_words"] >= 2
    assert sens_result["exclamation_marks"] == 3
    assert sens_result["sensationalism_score"] > 30.0
    assert sens_result["sensationalism_level"] in ["Moderate", "High"]

    neutral_text = "Treasury reported a modest increase in quarterly tax revenues on Tuesday."
    neutral_result = compute_sentiment_and_sensationalism(neutral_text)
    assert neutral_result["uppercase_words"] == 0
    assert neutral_result["exclamation_marks"] == 0
    assert neutral_result["sensationalism_level"] == "Low (Journalistic / Measured)"

    empty_result = compute_sentiment_and_sensationalism("")
    assert empty_result["sensationalism_score"] == 0.0

