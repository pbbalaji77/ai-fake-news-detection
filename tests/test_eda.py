"""
Unit tests for src/eda.py
"""

import pandas as pd
import pytest

from src.config import COL_CLEAN_TEXT, COL_LABEL, LABEL_FAKE, LABEL_REAL
from src.eda import (
    compute_class_distribution,
    compute_text_length_statistics,
    extract_top_ngrams,
    generate_eda_visualizations,
)


@pytest.fixture
def sample_eda_df():
    """Provides a synthetic DataFrame for EDA testing with varying text lengths."""
    return pd.DataFrame({
        COL_CLEAN_TEXT: [
            "federal reserve holds interest rates steady amid market inflation and economic stability",
            "nasa reveals ancient galaxy",
            "secret government lab admits tap water controls civilian thoughts and creates behavioral compliance among the population",
            "miracle root cures all cancers within twenty four hours and doctors are shocked",
        ],
        COL_LABEL: [LABEL_REAL, LABEL_REAL, LABEL_FAKE, LABEL_FAKE],
    })


def test_compute_class_distribution(sample_eda_df):
    """Verify class balance metric calculations."""
    metrics = compute_class_distribution(sample_eda_df)
    assert metrics["total_samples"] == 4
    assert metrics["real_count"] == 2
    assert metrics["fake_count"] == 2
    assert metrics["real_percentage"] == 50.0
    assert metrics["fake_percentage"] == 50.0
    assert metrics["imbalance_ratio"] == 1.0


def test_compute_text_length_statistics(sample_eda_df):
    """Verify word and character statistics by class."""
    stats = compute_text_length_statistics(sample_eda_df)
    assert "REAL" in stats
    assert "FAKE" in stats
    assert stats["REAL"]["avg_word_count"] > 0
    assert stats["FAKE"]["avg_word_count"] > 0
    assert stats["REAL"]["min_word_count"] <= stats["REAL"]["max_word_count"]


def test_extract_top_ngrams(sample_eda_df):
    """Verify unigram and bigram extraction."""
    texts = sample_eda_df[COL_CLEAN_TEXT].tolist()
    
    unigrams = extract_top_ngrams(texts, n=1, top_k=5)
    assert len(unigrams) > 0
    assert isinstance(unigrams[0], tuple)
    assert isinstance(unigrams[0][0], str)
    assert isinstance(unigrams[0][1], int)
    
    bigrams = extract_top_ngrams(texts, n=2, top_k=3)
    assert len(bigrams) > 0
    assert len(bigrams[0][0].split()) == 2


def test_generate_eda_visualizations(sample_eda_df, tmp_path):
    """Verify plot generation and image saving without display errors."""
    plot_paths = generate_eda_visualizations(sample_eda_df, output_dir=tmp_path)
    assert "class_distribution" in plot_paths
    assert "word_count_distribution" in plot_paths
    assert plot_paths["class_distribution"].exists()
    assert plot_paths["word_count_distribution"].exists()
