"""
Unit tests for src/feature_extractor.py
"""

from pathlib import Path
import numpy as np
import pytest

from src.feature_extractor import TextFeatureExtractor


@pytest.fixture
def sample_corpus():
    """Sample corpus of clean news documents."""
    return [
        "federal reserve keeps benchmark interest rate steady amidst inflation report",
        "nasa telescope captures oldest galaxy in distant cosmos",
        "secret government laboratory puts brain chemicals into municipal tap water",
        "miracle herbal root claims to cure all cancers within twenty four hours",
        "european central bank holds interest rate meeting in frankfurt today",
    ]


def test_extractor_not_fitted_error():
    """Verify error when transforming before fitting."""
    extractor = TextFeatureExtractor()
    with pytest.raises(RuntimeError, match="Cannot transform texts before vectorizer is fitted"):
        extractor.transform(["test text"])


def test_fit_and_transform(sample_corpus):
    """Verify fitting and transformation generates expected sparse matrix."""
    extractor = TextFeatureExtractor(min_df=1, ngram_range=(1, 2))
    matrix = extractor.fit_transform(sample_corpus)
    
    assert matrix.shape[0] == len(sample_corpus)
    assert matrix.shape[1] > 0
    assert extractor.is_fitted is True

    vocab = extractor.get_feature_names()
    assert len(vocab) == matrix.shape[1]
    
    # Check that both unigram and bigram features exist
    has_unigram = any(" " not in feature for feature in vocab)
    has_bigram = any(" " in feature for feature in vocab)
    assert has_unigram
    assert has_bigram


def test_get_top_features_for_text(sample_corpus):
    """Verify document-level TF-IDF feature importance ranking."""
    extractor = TextFeatureExtractor(min_df=1, ngram_range=(1, 2))
    extractor.fit(sample_corpus)
    
    text = "federal reserve holds interest rate"
    top_features = extractor.get_top_features_for_text(text, top_n=5)
    
    assert len(top_features) > 0
    # Highest scoring term should be first
    assert top_features[0][1] >= top_features[-1][1]
    
    # Edge case: text with completely out-of-vocabulary words
    oov_text = "xylophone zookeeper quantum quasar"
    oov_features = extractor.get_top_features_for_text(oov_text)
    assert oov_features == []

    # Edge case: empty string
    assert extractor.get_top_features_for_text("") == []


def test_save_and_load_roundtrip(sample_corpus, tmp_path):
    """Verify serialization and deserialization of fitted vectorizer."""
    extractor = TextFeatureExtractor(min_df=1)
    extractor.fit(sample_corpus)
    original_features = extractor.get_feature_names()
    
    save_file = tmp_path / "test_vectorizer.joblib"
    extractor.save(save_file)
    assert save_file.exists()
    
    loaded_extractor = TextFeatureExtractor.load(save_file)
    assert loaded_extractor.is_fitted is True
    assert np.array_equal(original_features, loaded_extractor.get_feature_names())
    
    # Check transform on loaded extractor
    original_trans = extractor.transform(["federal reserve"])
    loaded_trans = loaded_extractor.transform(["federal reserve"])
    assert np.allclose(original_trans.toarray(), loaded_trans.toarray())
