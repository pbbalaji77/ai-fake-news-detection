"""
Unit tests for src/preprocessing.py and src/data_loader.py
"""

import pandas as pd
import pytest

from src.config import (
    COL_CLEAN_TEXT,
    COL_ID,
    COL_LABEL,
    COL_TEXT,
    COL_TITLE,
    LABEL_FAKE,
    LABEL_REAL,
)
from src.data_loader import clean_dataset, load_raw_data, split_dataset, standardize_columns
from src.preprocessing import clean_text, combine_title_and_text, expand_contractions


# ==============================================================================
# Preprocessing Unit Tests
# ==============================================================================

def test_expand_contractions():
    """Verify contraction expansion."""
    raw = "Don't worry, they won't find out it's a secret."
    expanded = expand_contractions(raw)
    assert "do not" in expanded.lower()
    assert "will not" in expanded.lower()
    assert "it is" in expanded.lower()


def test_clean_text_basic():
    """Verify lowercasing, punctuation removal, and whitespace collapsing."""
    raw = "  BREAKING NEWS: Global summit reaches unprecedented accord!!!  "
    cleaned = clean_text(raw)
    assert cleaned == "breaking news global summit reaches unprecedented accord"


def test_clean_text_removes_urls_and_html():
    """Verify URLs and HTML tags are stripped properly."""
    raw = "<p>Visit <a href='https://fakenews.example.com'>here</a> for info http://test.org/news</p>"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "fakenews" not in cleaned
    assert "href" not in cleaned
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "visit" in cleaned
    assert "for info" in cleaned


def test_clean_text_edge_cases():
    """Verify resilience against empty, None, and non-string inputs."""
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("     ") == ""
    assert clean_text("!@#$%^&*()_+{}[]:;<>,.?/~`") == ""
    assert clean_text("12345 67890") == ""
    assert clean_text(12345) == ""


def test_combine_title_and_text():
    """Verify headline and body concatenation logic."""
    title = "Market Closes Higher"
    text = "The Dow Jones gained 200 points."
    combined = combine_title_and_text(title, text)
    assert combined == "Market Closes Higher The Dow Jones gained 200 points."

    # Handle missing components gracefully
    assert combine_title_and_text(title, "") == title
    assert combine_title_and_text("", text) == text
    assert combine_title_and_text(None, None) == ""


# ==============================================================================
# Data Loader & Cleaning Unit Tests
# ==============================================================================

def test_standardize_columns():
    """Verify column alias mapping and schema validation."""
    df = pd.DataFrame({
        "Headline": ["Test Title"],
        "Content": ["Test Body Content"],
        "Class": ["REAL"],
    })
    standardized = standardize_columns(df)
    assert COL_TITLE in standardized.columns
    assert COL_TEXT in standardized.columns
    assert COL_LABEL in standardized.columns


def test_standardize_columns_missing_required():
    """Verify error raised when required column is missing."""
    df = pd.DataFrame({"unrelated_col": [1, 2]})
    with pytest.raises(ValueError, match="Dataset missing required columns"):
        standardize_columns(df)


def test_clean_dataset_removes_duplicates_and_empty():
    """Verify removal of duplicates, empty articles, and label conversion."""
    raw_df = pd.DataFrame({
        "title": ["Real News Headline", "Real News Headline", "Fake Scandal", "", "   "],
        "text": ["Accurate report body.", "Accurate report body.", "Unverified rumor.", "", "   "],
        "label": ["REAL", "REAL", "FAKE", "REAL", "FAKE"],
    })
    
    cleaned = clean_dataset(raw_df)
    
    # Duplicates removed, empty removed -> should leave 2 rows
    assert len(cleaned) == 2
    assert set(cleaned[COL_LABEL].unique()) == {LABEL_REAL, LABEL_FAKE}
    assert COL_CLEAN_TEXT in cleaned.columns
    assert all(cleaned[COL_CLEAN_TEXT].str.len() > 0)


def test_clean_dataset_label_types():
    """Verify string and integer label parsing."""
    raw_df = pd.DataFrame({
        "title": ["News 1", "News 2", "News 3", "News 4", "News 5"],
        "text": ["Content 1", "Content 2", "Content 3", "Content 4", "Content 5"],
        "label": ["REAL", "FAKE", 0, 1, "INVALID_LABEL"],
    })
    
    cleaned = clean_dataset(raw_df)
    # The row with 'INVALID_LABEL' should be discarded
    assert len(cleaned) == 4
    assert list(cleaned[COL_LABEL]) == [LABEL_REAL, LABEL_FAKE, LABEL_REAL, LABEL_FAKE]


def test_split_dataset_stratification():
    """Verify stratified split maintains class balance."""
    # Create balanced dataset of 20 samples (10 REAL, 10 FAKE)
    titles = [f"Headline {i}" for i in range(20)]
    texts = [f"Body article text content number {i}" for i in range(20)]
    labels = [LABEL_REAL] * 10 + [LABEL_FAKE] * 10
    
    df = pd.DataFrame({
        COL_TITLE: titles,
        COL_TEXT: texts,
        COL_LABEL: labels,
        COL_CLEAN_TEXT: texts,
    })
    
    train_df, test_df = split_dataset(df, test_size=0.20, random_state=42)
    
    assert len(train_df) == 16
    assert len(test_df) == 4
    
    # Check stratified balance in test set (2 REAL, 2 FAKE)
    test_counts = test_df[COL_LABEL].value_counts()
    assert test_counts[LABEL_REAL] == 2
    assert test_counts[LABEL_FAKE] == 2
