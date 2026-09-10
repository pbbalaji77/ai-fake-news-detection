"""
Utility functions for logging, file serialization, and text metrics.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import joblib


def setup_logger(name: str = "fake_news_detector", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standardized logger.
    
    Args:
        name: Name of the logger instance.
        level: Logging level (default: logging.INFO).
        
    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_directory(path: Path) -> Path:
    """
    Ensures that the directory for a given path exists; creates it if needed.
    
    Args:
        path: Path object representing a file or directory.
        
    Returns:
        The verified Path object.
    """
    if path.suffix:
        directory = path.parent
    else:
        directory = path
    directory.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 4) -> None:
    """
    Saves a dictionary to a JSON file safely.
    
    Args:
        data: Dictionary data to persist.
        file_path: Destination path.
        indent: JSON indentation formatting.
    """
    ensure_directory(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Loads JSON data from file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Loaded dictionary.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_artifact(obj: Any, file_path: Path) -> None:
    """
    Serializes an object (model, vectorizer) using joblib.
    
    Args:
        obj: Python object to serialize.
        file_path: Destination path.
    """
    ensure_directory(file_path)
    joblib.dump(obj, file_path)


def load_artifact(file_path: Path) -> Any:
    """
    Loads a joblib-serialized artifact from file.
    
    Args:
        file_path: Path to serialized artifact.
        
    Returns:
        Deserialized Python object.
        
    Raises:
        FileNotFoundError: If artifact does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact not found at: {file_path}")
    return joblib.load(file_path)


def compute_basic_text_metrics(text: str) -> Dict[str, int]:
    """
    Computes basic descriptive statistics for text input.
    
    Args:
        text: Raw input string.
        
    Returns:
        Dictionary containing character_count, word_count, and line_count.
    """
    if not text or not isinstance(text, str):
        return {"character_count": 0, "word_count": 0, "line_count": 0}
    
    chars = len(text)
    words = len(text.split())
    lines = len([line for line in text.splitlines() if line.strip()])
    return {
        "character_count": chars,
        "word_count": words,
        "line_count": lines,
    }


def compute_sentiment_and_sensationalism(text: str) -> Dict[str, Any]:
    """
    Analyzes rhetorical sensationalism and emotional tone markers.
    
    Sensationalist and clickbait news articles frequently employ:
    1. Excessive uppercase lettering (SHOCKING, EXPOSED, UNBELIEVABLE).
    2. Punctuation intensity (!!!, ???).
    3. Emotionally charged adjectives and extreme superlatives.
    
    Args:
        text: Raw input text.
        
    Returns:
        Dictionary with sensationalism score, punctuation metrics, and sentiment indicator.
    """
    if not text or not isinstance(text, str):
        return {
            "uppercase_words": 0,
            "exclamation_marks": 0,
            "question_marks": 0,
            "sensationalism_score": 0.0,
            "sensationalism_level": "Low",
            "sentiment_label": "Neutral",
        }

    words = text.split()
    total_words = max(len(words), 1)

    # Count all-caps words (ignoring single letters like 'I' or 'A')
    uppercase_words = []
    for w in words:
        clean_w = w.strip(".,!?:;\"'()[]{}")
        if clean_w.isupper() and len(clean_w) > 1 and clean_w.isalpha():
            uppercase_words.append(clean_w)
    upper_count = len(uppercase_words)

    # Count dramatic punctuation
    exclamation_count = text.count("!")
    question_count = text.count("?")

    # Sensationalism cue words common in misleading headlines
    sensational_cues = {
        "shocking", "breaking", "exposed", "secret", "miracle", "bombshell",
        "conspiracy", "unbelievable", "panicking", "leaked", "hoax", "bizarre",
        "covert", "plot", "forbidden", "destroy", "annihilate", "hidden",
    }
    lowered_words = [w.strip(".,!?:;\"'()[]{}").lower() for w in words]
    cue_matches = sum(1 for w in lowered_words if w in sensational_cues)

    # Calculate composite sensationalism score (0.0 to 100.0)
    caps_weight = (upper_count / total_words) * 150
    punct_weight = min(exclamation_count * 8 + question_count * 4, 40)
    cue_weight = min(cue_matches * 15, 45)

    composite_score = min(round(caps_weight + punct_weight + cue_weight, 1), 100.0)

    if composite_score >= 45.0:
        sensationalism_level = "High"
    elif composite_score >= 18.0:
        sensationalism_level = "Moderate"
    else:
        sensationalism_level = "Low (Journalistic / Measured)"

    # Basic sentiment polarity estimation based on positive/negative lexical balance
    positive_lexicon = {
        "progress", "agreement", "growth", "approved", "successful", "solution",
        "support", "record", "peace", "gain", "stable", "rebound", "benefit",
    }
    negative_lexicon = {
        "crisis", "threat", "collapse", "deficit", "conflict", "decline",
        "outlaw", "catastrophic", "corrupt", "terrifying", "warning", "panic",
    }
    pos_matches = sum(1 for w in lowered_words if w in positive_lexicon)
    neg_matches = sum(1 for w in lowered_words if w in negative_lexicon)

    if pos_matches > neg_matches:
        sentiment_label = "Positive / Constructive"
    elif neg_matches > pos_matches:
        sentiment_label = "Negative / Critical"
    else:
        sentiment_label = "Neutral / Objective"

    return {
        "uppercase_words": upper_count,
        "exclamation_marks": exclamation_count,
        "question_marks": question_count,
        "sensationalism_score": composite_score,
        "sensationalism_level": sensationalism_level,
        "sentiment_label": sentiment_label,
    }

