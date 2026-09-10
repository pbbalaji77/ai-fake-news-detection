"""
Configuration module for the AI-Powered Fake News Detection System.

Defines all filesystem paths dynamically using pathlib to guarantee 100%
cross-platform compatibility across Windows, macOS, Linux, and Streamlit Cloud.
No hardcoded absolute paths.
"""

from pathlib import Path

# Project Root Directory (detected relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core Directory Hierarchy
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
ASSETS_DIR = PROJECT_ROOT / "assets"
SCREENSHOTS_DIR = ASSETS_DIR / "screenshots"

# Specific File Paths
SAMPLE_DATA_FILE = SAMPLE_DATA_DIR / "sample_news.csv"
RAW_DATA_FILE = RAW_DATA_DIR / "fake_or_real_news.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model Artifacts
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# Data Schema & Column Names
COL_ID = "id"
COL_TITLE = "title"
COL_TEXT = "text"
COL_LABEL = "label"
COL_CLEAN_TEXT = "clean_text"

REQUIRED_COLUMNS = [COL_TITLE, COL_TEXT, COL_LABEL]

# Label Definitions (Binary Classification)
# 0 = REAL (Credible/Authentic journalism)
# 1 = FAKE (Fabricated/Misleading news)
LABEL_REAL = 0
LABEL_FAKE = 1

LABEL_MAP = {
    LABEL_REAL: "REAL",
    LABEL_FAKE: "FAKE",
}

LABEL_TO_ID = {
    "REAL": LABEL_REAL,
    "FAKE": LABEL_FAKE,
    "0": LABEL_REAL,
    "1": LABEL_FAKE,
    0: LABEL_REAL,
    1: LABEL_FAKE,
    "TRUE": LABEL_REAL,
    "FALSE": LABEL_FAKE,
}

# Machine Learning Reproducibility & Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.20

# TF-IDF Feature Engineering Defaults
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.90
TFIDF_SUBLINEAR_TF = True

# Explainability Settings
TOP_N_EXPLANATION_FEATURES = 10

# Ethical Notice & Disclaimer
SYSTEM_DISCLAIMER = (
    "⚠️ **Disclaimer**: This tool uses machine learning pattern recognition trained on "
    "linguistic features and historical news datasets. It provides a probabilistic assessment "
    "and is NOT a definitive fact-checking authority. Always verify critical news claims through "
    "primary journalistic sources, official records, and accredited fact-checking organizations."
)
