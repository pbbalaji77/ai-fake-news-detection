"""
Dataset Acquisition and Management Utility

This script guides the setup of public datasets (such as Kaggle Fake News or ISOT),
verifies raw and sample data schemas, and provides helper commands for reproducible data loading.

Dataset Schema Expected:
- id: integer or string identifier
- title: string, headline of the article
- text: string, main body text of the article
- label: integer or string (0/REAL = genuine news, 1/FAKE = fabricated news)
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import (
    COL_ID,
    COL_LABEL,
    COL_TEXT,
    COL_TITLE,
    LABEL_FAKE,
    LABEL_REAL,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    REQUIRED_COLUMNS,
    SAMPLE_DATA_FILE,
)
from src.utils import setup_logger

logger = setup_logger("dataset_manager")

DATASET_RESOURCES = """
================================================================================
PUBLIC DATASET RECOMMENDATIONS FOR FAKE NEWS DETECTION
================================================================================
1. Kaggle Fake News Dataset (Getting Real about Fake News / Fake News Detection):
   URL: https://www.kaggle.com/c/fake-news/data
   Contains: train.csv (id, title, author, text, label: 1=unreliable, 0=reliable)

2. ISOT Fake News Dataset (University of Victoria):
   URL: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php
   Contains: True.csv (Reuters verified articles), Fake.csv (flagged fake articles)

3. WELFake (Word Embedding Overfitting Fake News Dataset):
   URL: https://zenodo.org/record/4561253 (72,134 news articles)

================================================================================
HOW TO LOAD YOUR OWN DATASET:
Place your CSV in 'data/raw/fake_or_real_news.csv' with columns:
  'title', 'text', 'label'
Where label is either 0/REAL or 1/FAKE.
================================================================================
"""


def print_dataset_info() -> None:
    """Prints documentation about public datasets."""
    print(DATASET_RESOURCES)


def check_datasets() -> bool:
    """
    Checks the status of available datasets (raw and sample).
    
    Returns:
        bool: True if at least the sample or raw dataset is present and valid.
    """
    logger.info("Checking dataset status...")
    
    sample_exists = SAMPLE_DATA_FILE.exists()
    raw_exists = RAW_DATA_FILE.exists()
    
    print("\n--- DATASET INTEGRITY CHECK ---")
    if sample_exists:
        try:
            df_sample = pd.read_csv(SAMPLE_DATA_FILE)
            real_count = (df_sample[COL_LABEL] == LABEL_REAL).sum()
            fake_count = (df_sample[COL_LABEL] == LABEL_FAKE).sum()
            print(f"[OK] Sample Dataset Found: {SAMPLE_DATA_FILE}")
            print(f"     Total records: {len(df_sample)} (REAL: {real_count}, FAKE: {fake_count})")
            print(f"     Columns: {list(df_sample.columns)}")
        except Exception as e:
            print(f"[ERROR] Error reading sample dataset: {e}")
            return False
    else:
        print(f"[ERROR] Sample Dataset NOT found at: {SAMPLE_DATA_FILE}")

    if raw_exists:
        try:
            df_raw = pd.read_csv(RAW_DATA_FILE)
            print(f"[OK] Raw Dataset Found: {RAW_DATA_FILE}")
            print(f"     Total records: {len(df_raw)}")
            print(f"     Columns: {list(df_raw.columns)}")
        except Exception as e:
            print(f"[ERROR] Error reading raw dataset: {e}")
    else:
        print(f"[INFO] Raw Dataset not present at: {RAW_DATA_FILE} (Using sample benchmark dataset)")
        
    print("-------------------------------\n")
    return sample_exists or raw_exists


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset management tool for Fake News Detector.")
    parser.add_argument("--info", action="store_true", help="Display public dataset resources and schema guidelines.")
    parser.add_argument("--check", action="store_true", help="Check integrity of existing sample and raw datasets.")
    
    args = parser.parse_args()
    
    if args.info or len(sys.argv) == 1:
        print_dataset_info()
    if args.check or len(sys.argv) == 1:
        check_datasets()


if __name__ == "__main__":
    main()
