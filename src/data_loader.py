"""
Data Loading and Validation Module.

Responsible for reading raw news datasets, standardizing schema, handling missing values,
removing duplicates, standardizing labels, combining text fields, running text cleaning,
and performing stratified train/test splits.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    COL_CLEAN_TEXT,
    COL_ID,
    COL_LABEL,
    COL_TEXT,
    COL_TITLE,
    LABEL_FAKE,
    LABEL_REAL,
    LABEL_TO_ID,
    PROCESSED_DATA_DIR,
    PROCESSED_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    RANDOM_STATE,
    RAW_DATA_FILE,
    REQUIRED_COLUMNS,
    SAMPLE_DATA_FILE,
    TEST_SIZE,
)
from src.preprocessing import clean_text, combine_title_and_text
from src.utils import ensure_directory, setup_logger

logger = setup_logger("data_loader")


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the news dataset from disk. Checks specified path, falls back to RAW_DATA_FILE,
    and then falls back to SAMPLE_DATA_FILE if raw data is unavailable.
    
    Args:
        file_path: Optional custom Path to CSV file.
        
    Returns:
        pd.DataFrame containing raw data.
        
    Raises:
        FileNotFoundError: If neither raw nor sample data files exist.
    """
    target_path: Optional[Path] = None

    if file_path and Path(file_path).exists():
        target_path = Path(file_path)
    elif RAW_DATA_FILE.exists():
        target_path = RAW_DATA_FILE
    elif SAMPLE_DATA_FILE.exists():
        logger.info(f"Raw data file not found. Loading benchmark sample data: {SAMPLE_DATA_FILE}")
        target_path = SAMPLE_DATA_FILE
    else:
        raise FileNotFoundError(
            f"No dataset found. Please ensure {RAW_DATA_FILE} or {SAMPLE_DATA_FILE} exists."
        )

    logger.info(f"Loading data from: {target_path}")
    df = pd.read_csv(target_path)
    logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to lowercase and handles common aliases.
    e.g., 'headline' -> 'title', 'content' -> 'text'.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with standardized column names.
    """
    df = df.copy()
    # Normalize column names to lowercase stripped strings
    df.columns = [str(c).strip().lower() for c in df.columns]

    alias_map = {
        "headline": COL_TITLE,
        "content": COL_TEXT,
        "article": COL_TEXT,
        "target": COL_LABEL,
        "class": COL_LABEL,
    }
    df = df.rename(columns=alias_map)

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns: {missing_cols}. Existing columns: {list(df.columns)}"
        )

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by handling missing values, dropping duplicates,
    mapping labels, and generating the cleaned text feature column.
    
    Processing steps:
    1. Standardize column names.
    2. Fill missing text/title with empty strings.
    3. Remove rows where both title and text are blank.
    4. Remove duplicate rows based on title and text.
    5. Standardize labels to binary 0 (REAL) and 1 (FAKE).
    6. Drop any invalid or unmapped labels.
    7. Generate combined text and cleaned text representation.
    8. Drop any rows where clean_text is empty.
    
    Args:
        df: Raw DataFrame.
        
    Returns:
        Cleaned, validated pd.DataFrame ready for splitting and modeling.
    """
    initial_count = len(df)
    df = standardize_columns(df)

    # Handle missing values in text and title
    df[COL_TITLE] = df[COL_TITLE].fillna("").astype(str)
    df[COL_TEXT] = df[COL_TEXT].fillna("").astype(str)

    # Filter out records where both title and text are completely blank
    non_empty_mask = (df[COL_TITLE].str.strip() != "") | (df[COL_TEXT].str.strip() != "")
    df = df[non_empty_mask].copy()

    # Drop duplicate records based on title and text content
    df = df.drop_duplicates(subset=[COL_TITLE, COL_TEXT]).copy()

    # Standardize labels
    def map_label(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            int_val = int(val)
            if int_val in (LABEL_REAL, LABEL_FAKE):
                return int_val
        key = str(val).strip().upper()
        return LABEL_TO_ID.get(key, None)

    df[COL_LABEL] = df[COL_LABEL].apply(map_label)
    
    # Drop records with invalid or missing labels
    valid_labels_mask = df[COL_LABEL].isin([LABEL_REAL, LABEL_FAKE])
    df = df[valid_labels_mask].copy()
    df[COL_LABEL] = df[COL_LABEL].astype(int)

    # Combine title and text, then clean text
    df["combined_text"] = df.apply(
        lambda row: combine_title_and_text(row[COL_TITLE], row[COL_TEXT]), axis=1
    )
    df[COL_CLEAN_TEXT] = df["combined_text"].apply(clean_text)

    # Drop records where cleaned text resulted in an empty string
    df = df[df[COL_CLEAN_TEXT].str.strip() != ""].copy()
    df = df.reset_index(drop=True)

    final_count = len(df)
    logger.info(
        f"Data cleaning complete. Retained {final_count}/{initial_count} records "
        f"({initial_count - final_count} dropped). Label distribution: "
        f"REAL={int((df[COL_LABEL] == LABEL_REAL).sum())}, "
        f"FAKE={int((df[COL_LABEL] == LABEL_FAKE).sum())}"
    )

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a reproducible stratified train-test split to preserve class balance.
    
    Args:
        df: Cleaned DataFrame.
        test_size: Fraction allocated to test set (default: 0.20).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, test_df).
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    # Check minimum class frequency for stratification
    label_counts = df[COL_LABEL].value_counts()
    can_stratify = (label_counts.min() >= 2) and (len(label_counts) > 1)

    stratify_target = df[COL_LABEL] if can_stratify else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        f"Dataset split completed. Train: {len(train_df)} rows, Test: {len(test_df)} rows. "
        f"Stratification: {'Enabled' if can_stratify else 'Disabled'}"
    )
    return train_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Persists cleaned train and test datasets as CSV files.
    
    Args:
        train_df: Cleaned training DataFrame.
        test_df: Cleaned testing DataFrame.
        output_dir: Destination folder (defaults to PROCESSED_DATA_DIR).
        
    Returns:
        Tuple of (train_file_path, test_file_path).
    """
    dest_dir = output_dir if output_dir else PROCESSED_DATA_DIR
    ensure_directory(dest_dir)

    train_path = dest_dir / "train.csv"
    test_path = dest_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved processed train set to: {train_path}")
    logger.info(f"Saved processed test set to: {test_path}")

    return train_path, test_path


def main() -> None:
    """CLI pipeline for data loading, cleaning, splitting, and saving."""
    logger.info("Starting data loading and preprocessing pipeline...")
    raw_df = load_raw_data()
    cleaned_df = clean_dataset(raw_df)
    train_df, test_df = split_dataset(cleaned_df)
    train_file, test_file = save_processed_data(train_df, test_df)
    print("\n--- DATA LOADING & PREPROCESSING COMPLETE ---")
    print(f"Cleaned Total Records: {len(cleaned_df)}")
    print(f"Train Records:         {len(train_df)} -> {train_file}")
    print(f"Test Records:          {len(test_df)} -> {test_file}")
    print("---------------------------------------------\n")


if __name__ == "__main__":
    main()

