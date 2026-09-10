"""
Model Training Pipeline for Fake News Detection.

Trains and encapsulates multiple classification architectures:
1. Logistic Regression: Linear probabilistic baseline with L2 regularization.
2. Multinomial Naive Bayes: Fast probabilistic classifier based on Bayes' theorem.
3. Linear Support Vector Machine (Linear SVM): Maximum-margin linear classifier calibrated
   with Platt scaling (CalibratedClassifierCV) to provide true probability outputs.

Guarantees strict train/test isolation to prevent data leakage.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import (
    COL_CLEAN_TEXT,
    COL_LABEL,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import clean_dataset, load_raw_data, split_dataset
from src.feature_extractor import TextFeatureExtractor
from src.utils import setup_logger

logger = setup_logger("train")


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Initializes the candidate classification models with tuned hyperparameter defaults.
    Linear SVM is wrapped in CalibratedClassifierCV to enable reliable probability estimation.
    
    Args:
        random_state: Random seed for deterministic reproducibility.
        
    Returns:
        Dictionary mapping model names to un-fitted scikit-learn estimators.
    """
    models: Dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0,
            fit_prior=True,
        ),
        "Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                random_state=random_state,
                max_iter=2000,
                dual="auto",
            ),
            method="sigmoid",
            cv=3,
        ),
    }
    return models


def train_single_model(
    model_name: str,
    model: BaseEstimator,
    X_train: Any,
    y_train: np.ndarray,
) -> Tuple[BaseEstimator, float]:
    """
    Fits an individual model on training features and records training latency.
    
    Args:
        model_name: Descriptive name of the algorithm.
        model: Scikit-learn estimator.
        X_train: Sparse TF-IDF training feature matrix.
        y_train: Training labels array.
        
    Returns:
        Tuple of (fitted_model, duration_seconds).
    """
    logger.info(f"Training [{model_name}]...")
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time
    logger.info(f"Finished [{model_name}] in {elapsed:.4f}s.")
    return model, elapsed


def train_all_models(
    X_train: Any,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, Any]]:
    """
    Trains all candidate classification models on the training feature matrix.
    
    Args:
        X_train: Sparse TF-IDF feature matrix.
        y_train: Training labels.
        random_state: Seed for reproducibility.
        
    Returns:
        Dictionary mapping model names to dictionaries containing 'model' and 'train_time'.
    """
    candidate_models = get_candidate_models(random_state=random_state)
    trained_results: Dict[str, Dict[str, Any]] = {}

    for name, model in candidate_models.items():
        fitted_model, duration = train_single_model(name, model, X_train, y_train)
        trained_results[name] = {
            "model": fitted_model,
            "train_time": duration,
        }

    return trained_results


def run_training_pipeline(
    data_path: Optional[Path] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, Dict[str, Any]], TextFeatureExtractor, Any, np.ndarray]:
    """
    Full end-to-end training orchestrator:
    1. Loads raw or sample data.
    2. Cleans and normalizes dataset.
    3. Performs stratified train/test split.
    4. Fits TF-IDF vectorizer strictly on training data (no leakage).
    5. Transforms training and testing splits.
    6. Trains all candidate models.
    
    Args:
        data_path: Optional custom dataset path.
        test_size: Test split ratio.
        random_state: Seed for reproducibility.
        
    Returns:
        Tuple of (trained_models_dict, feature_extractor, X_test, y_test).
    """
    logger.info("Initializing ML training pipeline...")
    raw_df = load_raw_data(data_path)
    cleaned_df = clean_dataset(raw_df)
    train_df, test_df = split_dataset(cleaned_df, test_size=test_size, random_state=random_state)

    train_texts = train_df[COL_CLEAN_TEXT].tolist()
    y_train = train_df[COL_LABEL].to_numpy()

    test_texts = test_df[COL_CLEAN_TEXT].tolist()
    y_test = test_df[COL_LABEL].to_numpy()

    # Fit feature extractor strictly on training data
    feature_extractor = TextFeatureExtractor()
    X_train = feature_extractor.fit_transform(train_texts)
    X_test = feature_extractor.transform(test_texts)

    logger.info(
        f"Feature matrices created: X_train shape={X_train.shape}, X_test shape={X_test.shape}"
    )

    # Train all candidate models
    trained_models = train_all_models(X_train, y_train, random_state=random_state)
    return trained_models, feature_extractor, X_test, y_test


def main() -> None:
    """CLI runner for training models."""
    trained_models, extractor, X_test, y_test = run_training_pipeline()
    print("\n" + "=" * 60)
    print("           MODEL TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    for name, res in trained_models.items():
        print(f"[OK] Model: {name:<25} | Fit Time: {res['train_time']:.4f}s")
    print(f"Test evaluation set prepared: {X_test.shape[0]} samples")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
