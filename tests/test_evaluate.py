"""
Unit tests for src/evaluate.py
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluate import (
    build_comparison_dataframe,
    evaluate_model,
    plot_confusion_matrices,
    plot_model_comparison,
    select_best_model,
)


@pytest.fixture
def mock_evaluation_data():
    """Synthetic fitted model, feature matrix, and labels for evaluation tests."""
    X_train = np.array([[1.0, 0.0], [0.8, 0.1], [0.1, 0.9], [0.0, 1.0]])
    y_train = np.array([0, 0, 1, 1])
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    X_test = np.array([[0.9, 0.1], [0.1, 0.8]])
    y_test = np.array([0, 1])
    return model, X_test, y_test


def test_evaluate_model(mock_evaluation_data):
    """Verify single model evaluation outputs."""
    model, X_test, y_test = mock_evaluation_data
    metrics = evaluate_model("Logistic Regression", model, X_test, y_test)

    assert metrics["model_name"] == "Logistic Regression"
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    assert "classification_report" in metrics


def test_build_comparison_and_select_best():
    """Verify comparison dataframe sorting and best model selection."""
    evaluations = {
        "Model A": {
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.85,
            "f1_score": 0.79,
            "roc_auc": 0.82,
            "train_time": 0.01,
        },
        "Model B": {
            "accuracy": 0.95,
            "precision": 0.95,
            "recall": 0.95,
            "f1_score": 0.95,
            "roc_auc": 0.96,
            "train_time": 0.02,
        },
    }
    mock_trained = {
        "Model A": {"model": "object_a"},
        "Model B": {"model": "object_b"},
    }

    comp_df = build_comparison_dataframe(evaluations)
    assert comp_df.iloc[0]["Model"] == "Model B"
    assert comp_df.iloc[0]["F1-Score"] == 0.95

    best_name, best_model, best_metrics = select_best_model(evaluations, mock_trained)
    assert best_name == "Model B"
    assert best_model == "object_b"
    assert best_metrics["f1_score"] == 0.95


def test_evaluation_plotting(tmp_path):
    """Verify confusion matrix and comparison plot generation."""
    evaluations = {
        "Logistic Regression": {
            "accuracy": 0.90,
            "precision": 0.90,
            "recall": 0.90,
            "f1_score": 0.90,
            "roc_auc": 0.92,
            "confusion_matrix": [[3, 0], [0, 3]],
            "train_time": 0.01,
        },
        "Naive Bayes": {
            "accuracy": 0.85,
            "precision": 0.85,
            "recall": 0.85,
            "f1_score": 0.85,
            "roc_auc": 0.87,
            "confusion_matrix": [[3, 0], [1, 2]],
            "train_time": 0.005,
        },
    }

    cm_file = tmp_path / "test_cm.png"
    comp_file = tmp_path / "test_comp.png"

    plot_confusion_matrices(evaluations, cm_file)
    assert cm_file.exists()

    comp_df = build_comparison_dataframe(evaluations)
    plot_model_comparison(comp_df, comp_file)
    assert comp_file.exists()
