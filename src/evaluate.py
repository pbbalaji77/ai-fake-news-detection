"""
Model Evaluation and Selection Engine.

Computes comprehensive classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC),
generates confusion matrix heatmaps, compares candidate models, and selects the best performer
based on balanced F1-score and ROC-AUC criteria.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Use non-interactive Agg backend for headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import ASSETS_DIR, METRICS_PATH, MODELS_DIR
from src.utils import (
    ensure_directory,
    load_artifact,
    load_json,
    save_artifact,
    save_json,
    setup_logger,
)

logger = setup_logger("evaluate")


def evaluate_model(
    model_name: str,
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates a single fitted model on an unseen holdout test set.
    
    Args:
        model_name: Name of the model algorithm.
        model: Fitted scikit-learn estimator.
        X_test: Test feature matrix.
        y_test: True test labels (0=REAL, 1=FAKE).
        
    Returns:
        Dictionary containing metric values, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)
    
    # Calculate probabilities for ROC-AUC if supported
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            auc = 0.5
    else:
        y_proba = None
        auc = 0.5

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test,
        y_pred,
        target_names=["REAL", "FAKE"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics


def evaluate_all_models(
    trained_models: Dict[str, Dict[str, Any]],
    X_test: Any,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates all candidate models on the holdout test set.
    
    Args:
        trained_models: Dictionary containing trained model objects.
        X_test: Test feature matrix.
        y_test: Ground truth test labels.
        
    Returns:
        Dictionary mapping model names to their evaluation metrics dictionary.
    """
    logger.info("Evaluating all candidate models on holdout test set...")
    evaluations: Dict[str, Dict[str, Any]] = {}

    for name, entry in trained_models.items():
        model = entry["model"]
        eval_metrics = evaluate_model(name, model, X_test, y_test)
        eval_metrics["train_time"] = entry.get("train_time", 0.0)
        evaluations[name] = eval_metrics
        logger.info(
            f"[{name}] Acc: {eval_metrics['accuracy']:.4f} | "
            f"F1: {eval_metrics['f1_score']:.4f} | "
            f"AUC: {eval_metrics['roc_auc']:.4f}"
        )

    return evaluations


def build_comparison_dataframe(evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts model evaluation metrics into a sorted comparison DataFrame.
    
    Args:
        evaluations: Dictionary of evaluation metrics per model.
        
    Returns:
        pd.DataFrame sorted by F1-Score and ROC-AUC descending.
    """
    records = []
    for name, data in evaluations.items():
        records.append({
            "Model": name,
            "Accuracy": data["accuracy"],
            "Precision": data["precision"],
            "Recall": data["recall"],
            "F1-Score": data["f1_score"],
            "ROC-AUC": data["roc_auc"],
            "Fit Time (s)": round(data.get("train_time", 0.0), 4),
        })

    df = pd.DataFrame(records)
    # Sort by F1-Score, then ROC-AUC, then Accuracy
    df = df.sort_values(by=["F1-Score", "ROC-AUC", "Accuracy"], ascending=False).reset_index(
        drop=True
    )
    return df


def select_best_model(
    evaluations: Dict[str, Dict[str, Any]],
    trained_models: Dict[str, Dict[str, Any]],
) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Selects the best performing model using F1-score as primary criteria,
    with ROC-AUC and Accuracy as tiebreakers.
    
    Args:
        evaluations: Evaluation metrics dictionary.
        trained_models: Dictionary containing fitted model objects.
        
    Returns:
        Tuple of (best_model_name, best_model_object, best_model_metrics).
    """
    comp_df = build_comparison_dataframe(evaluations)
    best_name = str(comp_df.iloc[0]["Model"])
    best_model = trained_models[best_name]["model"]
    best_metrics = evaluations[best_name]

    logger.info(
        f"Model Selection Result: Best Model is [{best_name}] "
        f"(F1: {best_metrics['f1_score']}, Acc: {best_metrics['accuracy']}, AUC: {best_metrics['roc_auc']})"
    )
    return best_name, best_model, best_metrics


def plot_confusion_matrices(
    evaluations: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Renders high-resolution confusion matrix heatmaps for all models side-by-side.
    
    Args:
        evaluations: Dictionary containing confusion matrices.
        output_path: Destination path (defaults to assets/confusion_matrices.png).
        
    Returns:
        Path to saved figure.
    """
    dest_path = output_path if output_path else (ASSETS_DIR / "confusion_matrices.png")
    ensure_directory(dest_path)

    n_models = len(evaluations)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), dpi=150)
    if n_models == 1:
        axes = [axes]

    labels = ["REAL", "FAKE"]

    for ax, (name, metrics) in zip(axes, evaluations.items()):
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"{name}\n(Acc: {metrics['accuracy']:.2f} | F1: {metrics['f1_score']:.2f})", fontsize=11, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    fig.savefig(dest_path)
    plt.close(fig)
    logger.info(f"Saved confusion matrices heatmap to: {dest_path}")
    return dest_path


def plot_model_comparison(
    comp_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generates a grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models.
    
    Args:
        comp_df: Comparison DataFrame.
        output_path: Destination path (defaults to assets/model_comparison.png).
        
    Returns:
        Path to saved figure.
    """
    dest_path = output_path if output_path else (ASSETS_DIR / "model_comparison.png")
    ensure_directory(dest_path)

    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    melted_df = comp_df.melt(
        id_vars=["Model"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=melted_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette="crest",
        ax=ax,
    )
    ax.set_ylim(0, 1.15)
    ax.set_title("Candidate Model Performance Comparison", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.legend(title="Model", loc="upper right")

    # Annotate bar values
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(
                f"{val:.2f}",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="bottom",
                fontsize=8.5,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    fig.savefig(dest_path)
    plt.close(fig)
    logger.info(f"Saved model comparison chart to: {dest_path}")
    return dest_path


def save_model_artifacts(
    best_model_name: str,
    best_model: Any,
    feature_extractor: Any,
    best_metrics: Dict[str, Any],
    models_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Serializes the best trained model, feature extractor, and execution metadata to disk.
    
    Args:
        best_model_name: Name of the winning algorithm.
        best_model: Trained model estimator.
        feature_extractor: Fitted TextFeatureExtractor.
        best_metrics: Dictionary of test metrics.
        models_dir: Destination directory (defaults to MODELS_DIR).
        
    Returns:
        Dictionary of saved artifact file paths.
    """
    target_dir = models_dir if models_dir else MODELS_DIR
    ensure_directory(target_dir)

    model_path = target_dir / "best_model.joblib"
    vectorizer_path = target_dir / "tfidf_vectorizer.joblib"
    metadata_path = target_dir / "model_metrics.json"

    # Save model estimator
    save_artifact(best_model, model_path)

    # Save feature extractor
    feature_extractor.save(vectorizer_path)

    # Compile comprehensive production metadata
    metadata = {
        "best_model_name": best_model_name,
        "classes": ["REAL", "FAKE"],
        "metrics": {
            "accuracy": best_metrics.get("accuracy"),
            "precision": best_metrics.get("precision"),
            "recall": best_metrics.get("recall"),
            "f1_score": best_metrics.get("f1_score"),
            "roc_auc": best_metrics.get("roc_auc"),
            "confusion_matrix": best_metrics.get("confusion_matrix"),
        },
        "vocabulary_size": len(feature_extractor.get_feature_names()),
        "hyperparameters": {
            k: str(v) for k, v in getattr(best_model, "get_params", lambda: {})().items()
        },
    }
    save_json(metadata, metadata_path)

    logger.info(f"Model artifacts saved successfully in: {target_dir}")
    return {
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "metadata_path": metadata_path,
    }


def load_model_artifacts(
    models_dir: Optional[Path] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Loads serialized model, vectorizer, and metadata from disk.
    
    Args:
        models_dir: Source directory (defaults to MODELS_DIR).
        
    Returns:
        Tuple of (model, feature_extractor, metadata).
    """
    from src.feature_extractor import TextFeatureExtractor

    target_dir = models_dir if models_dir else MODELS_DIR
    model_path = target_dir / "best_model.joblib"
    vectorizer_path = target_dir / "tfidf_vectorizer.joblib"
    metadata_path = target_dir / "model_metrics.json"

    model = load_artifact(model_path)
    extractor = TextFeatureExtractor.load(vectorizer_path)
    metadata = load_json(metadata_path)

    logger.info(f"Loaded model artifacts for: {metadata.get('best_model_name')}")
    return model, extractor, metadata


def run_evaluation_pipeline(
    data_path: Optional[Path] = None,
    save_artifacts: bool = True,
) -> Tuple[str, Any, Dict[str, Any], pd.DataFrame]:
    """
    Executes training, evaluation, comparison plotting, and best model selection.
    
    Args:
        data_path: Optional custom dataset path.
        save_artifacts: Whether to serialize winning model & metadata to disk.
        
    Returns:
        Tuple of (best_model_name, best_model, best_metrics, comparison_dataframe).
    """
    from src.train import run_training_pipeline

    logger.info("Executing end-to-end training and evaluation pipeline...")
    trained_models, feature_extractor, X_test, y_test = run_training_pipeline(data_path=data_path)

    evaluations = evaluate_all_models(trained_models, X_test, y_test)
    comp_df = build_comparison_dataframe(evaluations)
    best_name, best_model, best_metrics = select_best_model(evaluations, trained_models)

    # Render visualizations
    cm_plot = plot_confusion_matrices(evaluations)
    comp_plot = plot_model_comparison(comp_df)

    saved_paths: Dict[str, Path] = {}
    if save_artifacts:
        saved_paths = save_model_artifacts(best_name, best_model, feature_extractor, best_metrics)

    print("\n" + "=" * 75)
    print("                    MODEL EVALUATION & BENCHMARK REPORT")
    print("=" * 75)
    print(comp_df.to_string(index=False))
    print("-" * 75)
    print(f"[BEST MODEL SELECTED]: {best_name}")
    print(f"  Test Accuracy: {best_metrics['accuracy'] * 100:.2f}%")
    print(f"  Test F1-Score: {best_metrics['f1_score'] * 100:.2f}%")
    print(f"  Test ROC-AUC:  {best_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix: {best_metrics['confusion_matrix']}")
    print(f"[CHARTS SAVED]:")
    print(f"  - Confusion Matrices: {cm_plot}")
    print(f"  - Performance Chart:  {comp_plot}")
    if save_artifacts:
        print(f"[SERIALIZED ARTIFACTS]:")
        for k, p in saved_paths.items():
            print(f"  - {k}: {p}")
    print("=" * 75 + "\n")

    return best_name, best_model, best_metrics, comp_df


def main() -> None:
    """CLI runner for model evaluation and artifact serialization."""
    run_evaluation_pipeline(save_artifacts=True)


if __name__ == "__main__":
    main()


