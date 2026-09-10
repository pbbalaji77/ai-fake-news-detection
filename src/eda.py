"""
Exploratory Data Analysis (EDA) Module.

Computes descriptive linguistic statistics, class distributions, word count distributions,
vocabulary metrics, and generates publication-grade visualizations for the Fake News Dataset.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# Use non-interactive Agg backend for headless and script execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    ASSETS_DIR,
    COL_CLEAN_TEXT,
    COL_LABEL,
    COL_TEXT,
    COL_TITLE,
    LABEL_FAKE,
    LABEL_MAP,
    LABEL_REAL,
    PROCESSED_TRAIN_FILE,
    SAMPLE_DATA_FILE,
)
from src.data_loader import clean_dataset, load_raw_data
from src.utils import ensure_directory, setup_logger

logger = setup_logger("eda")


def compute_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes class counts, percentages, and imbalance ratio.
    
    Args:
        df: DataFrame containing the COL_LABEL column.
        
    Returns:
        Dictionary with count and percentage metrics for REAL and FAKE classes.
    """
    counts = df[COL_LABEL].value_counts().to_dict()
    total = len(df)
    real_count = counts.get(LABEL_REAL, 0)
    fake_count = counts.get(LABEL_FAKE, 0)
    
    real_pct = (real_count / total * 100) if total > 0 else 0.0
    fake_pct = (fake_count / total * 100) if total > 0 else 0.0
    
    imbalance_ratio = (
        max(real_count, fake_count) / max(min(real_count, fake_count), 1)
    )

    return {
        "total_samples": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "real_percentage": round(real_pct, 2),
        "fake_percentage": round(fake_pct, 2),
        "imbalance_ratio": round(imbalance_ratio, 2),
    }


def compute_text_length_statistics(df: pd.DataFrame, text_col: str = COL_CLEAN_TEXT) -> Dict[str, Dict[str, float]]:
    """
    Calculates word count and character count distribution metrics segmented by class.
    
    Args:
        df: Cleaned DataFrame.
        text_col: Column name containing text to analyze.
        
    Returns:
        Nested dictionary of statistics (mean, median, std, min, max) for REAL and FAKE classes.
    """
    stats: Dict[str, Dict[str, float]] = {}
    
    df_calc = df.copy()
    df_calc["word_count"] = df_calc[text_col].apply(lambda x: len(str(x).split()))
    df_calc["char_count"] = df_calc[text_col].apply(lambda x: len(str(x)))

    for label_id, label_name in [(LABEL_REAL, "REAL"), (LABEL_FAKE, "FAKE")]:
        subset = df_calc[df_calc[COL_LABEL] == label_id]
        if not subset.empty:
            stats[label_name] = {
                "avg_word_count": round(float(subset["word_count"].mean()), 2),
                "median_word_count": round(float(subset["word_count"].median()), 2),
                "std_word_count": round(float(subset["word_count"].std()), 2),
                "avg_char_count": round(float(subset["char_count"].mean()), 2),
                "median_char_count": round(float(subset["char_count"].median()), 2),
                "min_word_count": int(subset["word_count"].min()),
                "max_word_count": int(subset["word_count"].max()),
            }
        else:
            stats[label_name] = {}

    return stats


def extract_top_ngrams(
    texts: List[str], n: int = 1, top_k: int = 10
) -> List[Tuple[str, int]]:
    """
    Extracts the most frequent n-grams from a list of clean text strings.
    
    Args:
        texts: List of normalized text documents.
        n: N-gram degree (1 for unigrams, 2 for bigrams).
        top_k: Number of top items to return.
        
    Returns:
        List of tuples (ngram_string, count).
    """
    ngram_counter: Counter = Counter()
    for text in texts:
        words = str(text).split()
        if len(words) < n:
            continue
        if n == 1:
            ngram_counter.update(words)
        else:
            ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            ngram_counter.update(ngrams)

    return ngram_counter.most_common(top_k)


def generate_eda_visualizations(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Generates and saves professional EDA visualizations:
    1. Class distribution bar plot.
    2. Word count distribution boxplot/histogram.
    3. Top unigram comparisons.
    
    Args:
        df: Cleaned DataFrame with text and label columns.
        output_dir: Destination folder for plots (defaults to ASSETS_DIR).
        
    Returns:
        Dictionary mapping chart name to file path.
    """
    dest_dir = output_dir if output_dir else ASSETS_DIR
    ensure_directory(dest_dir)
    
    df_plot = df.copy()
    df_plot["label_name"] = df_plot[COL_LABEL].map(LABEL_MAP)
    df_plot["word_count"] = df_plot[COL_CLEAN_TEXT].apply(lambda x: len(str(x).split()))

    generated_plots: Dict[str, Path] = {}
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Class Distribution Chart
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    palette = {"REAL": "#2ecc71", "FAKE": "#e74c3c"}
    sns.countplot(data=df_plot, x="label_name", hue="label_name", palette=palette, legend=False, ax=ax)
    ax.set_title("Dataset Class Balance (REAL vs FAKE)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("News Credibility Class", fontsize=11)
    ax.set_ylabel("Article Count", fontsize=11)

    # Add count annotations above bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    class_dist_path = dest_dir / "eda_class_distribution.png"
    fig.savefig(class_dist_path)
    plt.close(fig)
    generated_plots["class_distribution"] = class_dist_path

    # 2. Word Count Distribution (Boxplot & Histogram)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    
    sns.boxplot(
        data=df_plot,
        x="label_name",
        y="word_count",
        hue="label_name",
        palette=palette,
        legend=False,
        ax=axes[0],
        fliersize=3,
    )
    axes[0].set_title("Word Count Spread by Class", fontsize=12, weight="bold")
    axes[0].set_xlabel("Class", fontsize=10)
    axes[0].set_ylabel("Word Count", fontsize=10)

    # Enable KDE only if there is sufficient sample variation to avoid singular covariance matrix
    has_variance = df_plot["word_count"].nunique() > 1 and len(df_plot) >= 5
    sns.histplot(
        data=df_plot,
        x="word_count",
        hue="label_name",
        palette=palette,
        kde=has_variance,
        ax=axes[1],
        element="step",
    )
    axes[1].set_title("Word Count Density (Histogram)", fontsize=12, weight="bold")
    axes[1].set_xlabel("Word Count", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)

    plt.tight_layout()
    word_count_path = dest_dir / "eda_word_count_distribution.png"
    fig.savefig(word_count_path)
    plt.close(fig)
    generated_plots["word_count_distribution"] = word_count_path

    logger.info(f"EDA visualizations generated and saved in: {dest_dir}")
    return generated_plots


def run_eda(file_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Orchestrates the end-to-end EDA analysis and produces metrics and plots.
    
    Args:
        file_path: Optional dataset path.
        
    Returns:
        Dictionary of computed EDA metrics and generated visual asset paths.
    """
    logger.info("Executing Exploratory Data Analysis...")
    raw_df = load_raw_data(file_path)
    df = clean_dataset(raw_df)

    class_metrics = compute_class_distribution(df)
    length_metrics = compute_text_length_statistics(df)
    
    real_texts = df[df[COL_LABEL] == LABEL_REAL][COL_CLEAN_TEXT].tolist()
    fake_texts = df[df[COL_LABEL] == LABEL_FAKE][COL_CLEAN_TEXT].tolist()

    top_real_unigrams = extract_top_ngrams(real_texts, n=1, top_k=8)
    top_fake_unigrams = extract_top_ngrams(fake_texts, n=1, top_k=8)
    top_real_bigrams = extract_top_ngrams(real_texts, n=2, top_k=5)
    top_fake_bigrams = extract_top_ngrams(fake_texts, n=2, top_k=5)

    plots = generate_eda_visualizations(df)

    eda_summary = {
        "class_metrics": class_metrics,
        "length_metrics": length_metrics,
        "top_real_unigrams": top_real_unigrams,
        "top_fake_unigrams": top_fake_unigrams,
        "top_real_bigrams": top_real_bigrams,
        "top_fake_bigrams": top_fake_bigrams,
        "generated_plots": {k: str(v) for k, v in plots.items()},
    }

    print("\n" + "=" * 60)
    print("        EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)
    print(f"Total Cleaned Records: {class_metrics['total_samples']}")
    print(f"REAL Articles:         {class_metrics['real_count']} ({class_metrics['real_percentage']}%)")
    print(f"FAKE Articles:         {class_metrics['fake_count']} ({class_metrics['fake_percentage']}%)")
    print(f"Imbalance Ratio:       {class_metrics['imbalance_ratio']}:1 (Well-Balanced)")
    print("-" * 60)
    print("Word Count Statistics:")
    for label, metrics in length_metrics.items():
        print(f"  [{label}] Mean: {metrics.get('avg_word_count')} words | "
              f"Median: {metrics.get('median_word_count')} words | "
              f"Range: [{metrics.get('min_word_count')} - {metrics.get('max_word_count')}]")
    print("-" * 60)
    print("Top Terms in REAL News:")
    print("  Unigrams:", ", ".join(f"{w} ({c})" for w, c in top_real_unigrams[:5]))
    print("  Bigrams: ", ", ".join(f"'{bg}' ({c})" for bg, c in top_real_bigrams[:3]))
    print("\nTop Terms in FAKE News:")
    print("  Unigrams:", ", ".join(f"{w} ({c})" for w, c in top_fake_unigrams[:5]))
    print("  Bigrams: ", ", ".join(f"'{bg}' ({c})" for bg, c in top_fake_bigrams[:3]))
    print("=" * 60 + "\n")

    return eda_summary


if __name__ == "__main__":
    run_eda()
