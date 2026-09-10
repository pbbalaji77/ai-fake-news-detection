"""
Explainability and Model Interpretability Module.

Provides both Global Explainability (model-wide feature importance and class associations)
and Local Explainability (document-specific token contributions and plain-English rationales).

Design Rationale:
Black-box AI systems are risky for content moderation and credibility assessment.
By inspecting linear weights and Naive Bayes log-odds ratios combined with TF-IDF weights,
we reveal exactly which words drive a classification, making the AI transparent,
auditable, and accountable without claiming infallible truth.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import LABEL_FAKE, LABEL_REAL, SYSTEM_DISCLAIMER
from src.feature_extractor import TextFeatureExtractor
from src.preprocessing import clean_text
from src.utils import setup_logger

logger = setup_logger("explainability")


def get_model_feature_weights(
    model: Any, feature_extractor: TextFeatureExtractor
) -> np.ndarray:
    """
    Extracts directional feature weights from linear or Naive Bayes models.
    Positive weights indicate association with FAKE news (class 1).
    Negative weights indicate association with REAL news (class 0).
    
    Args:
        model: Fitted scikit-learn estimator or calibrated wrapper.
        feature_extractor: Fitted TextFeatureExtractor.
        
    Returns:
        1D numpy array of feature weights aligned with vectorizer vocabulary.
    """
    n_features = len(feature_extractor.get_feature_names())

    # 1. Logistic Regression
    if hasattr(model, "coef_"):
        return model.coef_[0]

    # 2. Multinomial Naive Bayes (Log-Odds Ratio)
    elif hasattr(model, "feature_log_prob_"):
        # log P(word | FAKE) - log P(word | REAL)
        log_prob_fake = model.feature_log_prob_[LABEL_FAKE]
        log_prob_real = model.feature_log_prob_[LABEL_REAL]
        return log_prob_fake - log_prob_real

    # 3. CalibratedClassifierCV (Linear SVM wrapper)
    elif hasattr(model, "calibrated_classifiers_"):
        weights_list = []
        for calibrated in model.calibrated_classifiers_:
            base_est = getattr(calibrated, "estimator", None)
            if base_est and hasattr(base_est, "coef_"):
                weights_list.append(base_est.coef_[0])
        if weights_list:
            return np.mean(weights_list, axis=0)

    # Fallback to neutral zero weights if architecture is uninterpretable
    logger.warning("Model architecture does not expose direct linear weights. Using zeros.")
    return np.zeros(n_features)


def get_global_top_features(
    model: Any,
    feature_extractor: TextFeatureExtractor,
    top_n: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Computes the most influential words globally across the training dataset.
    
    Args:
        model: Fitted model estimator.
        feature_extractor: Fitted vectorizer.
        top_n: Number of top features to return per class.
        
    Returns:
        Dictionary with 'indicative_of_fake' and 'indicative_of_real' term lists.
    """
    feature_names = feature_extractor.get_feature_names()
    weights = get_model_feature_weights(model, feature_extractor)

    # Sort descending for FAKE (highest positive weights)
    fake_idx = np.argsort(weights)[::-1][:top_n]
    fake_terms = [
        (str(feature_names[i]), round(float(weights[i]), 4))
        for i in fake_idx
        if weights[i] > 0
    ]

    # Sort ascending for REAL (lowest negative weights)
    real_idx = np.argsort(weights)[:top_n]
    real_terms = [
        (str(feature_names[i]), round(float(abs(weights[i])), 4))
        for i in real_idx
        if weights[i] < 0
    ]

    return {
        "indicative_of_fake": fake_terms,
        "indicative_of_real": real_terms,
    }


def explain_prediction(
    text: str,
    model: Any,
    feature_extractor: TextFeatureExtractor,
    predicted_label: str,
    confidence: float,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Generates local, document-level interpretability for an individual prediction.
    Calculates token contribution scores: TF-IDF weight * Model Directional Weight.
    
    Args:
        text: Input raw or clean news text.
        model: Trained model estimator.
        feature_extractor: Fitted vectorizer.
        predicted_label: 'REAL' or 'FAKE'.
        confidence: Confidence score percentage.
        top_n: Number of influential terms to extract.
        
    Returns:
        Dictionary with token contributions, plain-English summary, and guidance.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "token_contributions": [],
            "fake_signals": [],
            "real_signals": [],
            "summary_text": "Input contains insufficient text for feature-level explanation.",
            "disclaimer": SYSTEM_DISCLAIMER,
        }

    # Transform document to TF-IDF vector
    tfidf_vec = feature_extractor.transform([cleaned])
    feature_names = feature_extractor.get_feature_names()
    model_weights = get_model_feature_weights(model, feature_extractor)

    indices = tfidf_vec.indices
    tfidf_values = tfidf_vec.data

    if len(indices) == 0:
        return {
            "token_contributions": [],
            "fake_signals": [],
            "real_signals": [],
            "summary_text": (
                "None of the words in this text appeared in the training vocabulary. "
                "The prediction relies on prior class distributions."
            ),
            "disclaimer": SYSTEM_DISCLAIMER,
        }

    # Contribution score = TF-IDF * Model Weight
    contributions = tfidf_values * model_weights[indices]

    # Partition terms into signals towards FAKE vs REAL
    fake_signals = []
    real_signals = []

    for idx, term_idx in enumerate(indices):
        term = str(feature_names[term_idx])
        contrib = float(contributions[idx])
        if contrib > 0:
            fake_signals.append((term, round(contrib, 4)))
        elif contrib < 0:
            real_signals.append((term, round(abs(contrib), 4)))

    # Sort descending by impact magnitude
    fake_signals.sort(key=lambda x: x[1], reverse=True)
    real_signals.sort(key=lambda x: x[1], reverse=True)

    top_fake = fake_signals[:top_n]
    top_real = real_signals[:top_n]

    # Generate plain-English explanation for recruiters and non-technical users
    if predicted_label == "FAKE":
        if top_fake:
            top_words_str = ", ".join(f"'{term}'" for term, _ in top_fake[:3])
            summary_text = (
                f"The model classified this article as **FAKE ({confidence}%)** primarily due to the presence of "
                f"sensationalist or unverified rhetorical patterns including {top_words_str}. In training data, "
                f"these vocabulary markers were strongly correlated with fabricated or conspiratorial reports."
            )
        else:
            summary_text = (
                f"The model classified this article as **FAKE ({confidence}%)** based on overall stylistic and syntactic "
                "patterns matching unverified news."
            )
    else:
        if top_real:
            top_words_str = ", ".join(f"'{term}'" for term, _ in top_real[:3])
            summary_text = (
                f"The model classified this article as **REAL ({confidence}%)** due to vocabulary characteristic of "
                f"factual journalistic reporting such as {top_words_str}. These terms regularly appear in verified news "
                "reporting from accredited agencies."
            )
        else:
            summary_text = (
                f"The model classified this article as **REAL ({confidence}%)** based on an absence of sensationalist "
                "cues and tone consistent with factual news."
            )

    return {
        "fake_signals": top_fake,
        "real_signals": top_real,
        "summary_text": summary_text,
        "disclaimer": SYSTEM_DISCLAIMER,
    }
