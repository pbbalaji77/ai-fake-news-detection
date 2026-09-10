"""
Inference Predictor Engine for Fake News Detection.

Decouples machine learning inference, preprocessing, and confidence calculation
from user interface layers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import (
    BEST_MODEL_PATH,
    LABEL_FAKE,
    LABEL_MAP,
    LABEL_REAL,
    METRICS_PATH,
    MODELS_DIR,
    SYSTEM_DISCLAIMER,
    TOP_N_EXPLANATION_FEATURES,
    VECTORIZER_PATH,
)
from src.evaluate import load_model_artifacts
from src.feature_extractor import TextFeatureExtractor
from src.preprocessing import clean_text
from src.utils import compute_basic_text_metrics, setup_logger

logger = setup_logger("predictor")


class FakeNewsPredictor:
    """
    Production inference engine for predicting news article credibility.
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        """
        Initializes the predictor by loading serialized model artifacts.
        
        Args:
            models_dir: Optional custom path to directory containing model artifacts.
        """
        self.models_dir = models_dir if models_dir else MODELS_DIR
        self.model, self.extractor, self.metadata = load_model_artifacts(self.models_dir)
        self.model_name = self.metadata.get("best_model_name", type(self.model).__name__)
        logger.info(f"FakeNewsPredictor initialized with model: [{self.model_name}]")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Processes an input news headline/article, extracts features, and outputs
        binary classification, calibrated confidence, text statistics, and key indicative terms.
        
        Args:
            text: Raw input news text or headline.
            
        Returns:
            Dictionary containing prediction results, probabilities, metrics, and keywords.
        """
        # 1. Compute raw text descriptive statistics
        text_metrics = compute_basic_text_metrics(text)

        # 2. Input validation
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "is_valid": False,
                "error_message": "Input text is empty. Please enter or paste a news article or headline.",
                "metrics": text_metrics,
            }

        # 3. Text cleaning and normalization
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "is_valid": False,
                "error_message": (
                    "Input text does not contain sufficient alphabetic words for analysis "
                    "(e.g., contains only numbers or punctuation)."
                ),
                "metrics": text_metrics,
            }

        # 4. Feature extraction
        X = self.extractor.transform([cleaned])

        # 5. Prediction & Confidence Estimation
        label_id = int(self.model.predict(X)[0])
        label_str = LABEL_MAP.get(label_id, "UNKNOWN")

        # Class probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            prob_real = float(probs[LABEL_REAL])
            prob_fake = float(probs[LABEL_FAKE])
        else:
            # Fallback for uncalibrated models
            prob_real = 1.0 if label_id == LABEL_REAL else 0.0
            prob_fake = 1.0 if label_id == LABEL_FAKE else 0.0

        confidence = round(max(prob_real, prob_fake) * 100, 2)

        # 6. Extract dominant TF-IDF keywords for the specific article
        top_keywords = self.extractor.get_top_features_for_text(
            cleaned, top_n=TOP_N_EXPLANATION_FEATURES
        )

        # 7. Generate local interpretability explanation
        from src.explainability import explain_prediction
        explanation = explain_prediction(
            text=cleaned,
            model=self.model,
            feature_extractor=self.extractor,
            predicted_label=label_str,
            confidence=confidence,
            top_n=5,
        )

        # 8. Compute rhetorical sensationalism and sentiment indicators
        from src.utils import compute_sentiment_and_sensationalism
        sentiment_analysis = compute_sentiment_and_sensationalism(text)

        return {
            "is_valid": True,
            "label": label_str,
            "label_id": label_id,
            "confidence": confidence,
            "probability_real": round(prob_real * 100, 2),
            "probability_fake": round(prob_fake * 100, 2),
            "model_name": self.model_name,
            "metrics": text_metrics,
            "sentiment_analysis": sentiment_analysis,
            "top_keywords": top_keywords,
            "explanation": explanation,
            "cleaned_text": cleaned,
            "disclaimer": SYSTEM_DISCLAIMER,
        }
