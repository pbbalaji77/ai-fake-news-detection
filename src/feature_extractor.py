"""
TF-IDF Feature Extraction Module.

Transforms clean text into numerical TF-IDF (Term Frequency - Inverse Document Frequency)
feature representations with configurable n-grams, sublinear scaling, and document frequency cutoffs.

Why TF-IDF is chosen for this project:
1. Interpretability: Unlike dense neural embeddings, each TF-IDF dimension directly maps
   to a specific word or bigram, making model predictions transparent and explainable.
2. Computational Efficiency: Sparse matrix representations enable ultra-fast model training
   and sub-millisecond inference latency on standard CPUs.
3. Frequency Damping (sublinear_tf): Replaces raw frequency tf with 1 + log(tf), preventing
   clickbait articles repeating sensational words 30 times from distorting the feature space.
4. Document-Level Discernment: Downweights ubiquitous terms (via IDF) while prioritizing
   distinctive discriminative markers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
    VECTORIZER_PATH,
)
from src.preprocessing import clean_text
from src.utils import load_artifact, save_artifact, setup_logger

logger = setup_logger("feature_extractor")


class TextFeatureExtractor:
    """
    Encapsulates TF-IDF vectorization with configurable parameters and
    document-level feature inspection capabilities.
    """

    def __init__(
        self,
        max_features: Optional[int] = TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
        min_df: Union[int, float] = TFIDF_MIN_DF,
        max_df: Union[int, float] = TFIDF_MAX_DF,
        sublinear_tf: bool = TFIDF_SUBLINEAR_TF,
        stop_words: Optional[str] = "english",
    ) -> None:
        """
        Initializes the TF-IDF vectorizer configuration.
        
        Args:
            max_features: Maximum number of top terms ordered by term frequency.
            ngram_range: Lower and upper boundary of range of n-values (e.g., (1, 2)).
            min_df: Minimum document frequency threshold (ignore terms below this).
            max_df: Maximum document frequency threshold (ignore ubiquitous terms).
            sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
            stop_words: Stopwords strategy ('english' or None).
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.stop_words = stop_words

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            stop_words=self.stop_words,
        )
        self.is_fitted: bool = False

    def fit(self, texts: List[str]) -> "TextFeatureExtractor":
        """
        Fits the vectorizer strictly on training texts to prevent data leakage.
        
        Args:
            texts: Iterable of training text strings.
            
        Returns:
            self
        """
        logger.info("Fitting TF-IDF vectorizer on training corpus...")
        # Adapt min_df dynamically if dataset has fewer documents than min_df
        if len(texts) < (self.min_df if isinstance(self.min_df, int) else 2):
            logger.warning(
                f"Corpus size ({len(texts)}) is smaller than min_df ({self.min_df}). Adjusting min_df=1."
            )
            self.vectorizer.set_params(min_df=1)

        self.vectorizer.fit(texts)
        self.is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"TF-IDF fitting complete. Vocabulary size: {vocab_size} features.")
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """
        Transforms texts into a sparse TF-IDF feature matrix.
        
        Args:
            texts: Iterable of text strings.
            
        Returns:
            scipy.sparse.csr_matrix of shape (n_samples, n_features).
            
        Raises:
            RuntimeError: If called before fitting.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot transform texts before vectorizer is fitted.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        """
        Fits vectorizer and returns transformed training matrix.
        
        Args:
            texts: Iterable of training text strings.
            
        Returns:
            scipy.sparse.csr_matrix of shape (n_samples, n_features).
        """
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> np.ndarray:
        """
        Returns an array of vocabulary feature terms.
        
        Returns:
            np.ndarray of feature strings.
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer is not fitted yet.")
        return self.vectorizer.get_feature_names_out()

    def get_top_features_for_text(
        self, text: str, top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extracts the highest-scoring TF-IDF terms for an individual input text.
        Essential for explainability and real-time Streamlit feature breakdown.
        
        Args:
            text: Input text string (raw or clean).
            top_n: Number of top weighted terms to return.
            
        Returns:
            List of (feature_name, tfidf_weight) tuples sorted descending by weight.
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before computing feature weights.")

        cleaned = clean_text(text)
        if not cleaned:
            return []

        tfidf_vec = self.transform([cleaned])
        feature_names = self.get_feature_names()

        # Extract non-zero coordinates
        non_zero_indices = tfidf_vec.indices
        weights = tfidf_vec.data

        if len(weights) == 0:
            return []

        # Sort non-zero indices by descending weight
        sorted_order = np.argsort(weights)[::-1][:top_n]
        top_terms = [
            (str(feature_names[non_zero_indices[idx]]), round(float(weights[idx]), 4))
            for idx in sorted_order
        ]
        return top_terms

    def save(self, file_path: Optional[Path] = None) -> Path:
        """
        Serializes the fitted vectorizer to disk.
        
        Args:
            file_path: Destination path (defaults to VECTORIZER_PATH).
            
        Returns:
            Path where vectorizer was saved.
        """
        dest_path = file_path if file_path else VECTORIZER_PATH
        logger.info(f"Saving TF-IDF vectorizer artifact to: {dest_path}")
        save_artifact(self, dest_path)
        return dest_path

    @classmethod
    def load(cls, file_path: Optional[Path] = None) -> "TextFeatureExtractor":
        """
        Loads a serialized TextFeatureExtractor artifact from disk.
        
        Args:
            file_path: Source path (defaults to VECTORIZER_PATH).
            
        Returns:
            Loaded TextFeatureExtractor instance.
        """
        src_path = file_path if file_path else VECTORIZER_PATH
        logger.info(f"Loading TF-IDF vectorizer artifact from: {src_path}")
        instance = load_artifact(src_path)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded artifact is not an instance of {cls.__name__}")
        return instance
