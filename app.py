"""
AI-Powered Fake News Detection System — Web Application.

A professional, portfolio-grade web application built with Streamlit, Scikit-Learn,
and Natural Language Processing.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ASSETS_DIR, SYSTEM_DISCLAIMER
from src.predictor import FakeNewsPredictor

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean, modern aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .result-card-real {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.1);
    }
    .result-card-fake {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.1);
    }
    .keyword-pill {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        padding: 0.3rem 0.75rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-container {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.9rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading machine learning pipeline...")
def get_predictor() -> FakeNewsPredictor:
    """Cached loader for the predictor engine."""
    return FakeNewsPredictor()


# Sample Articles for Instant Testing
SAMPLE_REAL_NEWS = (
    "WASHINGTON (Reuters) - The Federal Reserve kept its benchmark interest rate "
    "unchanged on Wednesday following a two-day monetary policy meeting. Policymakers noted "
    "that while inflation has eased notably over the past year, it remains somewhat elevated "
    "above the central bank's two percent target. Economic activity has continued to expand "
    "at a solid pace, supported by resilient consumer spending and a stable labor market."
)

SAMPLE_FAKE_NEWS = (
    "SHOCKING LEAK: An anonymous whistleblower claiming to be an insider at a covert "
    "underground facility has released explosive documents proving that municipal tap water is "
    "secretly laced with bio-frequency chemicals designed to control human brainwaves and force "
    "civilian obedience. Mainstream scientists have refused to cover this explosive revelation."
)


def initialize_session_state() -> None:
    """Manages input text and prediction state across reruns."""
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None


def set_input(text: str) -> None:
    st.session_state.input_text = text
    st.session_state.prediction_result = None


def clear_input() -> None:
    st.session_state.input_text = ""
    st.session_state.prediction_result = None


def main() -> None:
    initialize_session_state()

    try:
        predictor = get_predictor()
        metadata = predictor.metadata
    except Exception as e:
        st.error(
            f"⚠️ Could not load model artifacts. Please run training pipeline first.\nDetails: {e}"
        )
        st.stop()

    # =========================================================================
    # SIDEBAR: Overview & System Architecture
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ System Information")
        st.markdown(
            "An end-to-end NLP system that analyzes linguistic signatures, sensationalism, "
            "and vocabulary distributions to estimate news credibility."
        )

        st.subheader("🤖 Active Champion Model")
        st.info(f"**Model:** {predictor.model_name}")

        test_metrics = metadata.get("metrics", {})
        if test_metrics:
            st.markdown("### 📊 Test Evaluation")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Accuracy", f"{test_metrics.get('accuracy', 0)*100:.1f}%")
                st.metric("Recall", f"{test_metrics.get('recall', 0)*100:.1f}%")
            with col_m2:
                st.metric("F1-Score", f"{test_metrics.get('f1_score', 0)*100:.1f}%")
                st.metric("Precision", f"{test_metrics.get('precision', 0)*100:.1f}%")

        st.subheader("📚 Feature Space")
        st.markdown(
            f"""
            - **Representation:** TF-IDF (Unigrams & Bigrams)
            - **Vocabulary:** {metadata.get('vocabulary_size', 0)} terms
            - **Sublinear TF:** Enabled ($1 + \\log(tf)$)
            - **Calibration:** Enabled (Platt Scaling)
            """
        )

        with st.expander("🔍 Top Model Vocabulary"):
            from src.explainability import get_global_top_features
            global_feats = get_global_top_features(predictor.model, predictor.extractor, top_n=5)
            st.write("**Top Credible Markers:**")
            for t, w in global_feats["indicative_of_real"]:
                st.write(f"- `{t}` ({w:.2f})")
            st.write("**Top Fabricated Markers:**")
            for t, w in global_feats["indicative_of_fake"]:
                st.write(f"- `{t}` ({w:.2f})")

        st.markdown("---")
        st.warning(SYSTEM_DISCLAIMER)

    # =========================================================================
    # MAIN APPLICATION TABS
    # =========================================================================
    st.markdown('<div class="main-title">📰 AI Fake News Detection System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detect potentially misleading or fabricated news using Natural Language Processing and Machine Learning.</div>',
        unsafe_allow_html=True,
    )

    tab_analyze, tab_benchmarks, tab_eda, tab_methodology = st.tabs([
        "🔍 News Credibility Analyzer",
        "📊 Model Benchmarks & Comparison",
        "📈 Exploratory Data Analysis (EDA)",
        "ℹ️ Architecture & Methodology",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: News Credibility Analyzer
    # -------------------------------------------------------------------------
    with tab_analyze:
        st.write("### Analyze Article Credibility")
        st.write("Enter or paste a news headline, paragraph, or article to predict credibility:")

        # Quick Load Buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1.5, 3])
        with col_btn1:
            if st.button("📗 Load REAL News Sample", use_container_width=True):
                set_input(SAMPLE_REAL_NEWS)
                st.rerun()
        with col_btn2:
            if st.button("📕 Load FAKE News Sample", use_container_width=True):
                set_input(SAMPLE_FAKE_NEWS)
                st.rerun()
        with col_btn3:
            if st.button("🧹 Clear Input", use_container_width=True):
                clear_input()
                st.rerun()

        # Text input area
        user_text = st.text_area(
            label="News Text Input:",
            value=st.session_state.input_text,
            height=180,
            placeholder="Paste news headline or article body here...",
            label_visibility="collapsed",
        )

        col_run1, col_run2 = st.columns([2, 5])
        with col_run1:
            analyze_clicked = st.button("🔍 Analyze News", type="primary", use_container_width=True)

        if analyze_clicked:
            if not user_text.strip():
                st.error("⚠️ Please enter or paste text to analyze.")
            else:
                with st.spinner("Extracting TF-IDF n-grams and evaluating linguistic indicators..."):
                    result = predictor.predict(user_text)
                    st.session_state.prediction_result = result

        # Display Results
        if st.session_state.prediction_result:
            res = st.session_state.prediction_result

            if not res.get("is_valid", False):
                st.error(f"❌ {res.get('error_message')}")
            else:
                label = res["label"]
                conf = res["confidence"]
                p_real = res["probability_real"]
                p_fake = res["probability_fake"]
                metrics = res["metrics"]
                top_kw = res["top_keywords"]
                explanation = res.get("explanation", {})
                sentiment_info = res.get("sentiment_analysis", {})

                st.markdown("---")
                st.subheader("🎯 Prediction Result")

                # Credibility Card
                if label == "REAL":
                    st.markdown(
                        f"""
                        <div class="result-card-real">
                            <h2 style="color: #065F46; margin: 0;">✔ PREDICTION: REAL NEWS</h2>
                            <p style="color: #047857; margin-top: 0.3rem; font-size: 1.05rem;">
                                The text exhibits neutral journalistic framing, accredited reporting structure, and vocabulary 
                                typical of verified news organizations.
                            </p>
                            <h3 style="color: #065F46; margin-top: 0.5rem;">Confidence Score: {conf}%</h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-fake">
                            <h2 style="color: #991B1B; margin: 0;">⚠️ PREDICTION: FAKE / MISLEADING NEWS</h2>
                            <p style="color: #B91C1C; margin-top: 0.3rem; font-size: 1.05rem;">
                                The text exhibits sensationalist phrasing, unverified claims, or emotional rhetoric commonly 
                                found in misleading articles.
                            </p>
                            <h3 style="color: #991B1B; margin-top: 0.5rem;">Confidence Score: {conf}%</h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Class Probabilities
                st.write("**Class Probability Distribution:**")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.write(f"Credible / Authentic (REAL): **{p_real}%**")
                    st.progress(p_real / 100.0)
                with col_p2:
                    st.write(f"Fabricated / Misleading (FAKE): **{p_fake}%**")
                    st.progress(p_fake / 100.0)

                # Rhetorical & Stylistic Analysis
                st.markdown("### 📝 Linguistic & Rhetorical Profile")
                col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                with col_n1:
                    st.metric("Word Count", metrics.get("word_count", 0))
                with col_n2:
                    st.metric("Character Count", metrics.get("character_count", 0))
                with col_n3:
                    st.metric("Emotional Sentiment", sentiment_info.get("sentiment_label", "Neutral"))
                with col_n4:
                    st.metric("Sensationalism Index", f"{sentiment_info.get('sensationalism_score', 0)}%")

                # TF-IDF Keywords
                st.markdown("### 🔑 Key Indicative Terms (TF-IDF)")
                st.caption(
                    "Top terms identified by Term Frequency-Inverse Document Frequency weighting for this article."
                )
                if top_kw:
                    pills = " ".join([f'<span class="keyword-pill">{k} ({w:.3f})</span>' for k, w in top_kw])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.info("No dominant vocabulary terms detected.")

                # Explainability Section
                st.markdown("### 💡 Model Reasoning & Explainability")
                summary_text = explanation.get("summary_text", "")
                if summary_text:
                    st.markdown(summary_text)

                fake_signals = explanation.get("fake_signals", [])
                real_signals = explanation.get("real_signals", [])

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.write("🟢 **Authentic / Journalistic Indicators Found:**")
                    if real_signals:
                        for term, score in real_signals:
                            st.write(f"- `{term}` (impact: {score:.3f})")
                    else:
                        st.caption("No significant authentic indicators detected.")
                with col_s2:
                    st.write("🔴 **Sensationalist / Misleading Indicators Found:**")
                    if fake_signals:
                        for term, score in fake_signals:
                            st.write(f"- `{term}` (impact: {score:.3f})")
                    else:
                        st.caption("No significant sensationalist indicators detected.")

                st.markdown("---")
                st.info(res.get("disclaimer", SYSTEM_DISCLAIMER))

    # -------------------------------------------------------------------------
    # TAB 2: Model Benchmarks & Comparison
    # -------------------------------------------------------------------------
    with tab_benchmarks:
        st.header("📊 Model Evaluation & Benchmarking")
        st.markdown(
            "Three supervised machine learning architectures were trained and evaluated on the holdout test set "
            "using stratified sampling to prevent class imbalance skew."
        )

        comp_chart = ASSETS_DIR / "model_comparison.png"
        cm_chart = ASSETS_DIR / "confusion_matrices.png"

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            st.subheader("Performance Comparison")
            if comp_chart.exists():
                st.image(str(comp_chart), caption="Accuracy, Precision, Recall, and F1 Comparison", use_container_width=True)
            else:
                st.info("Performance chart will appear once model evaluation is executed.")

        with col_b2:
            st.subheader("Confusion Matrices")
            if cm_chart.exists():
                st.image(str(cm_chart), caption="Confusion Matrices across all Candidate Models", use_container_width=True)
            else:
                st.info("Confusion matrices will appear once model evaluation is executed.")

        st.subheader("Candidate Architecture Overview")
        st.markdown(
            """
            | Architecture | Theoretical Basis | Strengths for Fake News | Probability Calibration |
            | :--- | :--- | :--- | :--- |
            | **Logistic Regression** | Linear model optimizing log-likelihood | Highly interpretable, well-calibrated native probabilities | Built-in via sigmoid function |
            | **Multinomial Naive Bayes** | Generative probabilistic model using Bayes' theorem | High speed, excels with sparse high-dimensional bag-of-words | Posterior probability computation |
            | **Linear SVM** | Discriminative maximum-margin hyperplane separator | Strong boundary separation in high-dimensional text | Platt scaling (`CalibratedClassifierCV`) |
            """
        )

    # -------------------------------------------------------------------------
    # TAB 3: Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------------------
    with tab_eda:
        st.header("📈 Exploratory Data Analysis (EDA)")
        st.markdown(
            "Comprehensive statistical and linguistic profiling of the dataset prior to feature engineering."
        )

        eda_class_plot = ASSETS_DIR / "eda_class_distribution.png"
        eda_len_plot = ASSETS_DIR / "eda_word_count_distribution.png"

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if eda_class_plot.exists():
                st.image(str(eda_class_plot), caption="Class Balance (REAL vs FAKE)", use_container_width=True)
        with col_e2:
            if eda_len_plot.exists():
                st.image(str(eda_len_plot), caption="Word Count Spread & Distribution", use_container_width=True)

        st.subheader("Key Empirical Observations")
        st.markdown(
            """
            1. **Class Distribution**: Balanced 50% REAL and 50% FAKE articles, ensuring the classifier is not biased toward a majority class.
            2. **Stylistic Markers**: Fabricated news articles frequently exhibit dramatic modal verbs, clickbait superlatives, and anonymous attributions (*"leaked"*, *"claims to"*, *"secret"*).
            3. **Journalistic Form**: Verified news follows structured journalistic framing with accredited citations (*"reuters"*, *"officials noted that"*).
            """
        )

    # -------------------------------------------------------------------------
    # TAB 4: Architecture & Methodology
    # -------------------------------------------------------------------------
    with tab_methodology:
        st.header("ℹ️ System Architecture & Methodology")
        st.markdown(
            """
            ### End-to-End Machine Learning Pipeline
            ```
            Dataset (Kaggle / ISOT)
                │
                ▼
            Data Ingestion & Cleaning (Missing values, duplicates, label standardization)
                │
                ▼
            Exploratory Data Analysis (Linguistic distributions, class balance)
                │
                ▼
            NLP Preprocessing (Contractions, URL/HTML removal, punctuation, casing)
                │
                ▼
            Stratified Train / Test Split (80% Train, 20% Holdout Test)
                │
                ▼
            TF-IDF Feature Engineering (Unigrams + Bigrams, Sublinear TF scaling)
                │
                ▼
            Multi-Model Training (Logistic Regression, Naive Bayes, Linear SVM)
                │
                ▼
            Model Evaluation & Selection (F1-score & ROC-AUC champion selection)
                │
                ▼
            Model Serialization (joblib + metadata JSON)
                │
                ▼
            Streamlit Web Application & Explainability Engine
            ```

            ### Why TF-IDF with Sublinear Scaling?
            1. **Sublinear Term Frequency**: Standard TF counts word occurrences linearly. However, an article repeating a sensational word 20 times is rarely 20 times more relevant than one mentioning it once. Using $1 + \\log(tf)$ prevents keyword stuffing from distorting predictions.
            2. **Transparency**: Unlike opaque neural embeddings, every dimension corresponds to a human-readable n-gram, enabling full explainability.

            ### Ethical Considerations & Limitations
            - **No Ground Truth Guarantee**: Machine learning models recognize statistical patterns, not physical truth.
            - **Domain Shift**: Models trained on political news may exhibit reduced accuracy on financial or local sports news.
            - **Evolving Misinformation**: Adversarial actors continually modify phrasing to evade pattern detection.
            """
        )


if __name__ == "__main__":
    main()
