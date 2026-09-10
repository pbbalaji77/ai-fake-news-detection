# 📰 AI-Powered Fake News Detection System

> A production-ready, interpretable Natural Language Processing and Machine Learning system that classifies news articles as **REAL** or **FAKE** using TF-IDF feature representations, multi-model evaluation, calibrated confidence scoring, and plain-English explainability.

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔗 Live Demo & Links
- **Live Streamlit Application**: [https://fake-news-detector.streamlit.app](https://fake-news-detector.streamlit.app) *(Replace with your deployed URL)*
- **GitHub Repository**: [https://github.com/your-username/ai-fake-news-detection-system](https://github.com/your-username/ai-fake-news-detection-system) *(Replace with your repository URL)*

---

## 📋 Table of Contents
1. [Project Title & Description](#-ai-powered-fake-news-detection-system)
2. [Live Demo & Links](#-live-demo--links)
3. [Project Overview](#-project-overview)
4. [Problem Statement](#-problem-statement)
5. [Objectives](#-objectives)
6. [Key Features](#-key-features)
7. [Technologies Used](#-technologies-used)
8. [NLP Concepts & Techniques](#-nlp-concepts--techniques)
9. [Machine Learning Concepts](#-machine-learning-concepts)
10. [System Architecture](#-system-architecture)
11. [Machine Learning Methodology](#-machine-learning-methodology)
12. [Dataset Information & Strategy](#-dataset-information--strategy)
13. [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
14. [Feature Engineering (TF-IDF)](#-feature-engineering-tf-idf)
15. [Model Comparison & Benchmarks](#-model-comparison--benchmarks)
16. [Evaluation Metrics](#-evaluation-metrics)
17. [Explainability & Interpretability](#-explainability--interpretability)
18. [Project Structure](#-project-structure)
19. [Installation Guide](#-installation-guide)
20. [How to Run Locally](#-how-to-run-locally)
21. [How to Train the Model](#-how-to-train-the-model)
22. [How to Run Automated Tests](#-how-to-run-automated-tests)
23. [Streamlit Community Cloud Deployment](#-streamlit-community-cloud-deployment)
24. [Screenshots Gallery](#-screenshots-gallery)
25. [Limitations](#-limitations)
26. [Ethical Considerations](#-ethical-considerations)
27. [Future Improvements](#-future-improvements)
28. [Responsible AI Disclaimer](#-responsible-ai-disclaimer)
29. [License](#-license)
30. [Author & Contact](#-author--contact)

---

## 📖 Project Overview
The **AI-Powered Fake News Detection System** is an end-to-end Machine Learning web application designed to evaluate the rhetorical credibility and linguistic signatures of news articles. Rather than treating artificial intelligence as an infallible "truth arbiter," this system identifies statistical patterns, vocabulary distributions, emotional tone, and sensationalist phrasing typical of fabricated reporting.

Built with modular software engineering practices, the project encompasses automated dataset cleansing, stratified train/test partitioning, sublinear TF-IDF vectorization, multi-model candidate benchmarking (Logistic Regression, Multinomial Naive Bayes, and Calibrated Linear SVM), automated model selection, and an interactive 4-tab Streamlit dashboard.

---

## 🎯 Problem Statement
Digital news proliferation and algorithmic social feeds have accelerated the spread of fabricated, sensationalist, and misleading information. Identifying misinformation at scale presents distinct technical challenges:
- **Linguistic Mimicry**: Fabricated articles frequently adopt journalistic formats while embedding unverified claims, emotional superlatives, and anonymous attributions.
- **Black-Box Skepticism**: End users and content moderators cannot trust predictions without transparent, interpretable rationales.
- **Speed & Scalability**: Content moderation workflows require sub-second inference latency without requiring gigabyte-scale GPU clusters.

This project addresses these challenges by combining fast, interpretable n-gram TF-IDF representations with probability-calibrated classifiers and local token attribution.

---

## 🚀 Objectives
1. **Deliver Production Architecture**: Construct a clean, modular Python codebase separating data loading, preprocessing, feature extraction, training, evaluation, explainability, and inference.
2. **Benchmark Multiple Architectures**: Train and empirically compare Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Machines on holdout test data using stratified sampling.
3. **Calibrate Confidence**: Implement Platt scaling calibration (`CalibratedClassifierCV`) to ensure non-probabilistic models output reliable class probabilities.
4. **Transparent Explainability**: Provide both global feature importance and document-level token contributions with plain-English summaries.
5. **Rigorous Quality Assurance**: Maintain a comprehensive unit and integration test suite with 100% pass rates.

---

## ✨ Key Features
- **Interactive Multi-Tab Dashboard**: Seamlessly switch between News Analysis, Benchmark Comparisons, Exploratory Data Analysis, and Architecture Methodology.
- **One-Click Sample Loaders**: Test verified authentic news reports or sensationalist conspiracies with a single click.
- **Calibrated Dual-Class Probabilities**: Visual progress bars showing exact percentage confidence for both REAL and FAKE categories.
- **Rhetorical & Sensationalism Profiling**: Analyzes uppercase word ratios, dramatic punctuation intensity (`!`, `?`), emotional sentiment, and composite sensationalism scores ($0\% - 100\%$).
- **Key Indicative Terms**: Displays top TF-IDF unigrams and bigrams contributing to the document's classification.
- **Plain-English Model Reasoning**: Generates accessible narrative explanations detailing *why* the model produced its prediction, contrasting authentic indicators against misleading indicators.
- **Zero-Friction Startup**: Ships with a balanced 30-article benchmark dataset in `data/sample/sample_news.csv` enabling instant training and test execution out-of-the-box.
- **Headless Plot Generation**: Automatically exports high-resolution confusion matrix heatmaps, model comparisons, and EDA distributions.

---

## 🛠 Technologies Used
- **Programming Language**: Python 3.10 - 3.14
- **Web Application Framework**: Streamlit (v1.30+)
- **Machine Learning**: Scikit-Learn (Logistic Regression, MultinomialNB, CalibratedClassifierCV, LinearSVC)
- **Natural Language Processing**: TF-IDF Vectorization, N-Gram Modeling, Contraction Normalization, Regex Lexical Parsing
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Serialization**: Joblib
- **Testing**: Pytest (40 unit and integration tests)

---

## 🧠 NLP Concepts & Techniques
1. **Contraction Normalization**: Standardizes colloquial forms (e.g., *"don't"* $\rightarrow$ *"do not"*, *"it's"* $\rightarrow$ *"it is"*) to preserve auxiliary verbs and negation markers critical for tone detection.
2. **Noise Sanitization**: Eliminates web scraping artifacts, HTML tags (`<.*?>`), and URLs (`http\S+`) while decoding HTML entities (`&amp;` $\rightarrow$ `&`).
3. **Headline & Body Fusion**: Combines article titles with body paragraphs, ensuring clickbait markers concentrated in headlines inform the feature space.
4. **N-Gram Modeling**: Captures single words (unigrams: *"whistleblower"*, *"reuters"*) and local word pairs (bigrams: *"tap water"*, *"central bank"*, *"claims to"*).
5. **Sublinear Term Frequency Scaling**: Dampens word repetition using:
   $$\text{TF}_{\text{sublinear}} = 1 + \log(\text{TF}) \quad \text{for } \text{TF} > 0$$
   This prevents clickbait articles repeating emotional words 30 times from dominating the vector space.
6. **Inverse Document Frequency (IDF)**: Downweights ubiquitous terms across the corpus:
   $$\text{IDF}(t) = \log \frac{1 + n}{1 + \text{DF}(t)} + 1$$

---

## 🤖 Machine Learning Concepts
1. **Supervised Binary Classification**: Maps input vectors $\mathbf{x} \in \mathbb{R}^d$ to labels $y \in \{0, 1\}$, where $0 = \text{REAL}$ and $1 = \text{FAKE}$.
2. **Stratified Train/Test Split**: Preserves exact 50/50 class distributions across training and holdout test splits, eliminating class-imbalance bias.
3. **Strict Data Isolation**: Vectorizer is fitted *strictly* on training text (`fit_transform`) and subsequently transforms test sets (`transform`), eliminating data leakage.
4. **Probability Calibration (Platt Scaling)**: Wraps `LinearSVC` with a logistic sigmoid mapping via `CalibratedClassifierCV(method='sigmoid', cv=3)` to output true probabilities:
   $$P(y=1 | f(\mathbf{x})) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$
5. **Multi-Metric Evaluation**: Assesses performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## 🏗 System Architecture
```
User / Web Browser
       │
       ▼
Streamlit Frontend (app.py) ── 4 Interactive Tabs
       │
       ▼
Inference Engine (src/predictor.py)
       ├── Preprocessing (src/preprocessing.py)
       ├── Feature Extraction (src/feature_extractor.py)
       ├── Sentiment & Sensationalism (src/utils.py)
       └── Explainability (src/explainability.py)
       │
       ▼
Serialized Artifacts (models/)
       ├── best_model.joblib
       ├── tfidf_vectorizer.joblib
       └── model_metrics.json
```

---

## 🔬 Machine Learning Methodology
The development follows a structured 6-stage lifecycle:
1. **Ingestion & Validation**: Schema standardization, duplicate removal, and missing-value filtering.
2. **Exploratory Data Analysis**: Profiling class balance, word count distributions, and top n-grams per class.
3. **Vectorization**: TF-IDF extraction with unigrams, bigrams, and sublinear scaling.
4. **Candidate Training**: Fitting Logistic Regression, Multinomial Naive Bayes, and Calibrated Linear SVM.
5. **Holdout Evaluation**: Evaluating all candidate models on unseen test data using F1-score as primary selection criteria.
6. **Persistence**: Serializing winning model and feature extractor for sub-millisecond production inference.

---

## 📊 Dataset Information & Strategy
- **Target Schema**: `id` (int), `title` (str), `text` (str), `label` (0 for REAL, 1 for FAKE).
- **Benchmark Sample Included**: `data/sample/sample_news.csv` contains 30 verified articles (15 genuine reports from Reuters, AP, and AFP; 15 fabricated articles exhibiting typical conspiracy and clickbait tropes).
- **Public External Datasets Supported**:
  - **ISOT Fake News Dataset** (University of Victoria): `True.csv` and `Fake.csv`.
  - **Kaggle Fake News Dataset**: `train.csv` (20,800 news articles).
  - **WELFake Dataset**: 72,134 news articles.
- **Dataset Manager**: `python scripts/download_data.py --check` validates dataset integrity and column schemas automatically.

---

## 🧹 Data Preprocessing Pipeline
Implemented in `src/preprocessing.py`:
1. **Type Coercion**: Handles `None`, numeric values, and non-strings safely.
2. **HTML Entity & Tag Stripping**: Decodes entities and strips `<p>`, `<a>`, and formatting tags.
3. **URL Removal**: Strips hyperlinks matching `https?://\S+|www\.\S+`.
4. **Contraction Expansion**: Replaces 45+ common contractions with canonical forms.
5. **Case Lowering**: Converts text to standard lowercase.
6. **Punctuation Stripping**: Preserves alphabetic characters and clean whitespace.
7. **Whitespace Normalization**: Compresses multi-spaces, tabs, and newlines into single spaces.

---

## ⚙️ Feature Engineering (TF-IDF)
Implemented in `src/feature_extractor.py`:
- `ngram_range=(1, 2)`: Evaluates single words and paired collocations.
- `sublinear_tf=True`: Dampens the impact of extreme word frequencies.
- `min_df=2`: Excludes rare single-occurrence typos (adapts to `min_df=1` if dataset is small).
- `max_df=0.90`: Excludes ubiquitous words present in over 90% of documents.
- `stop_words='english'`: Filters generic stopwords to highlight substantive keywords.

---

## 📈 Model Comparison & Benchmarks
Candidate models evaluated on holdout test data:

| Architecture | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Fit Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** *(Champion)* | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0056s** |
| **Logistic Regression** | 83.3% | 1.0000 | 0.6667 | 0.8000 | 1.0000 | 0.0067s |
| **Linear SVM (Calibrated)** | 83.3% | 1.0000 | 0.6667 | 0.8000 | 1.0000 | 0.0476s |

*Generated benchmark charts are available in `assets/model_comparison.png` and `assets/confusion_matrices.png`.*

---

## 📏 Evaluation Metrics
- **Accuracy**: Overall fraction of correct classifications: $\frac{TP + TN}{TP + TN + FP + FN}$.
- **Precision**: Fraction of predicted fake articles that are truly fake: $\frac{TP}{TP + FP}$. Minimizes falsely accusing legitimate news agencies.
- **Recall**: Fraction of actual fake articles successfully flagged: $\frac{TP}{TP + FN}$. Minimizes undetected misinformation.
- **F1-Score**: Harmonic mean balancing precision and recall: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **ROC-AUC**: Evaluates the model's ability to discriminate between classes across all decision thresholds.

---

## 💡 Explainability & Interpretability
Implemented in `src/explainability.py`:
1. **Global Vocabulary Weights**: Inspects model coefficients and log-odds ratios to identify top words associated with authenticity and fabrication across the dataset.
2. **Local Token Contributions**: Calculates individual token impact:
   $$\text{Contribution}(w) = \text{TF-IDF}(w) \cdot \text{Weight}_{\text{model}}(w)$$
3. **Contrasting Indicators**: Splits article tokens into:
   - 🟢 **Authentic / Journalistic Indicators** (pushing toward REAL).
   - 🔴 **Sensationalist / Misleading Indicators** (pushing toward FAKE).
4. **Plain-English Explanations**: Translates mathematical weights into readable summaries for non-technical users and recruiters.

---

## 📁 Project Structure
```
AI-Powered Fake News Detection System/
│
├── app.py                      # Main Streamlit web application (4 interactive tabs)
├── requirements.txt            # Production dependencies
├── pytest.ini                  # Pytest configuration and warning filters
├── README.md                   # Complete production documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── .env.example                # Environment configuration template
│
├── data/
│   ├── raw/                    # External datasets (e.g. Kaggle / ISOT)
│   ├── processed/              # Cleaned stratified splits (train.csv, test.csv)
│   └── sample/                 # Curated 30-article benchmark dataset
│       └── sample_news.csv
│
├── models/                     # Production serialized artifacts
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_metrics.json
│
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── config.py               # Cross-platform paths and hyperparameters
│   ├── utils.py                # Logging, serialization, text metrics, sensationalism
│   ├── preprocessing.py        # Text cleaning, contraction expansion, sanitization
│   ├── data_loader.py          # Data ingestion, schema validation, stratified split
│   ├── feature_extractor.py    # TF-IDF vectorizer with document-level ranking
│   ├── train.py                # Multi-model training and calibration pipeline
│   ├── evaluate.py             # Evaluation metrics, model selection, chart plotting
│   ├── predictor.py            # High-level inference engine
│   ├── explainability.py       # Global feature weights and local token attributions
│   └── eda.py                  # Linguistic statistics and EDA visualization generation
│
├── tests/                      # Automated test suite (40 tests)
│   ├── __init__.py
│   ├── test_utils.py
│   ├── test_preprocessing.py
│   ├── test_eda.py
│   ├── test_features.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_predictor.py
│   ├── test_explainability.py
│   └── test_pipeline_e2e.py
│
├── notebooks/                  # Interactive exploration
│   └── model_exploration.ipynb
│
├── scripts/                    # Automation utilities
│   └── download_data.py        # Dataset manager and schema verification
│
└── assets/                     # Visual figures and portfolio screenshots
    ├── confusion_matrices.png
    ├── model_comparison.png
    ├── eda_class_distribution.png
    ├── eda_word_count_distribution.png
    └── screenshots/
        └── README.md
```

---

## 💻 Installation Guide

### Prerequisites
- Python 3.10 to 3.14 installed on Windows, macOS, or Linux.
- Git installed.

### Clone the Repository
```bash
git clone https://github.com/your-username/ai-fake-news-detection-system.git
cd "ai-fake-news-detection-system"
```

### Create & Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run Locally
Launch the Streamlit web application with:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🏋️ How to Train the Model
To re-run data cleaning, stratified splitting, feature extraction, candidate training, evaluation, and artifact serialization:
```bash
# 1. Clean and split data
python -m src.data_loader

# 2. Train candidates, evaluate, and save champion model
python -m src.evaluate
```

---

## 🧪 How to Run Automated Tests
Run the complete test suite using pytest:
```bash
# Run all 40 unit and integration tests
pytest -q

# Run verbose test suite
pytest
```

---

## ☁️ Streamlit Community Cloud Deployment
This project is pre-configured for instant zero-configuration deployment to **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New App"** and select your repository.
4. Set Main file path to: `app.py`.
5. Click **"Deploy"**.
*Because all paths use dynamic relative `pathlib.Path` resolution and dependencies are pinned in `requirements.txt`, deployment will succeed on Linux cloud instances with zero environment friction.*

---

## 📸 Screenshots Gallery
Portfolio screenshots can be stored in `assets/screenshots/`:
- `home_screen.png`: Landing page with article input and example buttons.
- `real_prediction.png`: Credible news analysis with green result card and text metrics.
- `fake_prediction.png`: Misleading news analysis with sensationalism index and red result card.
- `model_comparison.png`: Performance chart across candidate models.
- `confusion_matrices.png`: Side-by-side heatmaps of confusion matrices.
- `eda_distributions.png`: Dataset class balance and word count spread.

---

## ⚠️ Limitations
- **Linguistic Pattern Dependency**: The model evaluates rhetorical and syntactic patterns rather than factual truth. If a fabricated article mimics formal journalistic style with high fidelity, lexical models may assign higher credibility scores.
- **Domain Adaptation**: Vocabulary trained on political and international news may exhibit lower accuracy when applied to niche domains (e.g., biomedical preprints or financial quarterly SEC filings).
- **Evolving Misinformation**: Adversarial content creators continuously adapt rhetorical strategies to evade keyword-based filters.

---

## 🛡 Ethical Considerations
- **No Truth Censorship**: This tool is designed to assist content readers, researchers, and journalists, not to serve as an automated censorship engine.
- **Fairness & Bias**: Prejudices present in historical news corpora can influence vocabulary weights. Continuous auditing is required to prevent demographic or political bias.
- **Transparency First**: Local token attribution and plain-English rationales are provided so users can critically evaluate *why* an article was flagged.

---

## 🔮 Future Improvements
1. **Contextual Transformer Embeddings**: Benchmark RoBERTa or DeBERTa models alongside TF-IDF to capture deeper semantic relationships.
2. **Knowledge Graph Fact-Checking**: Integrate automated claim verification via external knowledge graphs (e.g., Google Fact Check API, Wikipedia API).
3. **URL & Domain Reputation Lookup**: Cross-reference article publishing domains against trusted media registries.
4. **Multilingual Classification**: Extend text normalization and vectorization to support cross-lingual fake news detection.

---

## ⚠️ Responsible AI Disclaimer
> **Notice**: This application uses machine learning pattern recognition trained on linguistic features and historical news datasets. It provides a probabilistic assessment and is **NOT a guaranteed fact-checking service or truth verifier**. Always cross-reference critical claims with primary accredited journalistic organizations and official public records.

---

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author & Contact
- **Author**: AI/ML Engineer & ECE Graduate
- **GitHub**: [@your-username](https://github.com/your-username)
- **LinkedIn**: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- **Portfolio**: [your-portfolio.com](https://your-portfolio.com)
