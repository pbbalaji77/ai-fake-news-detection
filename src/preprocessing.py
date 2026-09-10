"""
Text Preprocessing Pipeline for Fake News Detection.

Provides robust, modular text cleaning and normalization tailored for news articles.

Design Decisions & Rationale:
1. Contraction Expansion: Expanding contractions (e.g., "won't" -> "will not") normalizes
   negations and auxiliary verbs, preserving important semantic signals.
2. URL & HTML Removal: News articles scraped from web sources often contain HTML tags
   and tracking URLs that introduce noise rather than journalistic signal.
3. Lowercasing: Standardizes vocabulary so "BREAKING" and "breaking" share representation.
4. Punctuation Handling: Punctuation is stripped in text normalization, but excessive punctuation
   (e.g., "???", "!!!") can be measured as a stylistic signal before removal.
5. Stopword Handling: Stopwords are optionally removable. For TF-IDF, retaining common function
   words or letting TF-IDF's max_df/idf naturally downweight ubiquitous words often outperforms
   aggressive stopword removal, because stylistic function words and pronouns can signal
   deceptive or sensationalist writing styles.
6. Edge Case Safety: Gracefully handles None, non-strings, numbers, and whitespace-only inputs.
"""

import html
import re
from typing import Optional

# Common English contraction mappings for text normalization
CONTRACTIONS_DICT = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Pre-compiled regex patterns for performance
CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in CONTRACTIONS_DICT.keys()) + r")\b",
    flags=re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<.*?>")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def expand_contractions(text: str) -> str:
    """
    Expands common conversational contractions to standard full-form words.
    
    Args:
        text: Raw text string.
        
    Returns:
        String with contractions expanded.
    """
    def replace(match):
        matched_text = match.group(0).lower()
        return CONTRACTIONS_DICT.get(matched_text, match.group(0))

    return CONTRACTION_PATTERN.sub(replace, text)


def clean_text(text: Optional[str]) -> str:
    """
    Executes the full text cleaning and normalization pipeline.
    
    Processing steps:
    1. Type safety: Converts None or non-string inputs to empty string.
    2. HTML Unescape & Tag Removal: Decodes entities (&amp; -> &) and strips HTML tags.
    3. URL & Hyperlink Removal: Strips web addresses.
    4. Contraction Expansion: Normalizes colloquialisms.
    5. Case Normalization: Lowercases all characters.
    6. Punctuation & Special Character Removal: Preserves only alphabetic letters and whitespace.
    7. Whitespace Collapsing: Reduces multiple spaces, tabs, and newlines to a single space.
    
    Args:
        text: Raw text input from headline or article body.
        
    Returns:
        Cleaned, normalized string ready for feature extraction.
    """
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)

    # Decode HTML entities (e.g. &amp; -> &, &quot; -> ")
    text = html.unescape(text)

    # Strip HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)

    # Remove URLs
    text = URL_PATTERN.sub(" ", text)

    # Expand contractions
    text = expand_contractions(text)

    # Lowercase
    text = text.lower()

    # Remove non-alphabetic characters (numbers, punctuation, symbols)
    text = SPECIAL_CHAR_PATTERN.sub(" ", text)

    # Collapse consecutive whitespace and strip edges
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def combine_title_and_text(title: Optional[str], text: Optional[str]) -> str:
    """
    Combines headline title and article text into a single cohesive string.
    Headlines often contain the strongest clickbait/sensationalist indicators.
    
    Args:
        title: Headline string.
        text: Article body string.
        
    Returns:
        Combined raw text string.
    """
    safe_title = "" if title is None else str(title).strip()
    safe_text = "" if text is None else str(text).strip()

    if safe_title and safe_text:
        return f"{safe_title} {safe_text}"
    elif safe_title:
        return safe_title
    elif safe_text:
        return safe_text
    return ""
