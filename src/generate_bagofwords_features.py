#!/usr/bin/env python3
"""
Generate bag-of-words feature matrices from multilingual text data,
using language-specific tokenization where appropriate.
Handles Chinese, Japanese, Korean, Hindi, Hebrew, and Thai with specialized tokenizers.

Saves the resulting vocabularies and feature matrices for later use.

Key Features:
* Multilingual tokenization supporting 12+ languages (Chinese, Japanese, Korean, Hindi, Hebrew, English, French, German, Spanish, Italian, Portuguese, Indonesian)
* Language-specific tokenizers: jieba (Chinese), Sudachi (Japanese), KoNLPy (Korean), with regex fallbacks for others
* Mixed and language-specific vocabularies for both source (original) and target (translated) texts
* Configurable vocabulary size (default: 5000 most frequent tokens via `max_features`)
* L2-normalized feature matrices for consistent downstream use

Usage (from repository root):
    python src/generate_bagofwords_features.py \
    --input data/processed/eclektic_long_subset.csv \
    --output data/processed/bow/

Note on dimensions and metadata alignment:
    - target_<lang> files: Filter metadata by `language == <lang>` (e.g., 39 rows for target_zh)
    - target_mix: Use all metadata rows (e.g., 468 rows for all languages)
    
    To align (e.g., say if you want to explore chinese target q-a results using chinese-specific bow features):
        metadata = pd.read_csv("./data/bow/metadata.csv")
        vectorizer, matrix = joblib.load("./data/bow/target_zh_features.pkl")
        zh_metadata = metadata[metadata["language"] == "zh"].reset_index(drop=True)
        assert len(zh_metadata) == matrix.shape[0] # should be true
"""

import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import re
import jieba

try:
    from sudachipy import tokenizer as sudachi_tokenizer
    from sudachipy import dictionary
    SUDACHI_AVAILABLE = True
    sudachi_dict = dictionary.Dictionary().create()
    sudachi_mode = sudachi_tokenizer.Tokenizer.SplitMode.C
except ImportError:
    SUDACHI_AVAILABLE = False
    print("Warning: Sudachi not available. Install with: pip install sudachipy sudachidict-core")

try:
    from konlpy.tag import Okt
    # Test if Java is actually available
    try:
        okt = Okt()
        okt.morphs("테스트")
        KONLPY_AVAILABLE = True
    except:
        KONLPY_AVAILABLE = False
        print("Warning: KoNLPy installed but Java not available. Using fallback for Korean.")
except ImportError:
    KONLPY_AVAILABLE = False

try:
    from pythainlp.tokenize import word_tokenize as thai_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False


def chinese_tokenizer(text):
    """Use jieba to segment Chinese text."""
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip() and re.match(r'^[\u4e00-\u9fff]+$', t)]


def japanese_tokenizer(text):
    """Use Sudachi to segment Japanese text."""
    if not SUDACHI_AVAILABLE:
        # Fallback: character-level tokenization
        print("Warning: Using character-level fallback for Japanese. Install Sudachi for better results.")
        tokens = []
        for char in text:
            if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
        return tokens if tokens else [text]
    
    try:
        # Use Sudachi tokenizer
        tokens = [m.surface() for m in sudachi_dict.tokenize(text, sudachi_mode)]
        # Keep only tokens with Japanese characters
        tokens = [t for t in tokens if t.strip() and re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', t)]
        return tokens if tokens else [text]
    except Exception as e:
        print(f"Warning: Sudachi tokenization failed: {e}. Using character-level fallback.")
        # Fallback to character-level
        tokens = []
        for char in text:
            if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
        return tokens if tokens else [text]


def korean_tokenizer(text):
    """Use KoNLPy to segment Korean text."""
    if not KONLPY_AVAILABLE:
        # Fallback: simple space splitting + filter Korean
        tokens = text.split()
        return [t for t in tokens if t.strip() and re.search(r'[\uac00-\ud7af]', t)]
    
    try:
        okt = Okt()
        tokens = okt.morphs(text)
        return [t for t in tokens if t.strip() and re.match(r'^[\uac00-\ud7af]+$', t)]
    except:
        # Fallback
        tokens = text.split()
        return [t for t in tokens if t.strip() and re.search(r'[\uac00-\ud7af]', t)]


def hindi_tokenizer(text):
    """Simple tokenizer for Hindi (Devanagari script)."""
    # Split on whitespace and punctuation, keep Devanagari
    tokens = re.findall(r'[\u0900-\u097f]+', text)
    return [t for t in tokens if len(t) > 1]  # Filter single characters


def hebrew_tokenizer(text):
    """Simple tokenizer for Hebrew."""
    # Split on whitespace and punctuation, keep Hebrew characters
    tokens = re.findall(r'[\u0590-\u05ff]+', text)
    return [t for t in tokens if len(t) > 1]  # Filter single characters


def thai_tokenizer(text):
    """Use pythainlp to segment Thai text."""
    if not PYTHAINLP_AVAILABLE:
        # Fallback: extract Thai character sequences
        return re.findall(r'[\u0e00-\u0e7f]+', text)
    
    try:
        tokens = thai_tokenize(text, engine='newmm')
        return [t for t in tokens if t.strip() and re.match(r'^[\u0e00-\u0e7f]+$', t)]
    except:
        return re.findall(r'[\u0e00-\u0e7f]+', text)


def get_language_config(lang):
    """
    Return tokenizer and stop_words configuration for a given language.
    Returns: (tokenizer_function, stop_words, token_pattern)
    
    Note: scikit-learn only has built-in 'english' stop words.
    For other languages, we use None or you can provide custom lists.
    """
    # Languages requiring special tokenization (no word boundaries or complex morphology)
    if lang and lang.startswith("zh"):
        return chinese_tokenizer, None, None
    elif lang and lang.startswith("ja"):
        return japanese_tokenizer, None, None
    elif lang and lang.startswith("ko"):
        return korean_tokenizer, None, None
    elif lang and lang.startswith("hi"):
        return hindi_tokenizer, None, None
    elif lang and lang.startswith("he"):
        return hebrew_tokenizer, None, None
    elif lang and lang.startswith("th"):
        return thai_tokenizer, None, None
    
    # English has built-in stop words
    elif lang and lang.startswith("en"):
        return None, 'english', r"(?u)\b\w\w+\b"
    
    # Other European languages: use default tokenization, no stop words
    # (sklearn only has 'english' built-in)
    elif lang in ['fr', 'de', 'es', 'it', 'pt', 'id']:
        return None, None, r"(?u)\b\w\w+\b"
    
    # Default for other languages
    else:
        return None, None, r"(?u)\b\w\w+\b"


def make_bow(texts, max_features=5000, stop_words='english', lang=None):
    """
    Fit a CountVectorizer and return (vectorizer, normalized matrix).
    Uses language-specific tokenization when needed.
    """
    vectorizer_params = {
        'max_features': max_features,
    }
    
    # Get language-specific configuration
    tokenizer, stop_words_lang, token_pattern = get_language_config(lang)
    
    if tokenizer is not None:
        vectorizer_params['tokenizer'] = tokenizer
        vectorizer_params['token_pattern'] = None
    else:
        if stop_words_lang:
            vectorizer_params['stop_words'] = stop_words_lang
        vectorizer_params['token_pattern'] = token_pattern
    
    vectorizer = CountVectorizer(**vectorizer_params)

    mat = vectorizer.fit_transform(texts)
    mat = normalize(mat, norm="l2")
    return vectorizer, mat


def save_artifacts(name, vectorizer, matrix, out_dir):
    """
    Save vectorizer vocab (.json) and matrix (.pkl) under consistent naming.
    """
    os.makedirs(out_dir, exist_ok=True)
    vocab_path = os.path.join(out_dir, f"{name}_vocab.json")
    pkl_path = os.path.join(out_dir, f"{name}_features.pkl")

    with open(vocab_path, "w", encoding="utf-8") as f:
        vocab_clean = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
        json.dump(vocab_clean, f, ensure_ascii=False, indent=2)

    joblib.dump((vectorizer, matrix), pkl_path)

    print(f"Saved {name}: {matrix.shape} → {pkl_path}")


def main(input_path, out_dir="../processed/bow", max_features=5000):
    df = pd.read_csv(input_path)
    df["original_lang"] = df["original_lang"].astype(str)
    df["language"] = df["language"].astype(str)

    os.makedirs(out_dir, exist_ok=True)
    
    # Print available tokenizers
    print("=" * 60)
    print("Tokenizer Availability:")
    print(f"  Chinese (jieba): Always available")
    print(f"  Japanese (Sudachi): {SUDACHI_AVAILABLE}")
    print(f"  Korean (KoNLPy): {KONLPY_AVAILABLE}")
    print(f"  Thai (pythainlp): {PYTHAINLP_AVAILABLE}")
    print(f"  Hindi/Hebrew: Fallback regex-based")
    print("=" * 60)

    # -----------------------------------------------------
    # 1 SOURCE MIX  (all source texts, regardless of lang)
    # -----------------------------------------------------
    src_texts = df[["original_question"]].fillna("").agg(" ".join, axis=1)
    vect, mat = make_bow(src_texts, max_features=max_features)
    save_artifacts("source_mix", vect, mat, out_dir)

    # -----------------------------------------------------
    # 2 SOURCE-LANGUAGE-SPECIFIC
    # -----------------------------------------------------
    for lang in sorted(df["original_lang"].unique()):
        sub = df[df["original_lang"] == lang]
        if sub.empty:
            continue
        texts = sub[["original_question"]].fillna("").agg(" ".join, axis=1)
        vect, mat = make_bow(texts, max_features=max_features, lang=lang)
        save_artifacts(f"source_{lang}", vect, mat, out_dir)

    # -----------------------------------------------------
    # 3 TARGET MIX  (all target texts, all langs combined)
    # -----------------------------------------------------
    target_texts = df["question"].fillna("")
    vect, mat = make_bow(target_texts, max_features=max_features)
    save_artifacts("target_mix", vect, mat, out_dir)

    # -----------------------------------------------------
    # 4 TARGET-LANGUAGE-SPECIFIC
    # -----------------------------------------------------
    for lang in sorted(df["language"].unique()):
        sub = df[df["language"] == lang]
        if sub.empty:
            continue
        texts = sub["question"].fillna("")
        vect, mat = make_bow(texts, max_features=max_features, lang=lang)
        save_artifacts(f"target_{lang}", vect, mat, out_dir)

    # -----------------------------------------------------
    # 5 Metadata save (optional)
    # -----------------------------------------------------
    df[["q_id", "original_lang", "language", "title", "url"]].to_csv(
        os.path.join(out_dir, "metadata.csv"), index=False
    )
    print("\n" + "=" * 60)
    print("All vocabularies and feature matrices saved.")
    print("=" * 60)


if __name__ == "__main__":
    input_path = "./data/processed/eclektic_long_subset.csv"
    out_dir="./data/processed/bow"
    main(input_path, out_dir, max_features=5000)