"""
annies_queries.py

Reproduces the cooc_features.py pipeline, replacing the local streaming +
sliding-window co-occurrence logic with Elasticsearch queries via WimbdHelper.

Tokenization and the CLI entry point are kept identical to cooc_features.py.
Co-occurrence is now document-level (a "window" = one mC4 document).
"""

import sys
import os

# Allow importing from the parent src/ directory (where `tokenizers` lives)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import json
import itertools
import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tokenizers import get_language_config, get_stopwords_for_lang
from wimbd_helper import WimbdHelper
from calculate_co_occurrences import calculate_ppmi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Tokenization  (identical to cooc_features.py)
# ---------------------------------------------------------------------------

def tokenize(text: str, lang: Optional[str] = None) -> list[str]:
    if lang:
        tokenizer_fn, _, _ = get_language_config(lang)
        if tokenizer_fn is not None:
            return [t.lower() for t in tokenizer_fn(text)]
    return re.findall(r"\w+", str(text).lower())


def extract_keywords_from_text(
    text: str,
    stopwords: Optional[set] = None,
    max_per_text: Optional[int] = None,
    lang: Optional[str] = None,
) -> list[str]:
    toks = tokenize(text, lang=lang)
    stopwords = get_stopwords_for_lang(lang or "")
    if stopwords is None:
        stopwords = set()
    content = [t for t in toks if t not in stopwords]
    seen: set[str] = set()
    keywords: list[str] = []
    for t in content:
        if t not in seen:
            seen.add(t)
            keywords.append(t)
    if max_per_text is not None:
        keywords = keywords[:max_per_text]
    return keywords


def build_language_and_qa_keywords(
    df: pd.DataFrame,
    question_col: str = "question",
    answer_col: str = "answer",
    lang_col: str = "language",
    qid_col: str = "q_id",
) -> dict[tuple[int, str], list[str]]:
    qa_keywords: dict[tuple[int, str], list[str]] = {}
    for _, row in df.iterrows():
        lang = row[lang_col]
        qid = row[qid_col]
        text = f'{row[question_col]} {row[answer_col]}'
        kws = extract_keywords_from_text(text, lang=lang)
        qa_keywords[(qid, lang)] = kws
    return qa_keywords


def save_qa_keywords(qa_keywords, path="./cooc_features/qa_keywords_wimbd.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {f"{qid}_{lang}": kws for (qid, lang), kws in qa_keywords.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_qa_keywords(path="./cooc_features/qa_keywords.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {tuple(k.split("_", 1)): v for k, v in data.items()}


# ---------------------------------------------------------------------------
# ES-based co-occurrence  (replaces streaming + sliding window)
# ---------------------------------------------------------------------------

def compute_cooc_for_language_es(
    lang_code: str,
    keyword_vocab: set[str],
    helper: WimbdHelper,
) -> tuple[Counter, Counter, int]:
    """
    Query Elasticsearch for document-level unigram and co-occurrence counts.

    Returns:
        unigram_counts  – Counter mapping keyword -> doc count
        cooc_counts     – Counter mapping (w1, w2) -> doc count (w1 < w2)
        total_docs      – total documents in the index
    """
    if lang_code == "he":
        lang_code = "iw"

    index = f"mc4_{lang_code}"

    all_stats = helper.get_index_stats()
    if index not in all_stats:
        logger.warning(f"Index {index} not found in Elasticsearch.")
        return Counter(), Counter(), 0
    total_docs = int(all_stats[index].get("docs.count", 0))

    if total_docs == 0:
        logger.warning(f"Index {index} has 0 documents.")
        return Counter(), Counter(), 0

    keywords = sorted(keyword_vocab)
    unigram_counts: Counter = Counter()
    cooc_counts: Counter = Counter()

    logger.info(f"[{index}] Querying {len(keywords)} unigrams …")
    for kw in keywords:
        count = helper.count_phrases(index, [kw])
        if count > 0:
            unigram_counts[kw] = count

    seen_kws = sorted(kw for kw in keywords if kw in unigram_counts)
    n_pairs = len(seen_kws) * (len(seen_kws) - 1) // 2
    logger.info(f"[{index}] Querying {n_pairs} keyword pairs …")

    for w1, w2 in itertools.combinations(seen_kws, 2):
        count = helper.count_phrases(index, [w1, w2], all_phrases=True)
        if count > 0:
            cooc_counts[(w1, w2)] = count

    return unigram_counts, cooc_counts, total_docs


def compute_pmi_matrix(
    unigram_counts: Counter,
    cooc_counts: Counter,
    total_docs: int,
    min_cooc: int = 1,
) -> dict[tuple[str, str], float]:
    """PPMI from ES document counts (reuses calculate_ppmi from calculate_co_occurrences)."""
    pmi: dict[tuple[str, str], float] = {}
    for (w1, w2), cooc_count in cooc_counts.items():
        if cooc_count < min_cooc:
            continue
        n1 = unigram_counts.get(w1, 0)
        n2 = unigram_counts.get(w2, 0)
        ppmi_val = calculate_ppmi(cooc_count, n1, n2, total_docs)
        if ppmi_val > 0.0:
            pmi[(w1, w2)] = ppmi_val
    return pmi


# ---------------------------------------------------------------------------
# Feature building  (mirrors build_qa_cooc_features from cooc_features.py)
# ---------------------------------------------------------------------------

def _process_lang_worker(
    lang: str,
    lang_kw_vocab: dict,
    es_url: str,
    min_cooc: int,
):
    """Top-level worker so it can be pickled by ProcessPoolExecutor."""
    kw_vocab = lang_kw_vocab.get(lang, set())
    if not kw_vocab:
        return lang, (Counter(), Counter(), 0, {})
    lang_helper = WimbdHelper(es_url=es_url)
    unigram_counts, cooc_counts, total_docs = compute_cooc_for_language_es(
        lang_code=lang,
        keyword_vocab=kw_vocab,
        helper=lang_helper,
    )
    pmi_dict = compute_pmi_matrix(unigram_counts, cooc_counts, total_docs, min_cooc=min_cooc)
    return lang, (unigram_counts, cooc_counts, total_docs, pmi_dict)

def save_qa_pair_details(q_id, lang, pairs_info, out_dir="data/intermediate/qa_cooc"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{lang}_{q_id}_wimbd.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs_info, f, ensure_ascii=False, indent=2)


def build_qa_cooc_features(
    df: pd.DataFrame,
    qa_keywords: dict,
    helper: WimbdHelper,
    per_qa_out_dir: str = "./cooc_features/qa_cooc",
    min_cooc: int = 1,
    iterative_out_csv: Optional[str] = None,
) -> pd.DataFrame:
    if iterative_out_csv is not None:
        iterative_out_csv = str(iterative_out_csv)

    feature_rows = []
    output_rows = []
    batch_rows = []
    batch_size = 12

    langs = sorted(df["language"].unique())

    # Union keyword vocabulary per language
    lang_kw_vocab: dict[str, set[str]] = {lang: set() for lang in langs}
    for (qid, lang), kws in qa_keywords.items():
        if lang in lang_kw_vocab:
            lang_kw_vocab[lang].update(kws)

    # Compute ES-based co-occurrence once per language, in parallel (one worker per language)
    from concurrent.futures import ProcessPoolExecutor
    lang_stats: dict[str, tuple[Counter, Counter, int, dict]] = {}
    with ProcessPoolExecutor(max_workers=len(langs)) as executor:
        for lang, stats in executor.map(
            _process_lang_worker,
            langs,
            [lang_kw_vocab] * len(langs),
            [helper.es_url] * len(langs),
            [min_cooc] * len(langs),
        ):
            lang_stats[lang] = stats

    def _flush_batch():
        nonlocal batch_rows
        if iterative_out_csv is not None and batch_rows:
            out_df = pd.DataFrame(batch_rows)
            if not os.path.exists(iterative_out_csv):
                out_df.to_csv(iterative_out_csv, index=False)
            else:
                out_df.to_csv(iterative_out_csv, mode="a", header=False, index=False)
            batch_rows = []

    for _, row in df.iterrows():
        qid = row["q_id"]
        lang = row["language"]
        kws = qa_keywords.get((qid, lang), [])
        unique_kws = list(set(kws))

        base_info = {
            "question": row["question"],
            "answer": row["answer"],
            "language": lang,
            "q_id": qid,
        }

        def _empty_features():
            return {
                "q_id": qid,
                "language": lang,
                "cooc_num_pairs": 0,
                "cooc_total_pairs": 0,
                "cooc_coverage_ratio": np.nan,
                "cooc_unseen_keywords_count": 0,
                "cooc_unseen_keywords_ratio": np.nan,
                "cooc_avg_pmi": np.nan,
                "cooc_max_pmi": np.nan,
                "cooc_min_pmi": np.nan,
                "cooc_std_pmi": np.nan,
            }

        if len(unique_kws) < 2:
            feature_row = _empty_features()
            out_row = {**base_info, **feature_row}
            feature_rows.append(feature_row)
            output_rows.append(out_row)
            batch_rows.append(out_row)
            if len(batch_rows) >= batch_size:
                _flush_batch()
            continue

        unigram_counts, cooc_counts, total_docs, pmi_dict = lang_stats.get(
            lang, (Counter(), Counter(), 0, {})
        )

        seen_kws = [kw for kw in unique_kws if kw in unigram_counts]
        unseen_kws = [kw for kw in unique_kws if kw not in unigram_counts]
        num_unseen = len(unseen_kws)
        unseen_ratio = (num_unseen / len(unique_kws)) if unique_kws else np.nan

        pmi_vals = []
        pair_details = []
        for i in range(len(unique_kws)):
            for j in range(i + 1, len(unique_kws)):
                w1, w2 = sorted((unique_kws[i], unique_kws[j]))
                pmi_val = pmi_dict.get((w1, w2))
                if pmi_val is None:
                    continue
                pmi_vals.append(pmi_val)
                pair_details.append({
                    "w1": w1,
                    "w2": w2,
                    "pmi": pmi_val,
                    "cooc_count": cooc_counts.get((w1, w2)),
                    "unigram_w1": unigram_counts.get(w1),
                    "unigram_w2": unigram_counts.get(w2),
                })

        n_k = len(unique_kws)
        total_pairs = n_k * (n_k - 1) / 2 if n_k > 1 else 0
        num_pairs = len(pmi_vals)
        coverage_ratio = (num_pairs / total_pairs) if total_pairs > 0 else np.nan

        if pair_details:
            save_qa_pair_details(qid, lang, pair_details, per_qa_out_dir)

        if pmi_vals:
            arr = np.array(pmi_vals, dtype=float)
            avg_pmi, max_pmi, min_pmi, std_pmi = arr.mean(), arr.max(), arr.min(), arr.std()
        else:
            avg_pmi = max_pmi = min_pmi = std_pmi = np.nan

        feature_row = {
            "q_id": qid,
            "language": lang,
            "cooc_num_pairs": num_pairs,
            "cooc_total_pairs": total_pairs,
            "cooc_coverage_ratio": coverage_ratio,
            "cooc_unseen_keywords_count": num_unseen,
            "cooc_unseen_keywords_ratio": unseen_ratio,
            "cooc_avg_pmi": avg_pmi,
            "cooc_max_pmi": max_pmi,
            "cooc_min_pmi": min_pmi,
            "cooc_std_pmi": std_pmi,
        }
        out_row = {**base_info, **feature_row}
        feature_rows.append(feature_row)
        output_rows.append(out_row)
        batch_rows.append(out_row)
        if len(batch_rows) >= batch_size:
            _flush_batch()

    _flush_batch()
    return pd.DataFrame(output_rows)


# ---------------------------------------------------------------------------
# Entry point  (same CLI as cooc_features.py)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute QA co-occurrence features via Elasticsearch (wimbd)."
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--es_url", default="http://localhost:9200")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "eclektic" / "raw" / "eclektic_7.csv"
    df = pd.read_csv(csv_path)
    df = df.loc[:, ["question", "answer", "language", "q_id"]].drop_duplicates()

    start = args.start_idx or 0
    end = args.end_idx
    df = df.iloc[start:end].reset_index(drop=True)
    print(f"Processing rows {start} to {end if end is not None else 'end'} ({len(df)} rows)")

    qa_keywords = build_language_and_qa_keywords(df)
    save_qa_keywords(qa_keywords)
    print("Extracted keywords for QA pairs.")

    out_csv = project_root / "data" / "processed" / "eclektic_long_with_cooc_features_wimbd.csv"
    os.makedirs(out_csv.parent, exist_ok=True)

    helper = WimbdHelper(es_url=args.es_url)

    merged = build_qa_cooc_features(
        df,
        qa_keywords=qa_keywords,
        helper=helper,
        per_qa_out_dir="./cooc_features/qa_cooc",
        iterative_out_csv=str(out_csv),
    )
    print(f"Saved features to {out_csv}")


if __name__ == "__main__":
    main()
