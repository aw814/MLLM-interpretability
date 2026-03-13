"""
eclektic_cooc_features.py

Behavior summary (updated):
- Co-occurrence queries are performed per QA row only.
- No cross-question (dataset-wide union vocabulary) pairs are queried.
- Query terms for each QA row are built from:
  1) POS/NER token list from pos_tokens.json matched by eclektic_id = "{language}_{q_id}"
  2) the raw answer text as a single phrase
- Duplicate eclektic_id entries in pos_tokens.json are handled by keeping the first match.
"""

import sys
import os
import json
import itertools
import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from wimbd_helper import WimbdHelper
from calculate_co_occurrences import calculate_ppmi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)


def _normalize_term(term: str) -> str:
    return str(term).strip().lower()


def load_pos_tokens_index(path: str) -> dict[str, list[str]]:
    """
    Load pos_tokens.json and return eclektic_id -> tokens.

    If an eclektic_id appears multiple times, the first occurrence is kept.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    idx: dict[str, list[str]] = {}
    for entry in data:
        eclektic_id = str(entry.get("eclektic_id", "")).strip()
        if not eclektic_id or eclektic_id in idx:
            continue

        terms: list[str] = []
        seen: set[str] = set()
        for token_obj in entry.get("tokens", []):
            token = _normalize_term(token_obj.get("token", ""))
            if token and token not in seen:
                seen.add(token)
                terms.append(token)
        idx[eclektic_id] = terms
    return idx


def build_qa_keywords_from_pos_tokens_and_answer(
    df: pd.DataFrame,
    pos_tokens_index: dict[str, list[str]],
    lang_col: str = "language",
    qid_col: str = "q_id",
    answer_col: str = "answer",
) -> dict[tuple[int, str], list[str]]:
    """
    Build QA-local query terms from:
    - pos_tokens.json tokens (matched by eclektic_id = "{lang}_{q_id}")
    - answer as one raw phrase
    """
    qa_keywords: dict[tuple[int, str], list[str]] = {}
    for _, row in df.iterrows():
        lang = str(row[lang_col]).strip()
        qid = row[qid_col]
        eclektic_id = f"{lang}_{qid}"

        terms: list[str] = []
        seen: set[str] = set()

        for token in pos_tokens_index.get(eclektic_id, []):
            if token not in seen:
                seen.add(token)
                terms.append(token)

        answer_phrase = _normalize_term(row[answer_col])
        if answer_phrase and answer_phrase not in seen:
            seen.add(answer_phrase)
            terms.append(answer_phrase)

        qa_keywords[(qid, lang)] = terms
    return qa_keywords


def compute_cooc_for_qa_es(
    lang_code: str,
    keywords: list[str],
    helper: WimbdHelper,
    index_docs_cache: dict[str, int],
) -> tuple[Counter, Counter, int]:
    """
    Query ES unigram/pair document counts for one QA row only.
    """
    if lang_code == "he":
        lang_code = "iw"
    index = f"mc4_{lang_code}"

    if index not in index_docs_cache:
        all_stats = helper.get_index_stats()
        if index not in all_stats:
            logger.warning(f"Index {index} not found in Elasticsearch.")
            index_docs_cache[index] = 0
        else:
            index_docs_cache[index] = int(all_stats[index].get("docs.count", 0))
    total_docs = index_docs_cache[index]
    if total_docs == 0:
        return Counter(), Counter(), 0

    unique_kws = sorted(set(keywords))
    unigram_counts: Counter = Counter()
    cooc_counts: Counter = Counter()

    for kw in unique_kws:
        count = helper.count_phrases(index, [kw])
        if count > 0:
            unigram_counts[kw] = count

    seen_kws = sorted(kw for kw in unique_kws if kw in unigram_counts)
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


def save_qa_pair_details(
    q_id,
    lang,
    query_terms,
    pairs_info,
    out_dir="data/intermediate/eclektic_cooc",
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{lang}_{q_id}_wimbd.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "q_id": q_id,
                "language": lang,
                "query_terms": query_terms,
                "pairs": pairs_info,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def build_qa_cooc_features(
    df: pd.DataFrame,
    qa_keywords: dict,
    helper: WimbdHelper,
    per_qa_out_dir: str = "./cooc_features/eclektic_cooc",
    min_cooc: int = 1,
    iterative_out_csv: Optional[str] = None,
) -> pd.DataFrame:
    if iterative_out_csv is not None:
        iterative_out_csv = str(iterative_out_csv)

    output_rows = []
    batch_rows = []
    batch_size = 12
    index_docs_cache: dict[str, int] = {}

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
        unique_kws = list(dict.fromkeys(kws))

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
            output_rows.append(out_row)
            batch_rows.append(out_row)
            if len(batch_rows) >= batch_size:
                _flush_batch()
            continue

        unigram_counts, cooc_counts, total_docs = compute_cooc_for_qa_es(
            lang_code=lang,
            keywords=unique_kws,
            helper=helper,
            index_docs_cache=index_docs_cache,
        )
        pmi_dict = compute_pmi_matrix(
            unigram_counts=unigram_counts,
            cooc_counts=cooc_counts,
            total_docs=total_docs,
            min_cooc=min_cooc,
        )

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
                pair_details.append(
                    {
                        "w1": w1,
                        "w2": w2,
                        "pmi": pmi_val,
                        "cooc_count": cooc_counts.get((w1, w2)),
                        "unigram_w1": unigram_counts.get(w1),
                        "unigram_w2": unigram_counts.get(w2),
                    }
                )

        n_k = len(unique_kws)
        total_pairs = n_k * (n_k - 1) / 2 if n_k > 1 else 0
        num_pairs = len(pmi_vals)
        coverage_ratio = (num_pairs / total_pairs) if total_pairs > 0 else np.nan

        if pair_details:
            save_qa_pair_details(
                q_id=qid,
                lang=lang,
                query_terms=unique_kws,
                pairs_info=pair_details,
                out_dir=per_qa_out_dir,
            )

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
        output_rows.append(out_row)
        batch_rows.append(out_row)
        if len(batch_rows) >= batch_size:
            _flush_batch()

    _flush_batch()
    return pd.DataFrame(output_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute QA co-occurrence features via Elasticsearch (wimbd)."
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--es_url", default="http://localhost:9200")
    parser.add_argument(
        "--pos_tokens_path",
        default="src/wimbd_cooc_features/pos_ner_keywords/pos_tokens.json",
        help="Path to POS token JSON used for QA-local query terms",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "eclektic" / "raw" / "eclektic_7.csv"
    df = pd.read_csv(csv_path)
    df = df.loc[:, ["question", "answer", "language", "q_id"]].drop_duplicates()

    start = args.start_idx or 0
    end = args.end_idx
    df = df.iloc[start:end].reset_index(drop=True)
    print(f"Processing rows {start} to {end if end is not None else 'end'} ({len(df)} rows)")

    pos_tokens_path = project_root / args.pos_tokens_path
    pos_tokens_index = load_pos_tokens_index(str(pos_tokens_path))
    qa_keywords = build_qa_keywords_from_pos_tokens_and_answer(df, pos_tokens_index)
    print("Built QA query terms from pos_tokens + raw answer phrase.")

    out_csv = project_root / "data" / "eclektic" / "processed" / "eclektic_7_cooc_features.csv"
    os.makedirs(out_csv.parent, exist_ok=True)

    helper = WimbdHelper(es_url=args.es_url)

    build_qa_cooc_features(
        df,
        qa_keywords=qa_keywords,
        helper=helper,
        per_qa_out_dir="./cooc_features/eclektic_cooc",
        iterative_out_csv=str(out_csv),
    )
    print(f"Saved features to {out_csv}")


if __name__ == "__main__":
    main()
