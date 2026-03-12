"""
klar_cooc_features.py

For each unique (subject, relation, object) triple in the KLAR dataset,
queries the corresponding klar_{lang} Elasticsearch index for the three
fixed phrase pairs:
    (subject, relation), (relation, object), (subject, object)

Computes PPMI for each pair and writes a feature CSV.
"""

import os
import sys
import json
import argparse
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from elastic_transport import ConnectionTimeout

sys.path.insert(0, os.path.dirname(__file__))
from wimbd_helper import WimbdHelper
from calculate_co_occurrences import calculate_ppmi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Per-process ES worker (initializer + query functions)
# ---------------------------------------------------------------------------

_worker_helper = None
_worker_retries = 2
_worker_retry_sleep = 1.0


def _init_worker(
    es_url: str,
    request_timeout: int,
    max_retries: int,
    query_retries: int,
    retry_sleep: float,
):
    global _worker_helper, _worker_retries, _worker_retry_sleep
    _worker_retries = max(0, query_retries)
    _worker_retry_sleep = max(0.0, retry_sleep)
    _worker_helper = WimbdHelper(
        es_url=es_url,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )


def _query_unigram(args):
    index, term = args
    for attempt in range(_worker_retries + 1):
        try:
            count = _worker_helper.count_phrases(index, [term])
            return term, count
        except ConnectionTimeout:
            if attempt == _worker_retries:
                raise
            time.sleep(_worker_retry_sleep * (2 ** attempt))


def _query_pair(args):
    index, pair = args
    for attempt in range(_worker_retries + 1):
        try:
            count = _worker_helper.count_phrases(index, list(pair), all_phrases=True)
            return pair, count
        except ConnectionTimeout:
            if attempt == _worker_retries:
                raise
            time.sleep(_worker_retry_sleep * (2 ** attempt))

PAIRS = [
    ("subject", "relation"),
    ("relation", "object"),
    ("subject", "object"),
]


# ---------------------------------------------------------------------------
# Per-triple JSON saving
# ---------------------------------------------------------------------------

def save_triple_details(fact_index, lang, pair_details, out_dir="cooc_features/klar_cooc"):
    os.makedirs(out_dir, exist_ok=True)
    safe_fact_index = str(fact_index).replace(os.sep, "_")
    path = os.path.join(out_dir, f"{safe_fact_index}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pair_details, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# ES queries
# ---------------------------------------------------------------------------

def query_lang(
    lang: str,
    triples: list[dict],
    es_url: str,
    min_cooc: int,
    workers: int = 32,
    per_triple_out_dir: str = "cooc_features/klar_cooc",
    es_timeout: int = 60,
    es_client_retries: int = 3,
    query_retries: int = 2,
    retry_sleep: float = 1.0,
) -> tuple[str, list[dict]]:
    """Query mc4_{lang} for all unique triples in this language."""
    index_lang = "iw" if lang == "he" else lang
    index = f"mc4_{index_lang}"

    helper = WimbdHelper(
        es_url=es_url,
        request_timeout=es_timeout,
        max_retries=es_client_retries,
    )
    all_stats = helper.get_index_stats()
    if index not in all_stats:
        logger.warning(f"Index {index} not found.")
        return lang, []

    total_docs = int(all_stats[index].get("docs.count", 0))
    if total_docs == 0:
        logger.warning(f"Index {index} has 0 documents.")
        return lang, []

    # Collect unique terms and pairs
    unique_terms: set[str] = set()
    unique_pairs_set: set[tuple[str, str]] = set()
    for t in triples:
        for role in ("subject", "relation", "object"):
            unique_terms.add(t[role])
        for r1, r2 in PAIRS:
            unique_pairs_set.add((t[r1], t[r2]))

    all_terms = list(unique_terms)
    all_pairs = list(unique_pairs_set)
    logger.info(f"[{index}] {len(all_terms)} unique terms, {len(all_pairs)} unique pairs, {workers} workers")

    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(es_url, es_timeout, es_client_retries, query_retries, retry_sleep),
    ) as pool:
        unigram_counts: dict[str, int] = {}
        for term, count in tqdm(
            pool.imap_unordered(_query_unigram, [(index, t) for t in all_terms]),
            total=len(all_terms),
            desc=f"[{index}] unigrams",
        ):
            if count > 0:
                unigram_counts[term] = count

        unique_pairs: dict[tuple[str, str], int] = {}
        for pair, count in tqdm(
            pool.imap_unordered(_query_pair, [(index, p) for p in all_pairs]),
            total=len(all_pairs),
            desc=f"[{index}] pairs",
        ):
            unique_pairs[pair] = count

    rows = []
    for t in triples:
        row = {
            "language": lang,
            "subject": t["subject"],
            "relation": t["relation"],
            "object": t["object"],
            "fact_index": t["fact_index"],
            "total_docs": total_docs,
            "count_subject": unigram_counts.get(t["subject"], 0),
            "count_relation": unigram_counts.get(t["relation"], 0),
            "count_object": unigram_counts.get(t["object"], 0),
        }
        pair_details = []
        pmi_vals = []
        for r1, r2 in PAIRS:
            cooc = unique_pairs.get((t[r1], t[r2]), 0)
            n1 = unigram_counts.get(t[r1], 0)
            n2 = unigram_counts.get(t[r2], 0)
            ppmi = calculate_ppmi(cooc, n1, n2, total_docs) if cooc >= min_cooc else 0.0
            col = f"{r1}_{r2}"
            row[f"cooc_{col}"] = cooc
            row[f"ppmi_{col}"] = ppmi
            if ppmi > 0.0:
                pmi_vals.append(ppmi)
            pair_details.append({
                "w1": t[r1],
                "w2": t[r2],
                "role1": r1,
                "role2": r2,
                "cooc_count": cooc,
                "unigram_w1": n1,
                "unigram_w2": n2,
                "ppmi": ppmi,
            })

        roles = ("subject", "relation", "object")
        unseen = sum(1 for r in roles if unigram_counts.get(t[r], 0) == 0)
        num_pairs = len(pmi_vals)
        row["cooc_total_pairs"] = 3
        row["cooc_num_pairs"] = num_pairs
        row["cooc_coverage_ratio"] = num_pairs / 3
        row["cooc_unseen_keywords_count"] = unseen
        row["cooc_unseen_keywords_ratio"] = unseen / 3
        if pmi_vals:
            arr = np.array(pmi_vals)
            row["cooc_avg_pmi"] = arr.mean()
            row["cooc_max_pmi"] = arr.max()
            row["cooc_min_pmi"] = arr.min()
            row["cooc_std_pmi"] = arr.std()
        else:
            row["cooc_avg_pmi"] = row["cooc_max_pmi"] = row["cooc_min_pmi"] = row["cooc_std_pmi"] = float("nan")

        save_triple_details(t["fact_index"], lang, pair_details, out_dir=per_triple_out_dir)
        rows.append(row)

    return lang, rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_combined_klar(data_file: str) -> pd.DataFrame:
    df = pd.read_csv(data_file)
    if "target_lang" in df.columns:
        df["language"] = df["target_lang"]
    if "klar_id" in df.columns:
        df["index"] = df["klar_id"]
    elif "q_index" in df.columns:
        df["index"] = df["q_index"]

    required = {"language", "index", "subject", "relation", "object"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Combined KLAR source is missing required columns: {sorted(missing)}"
        )

    return df


def build_triples(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Return unique (subject, relation, object, fact_index) per language."""
    lang_triples: dict[str, list[dict]] = {}
    for lang, group in df.groupby("language"):
        seen = set()
        triples = []
        for _, row in group.iterrows():
            key = (row["subject"], row["relation"], row["object"], row["index"])
            if key not in seen:
                seen.add(key)
                triples.append({
                    "subject": row["subject"],
                    "relation": row["relation"],
                    "object": row["object"],
                    "fact_index": row["index"],
                })
        lang_triples[lang] = triples
        logger.info(f"[{lang}] {len(triples)} unique triples")
    return lang_triples


def main():
    parser = argparse.ArgumentParser(
        description="Compute KLAR co-occurrence features via Elasticsearch."
    )
    parser.add_argument(
        "--data_file",
        default="data/KLAR/raw/combined/klar_7_cleaned.csv",
        help="Combined KLAR CSV file path",
    )
    parser.add_argument("--es_url", default="http://localhost:9200")
    parser.add_argument("--min_cooc", type=int, default=1,
                        help="Minimum co-occurrence count to compute PPMI")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--per_triple_out_dir", default="cooc_features/klar_cooc",
                        help="Directory for per-triple co-occurrence JSON files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel ES query workers per language")
    parser.add_argument(
        "--es_timeout",
        type=int,
        default=180,
        help="Elasticsearch request timeout in seconds per query",
    )
    parser.add_argument(
        "--es_client_retries",
        type=int,
        default=5,
        help="Elasticsearch client retry attempts on timeout",
    )
    parser.add_argument(
        "--query_retries",
        type=int,
        default=5,
        help="Extra retry attempts around each unigram/pair query",
    )
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=1.0,
        help="Base retry sleep (seconds), applied with exponential backoff",
    )
    args = parser.parse_args()
    args.workers = max(1, args.workers)

    project_root = Path(__file__).resolve().parents[2]
    data_file = project_root / args.data_file

    df = load_combined_klar(str(data_file))

    start = args.start_idx or 0
    end = args.end_idx
    df = df.iloc[start:end].reset_index(drop=True)
    actual_end = start + len(df)
    logger.info(f"Processing rows {start} to {actual_end} ({len(df)} rows)")
    if df.empty:
        logger.warning("No rows selected for processing. Exiting.")
        return

    out_csv = (
        project_root
        / "data"
        / "KLAR"
        / "processed"
        / f"klar_cooc_wimbd_{start}_to_{actual_end}.csv"
    )
    os.makedirs(out_csv.parent, exist_ok=True)

    lang_triples = build_triples(df)

    print("TIP: Consider running in tmux: tmux new -s klar_cooc")

    all_rows = []
    for lang, triples in lang_triples.items():
        lang, rows = query_lang(
            lang, triples, args.es_url, args.min_cooc,
            workers=args.workers,
            per_triple_out_dir=args.per_triple_out_dir,
            es_timeout=args.es_timeout,
            es_client_retries=args.es_client_retries,
            query_retries=args.query_retries,
            retry_sleep=args.retry_sleep,
        )
        logger.info(f"[{lang}] Got {len(rows)} result rows")
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(out_df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
