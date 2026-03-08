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
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from wimbd_helper import WimbdHelper
from calculate_co_occurrences import calculate_ppmi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

ALL_LANGS = ["en", "es", "fr", "he", "ja", "ko", "zh"]

# ---------------------------------------------------------------------------
# Per-process ES worker (initializer + query functions)
# ---------------------------------------------------------------------------

_worker_helper = None


def _init_worker(es_url: str):
    global _worker_helper
    _worker_helper = WimbdHelper(es_url=es_url)


def _query_unigram(args):
    index, term = args
    count = _worker_helper.count_phrases(index, [term])
    return term, count


def _query_pair(args):
    index, pair = args
    count = _worker_helper.count_phrases(index, list(pair), all_phrases=True)
    return pair, count

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
    path = os.path.join(out_dir, f"{lang}_{fact_index}_klar.json")
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
) -> tuple[str, list[dict]]:
    """Query mc4_{lang} for all unique triples in this language."""
    index_lang = "iw" if lang == "he" else lang
    index = f"mc4_{index_lang}"

    helper = WimbdHelper(es_url=es_url)
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

    with Pool(processes=workers, initializer=_init_worker, initargs=(es_url,)) as pool:
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

def load_klar(data_dir: str, langs: list[str]) -> pd.DataFrame:
    frames = []
    for lang in langs:
        path = os.path.join(data_dir, f"{lang}.csv")
        if not os.path.exists(path):
            logger.warning(f"CSV not found: {path}, skipping.")
            continue
        df = pd.read_csv(path)
        df["language"] = lang
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


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
    parser.add_argument("--langs", default=",".join(ALL_LANGS),
                        help="Comma-separated language codes (default: all)")
    parser.add_argument("--data_dir", default="data/KLAR/raw",
                        help="Directory with KLAR CSV files")
    parser.add_argument("--es_url", default="http://localhost:9200")
    parser.add_argument("--min_cooc", type=int, default=1,
                        help="Minimum co-occurrence count to compute PPMI")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--per_triple_out_dir", default="cooc_features/klar_cooc",
                        help="Directory for per-triple co-occurrence JSON files")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel ES query workers per language")
    args = parser.parse_args()
    args.workers = max(32, args.workers)

    langs = [l.strip() for l in args.langs.split(",")]

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / args.data_dir

    df = load_klar(str(data_dir), langs)

    start = args.start_idx or 0
    end = args.end_idx
    df = df.iloc[start:end].reset_index(drop=True)
    actual_end = start + len(df)
    logger.info(f"Processing rows {start} to {actual_end} ({len(df)} rows)")

    langs_str = "_".join(langs)
    out_csv = project_root / "data" / "KLAR" / "processed" / f"klar_cooc_wimbd_{langs_str}_{start}_to_{actual_end}.csv"
    os.makedirs(out_csv.parent, exist_ok=True)

    lang_triples = build_triples(df)

    print("TIP: Consider running in tmux: tmux new -s klar_cooc")

    all_rows = []
    for lang, triples in lang_triples.items():
        lang, rows = query_lang(
            lang, triples, args.es_url, args.min_cooc,
            workers=args.workers,
            per_triple_out_dir=args.per_triple_out_dir,
        )
        logger.info(f"[{lang}] Got {len(rows)} result rows")
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(out_df)} rows to {out_csv}")


if __name__ == "__main__":
    main()