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
from functools import lru_cache

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
# Relation simplification and translation
# ---------------------------------------------------------------------------

RELATION_MAPPING = {
    "applies_to_jurisdiction": "jurisdiction",
    "capital_of": "capital",
    "country_of_citizenship": "citizenship",
    "field_of_work": "work",
    "headquarters_location": "location",
    "language_of_work_or_name": "language",
    "languages_spoken": "spoke",
    "location_of_formation": "location",
    "native_language": "language",
    "official_language": "language",
    "owned_by": "owns",
    "place_of_birth": "born",
    "place_of_death": "died",
}

# Translation cache: {(term, lang): translated_term}
TRANSLATION = {
    # jurisdiction
    ("jurisdiction", "en"): "jurisdiction",
    ("jurisdiction", "es"): "jurisdicción",
    ("jurisdiction", "fr"): "juridiction",
    ("jurisdiction", "ja"): "管轄",
    ("jurisdiction", "ko"): "관할권",
    ("jurisdiction", "zh"): "司法管辖权",
    ("jurisdiction", "he"): "משפטי",
    
    # capital
    ("capital", "en"): "capital",
    ("capital", "es"): "capital",
    ("capital", "fr"): "capitale",
    ("capital", "ja"): "首都",
    ("capital", "ko"): "수도",
    ("capital", "zh"): "首都",
    ("capital", "he"): "עיר בירה",
    
    # citizenship
    ("citizenship", "en"): "citizenship",
    ("citizenship", "es"): "ciudadanía",
    ("citizenship", "fr"): "citoyenneté",
    ("citizenship", "ja"): "市民権",
    ("citizenship", "ko"): "국적",
    ("citizenship", "zh"): "公民身份",
    ("citizenship", "he"): "אזרחות",
    
    # work
    ("work", "en"): "work",
    ("work", "es"): "trabajo",
    ("work", "fr"): "travail",
    ("work", "ja"): "仕事",
    ("work", "ko"): "작업",
    ("work", "zh"): "工作",
    ("work", "he"): "עבודה",
    
    # location
    ("location", "en"): "location",
    ("location", "es"): "ubicación",
    ("location", "fr"): "localisation",
    ("location", "ja"): "場所",
    ("location", "ko"): "위치",
    ("location", "zh"): "位置",
    ("location", "he"): "מיקום",
    
    # language
    ("language", "en"): "language",
    ("language", "es"): "idioma",
    ("language", "fr"): "langue",
    ("language", "ja"): "言語",
    ("language", "ko"): "언어",
    ("language", "zh"): "语言",
    ("language", "he"): "שפה",
    
    # spoke
    ("spoke", "en"): "spoke",
    ("spoke", "es"): "habló",
    ("spoke", "fr"): "parlait",
    ("spoke", "ja"): "話した",
    ("spoke", "ko"): "말했다",
    ("spoke", "zh"): "说话",
    ("spoke", "he"): "דיבר",
    
    # owns
    ("owns", "en"): "owns",
    ("owns", "es"): "posee",
    ("owns", "fr"): "possède",
    ("owns", "ja"): "所有する",
    ("owns", "ko"): "소유",
    ("owns", "zh"): "拥有",
    ("owns", "he"): "שייך",
    
    # born
    ("born", "en"): "born",
    ("born", "es"): "nacido",
    ("born", "fr"): "né",
    ("born", "ja"): "生まれた",
    ("born", "ko"): "태어난",
    ("born", "zh"): "出生",
    ("born", "he"): "נולד",
    
    # died
    ("died", "en"): "died",
    ("died", "es"): "murió",
    ("died", "fr"): "mort",
    ("died", "ja"): "死亡した",
    ("died", "ko"): "죽었다",
    ("died", "zh"): "去世",
    ("died", "he"): "מת",

    # continent
    ("continent", "en"): "continent",
    ("continent", "es"): "continente",
    ("continent", "fr"): "continent",
    ("continent", "ja"): "大陸",
    ("continent", "ko"): "대륙",
    ("continent", "zh"): "大陆",
    ("continent", "he"): "יבשת",

    # developer
    ("developer", "en"): "developer",
    ("developer", "es"): "desarrollador",
    ("developer", "fr"): "développeur",
    ("developer", "ja"): "開発者",
    ("developer", "ko"): "개발자",
    ("developer", "zh"): "开发者",
    ("developer", "he"): "מפתח",

    # instrument
    ("instrument", "en"): "instrument",
    ("instrument", "es"): "instrumento",
    ("instrument", "fr"): "instrument",
    ("instrument", "ja"): "楽器",
    ("instrument", "ko"): "악기",
    ("instrument", "zh"): "乐器",
    ("instrument", "he"): "כלי נגינה",

    # manufacturer
    ("manufacturer", "en"): "manufacturer",
    ("manufacturer", "es"): "fabricante",
    ("manufacturer", "fr"): "fabricant",
    ("manufacturer", "ja"): "製造元",
    ("manufacturer", "ko"): "제조사",
    ("manufacturer", "zh"): "制造商",
    ("manufacturer", "he"): "יצרן",

    # occupation
    ("occupation", "en"): "occupation",
    ("occupation", "es"): "ocupación",
    ("occupation", "fr"): "profession",
    ("occupation", "ja"): "職業",
    ("occupation", "ko"): "직업",
    ("occupation", "zh"): "职业",
    ("occupation", "he"): "עיסוק",

    # religion
    ("religion", "en"): "religion",
    ("religion", "es"): "religión",
    ("religion", "fr"): "religion",
    ("religion", "ja"): "宗教",
    ("religion", "ko"): "종교",
    ("religion", "zh"): "宗教",
    ("religion", "he"): "דת",
}

# Query result cache: {(index, term, lang): count}
_es_query_cache = {}


def simplify_relation(relation: str) -> str:
    """Map complex relation names to simpler versions."""
    return RELATION_MAPPING.get(relation, relation)


def get_translated_term(term: str, lang: str) -> str:
    """Get cached translation of a term for a given language."""
    return TRANSLATION.get((term, lang), term)

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
    cache_key = (index, term, "unigram")
    if cache_key in _es_query_cache:
        return term, _es_query_cache[cache_key]
    
    for attempt in range(_worker_retries + 1):
        try:
            count = _worker_helper.count_phrases(index, [term])
            _es_query_cache[cache_key] = count
            return term, count
        except ConnectionTimeout:
            if attempt == _worker_retries:
                raise
            time.sleep(_worker_retry_sleep * (2 ** attempt))


def _query_pair(args):
    index, pair = args
    cache_key = (index, pair, "pair")
    if cache_key in _es_query_cache:
        return pair, _es_query_cache[cache_key]
    
    for attempt in range(_worker_retries + 1):
        try:
            count = _worker_helper.count_phrases(index, list(pair), all_phrases=True)
            _es_query_cache[cache_key] = count
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

    # Collect unique terms and pairs, translating relations to target language
    unique_terms: set[str] = set()
    unique_pairs_set: set[tuple[str, str]] = set()
    relation_translations: dict[str, str] = {}  # Map original relation to translated
    
    for t in triples:
        # Translate relation to target language
        original_relation = t["relation"]
        if original_relation not in relation_translations:
            relation_translations[original_relation] = get_translated_term(original_relation, lang)
        translated_relation = relation_translations[original_relation]
        
        # Add terms (subject/object as-is, relation translated)
        unique_terms.add(t["subject"])
        unique_terms.add(translated_relation)
        unique_terms.add(t["object"])
        
        # Add pairs with translated relation
        for r1, r2 in PAIRS:
            term1 = t[r1]
            term2 = translated_relation if r2 == "relation" else t[r2]
            unique_pairs_set.add((term1, term2))

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
        # Get translated relation for this triple
        translated_relation = relation_translations[t["relation"]]
        
        row = {
            "language": lang,
            "subject": t["subject"],
            "relation": translated_relation,
            "object": t["object"],
            "fact_index": t["fact_index"],
            "total_docs": total_docs,
            "count_subject": unigram_counts.get(t["subject"], 0),
            "count_relation": unigram_counts.get(translated_relation, 0),
            "count_object": unigram_counts.get(t["object"], 0),
        }
        pair_details = []
        pmi_vals = []
        for r1, r2 in PAIRS:
            # Use translated relation in pairs (can be in position r1 or r2)
            term1 = translated_relation if r1 == "relation" else t[r1]
            term2 = translated_relation if r2 == "relation" else t[r2]
            cooc = unique_pairs.get((term1, term2), 0)
            n1 = unigram_counts.get(term1, 0)
            n2 = unigram_counts.get(term2, 0)
            ppmi = calculate_ppmi(cooc, n1, n2, total_docs) if cooc >= min_cooc else 0.0
            col = f"{r1}_{r2}"
            row[f"cooc_{col}"] = cooc
            row[f"ppmi_{col}"] = ppmi
            if ppmi > 0.0:
                pmi_vals.append(ppmi)
            pair_details.append({
                "w1": term1,
                "w2": term2,
                "role1": r1,
                "role2": r2,
                "cooc_count": cooc,
                "unigram_w1": n1,
                "unigram_w2": n2,
                "ppmi": ppmi,
            })

        roles = ("subject", "relation", "object")
        unseen = sum(1 for r in roles if unigram_counts.get(
            t[r] if r != "relation" else translated_relation, 0) == 0)
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
    """Return unique (subject, relation, object, fact_index) per language.
    
    Relations are simplified using RELATION_MAPPING (e.g., applies_to_jurisdiction -> jurisdiction).
    """
    lang_triples: dict[str, list[dict]] = {}
    for lang, group in df.groupby("language"):
        seen = set()
        triples = []
        for _, row in group.iterrows():
            simplified_relation = simplify_relation(row["relation"])
            key = (row["subject"], simplified_relation, row["object"], row["index"])
            if key not in seen:
                seen.add(key)
                triples.append({
                    "subject": row["subject"],
                    "relation": simplified_relation,
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
    parser.add_argument("--per_triple_out_dir", default="cooc_features/klar_cooc/after_relation_change",
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
        / f"klar_cooc_wimbd_{start}_to_{actual_end}_new.csv"
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
