from collections import defaultdict,Counter
import re
import pandas as pd
import os
import json
import itertools
from datasets import load_dataset
from tokenizers import get_language_config, get_stopwords_for_lang
import argparse
from pathlib import Path


def tokenize(text: str, lang: str | None = None) -> list[str]:
    """
    Tokenize text into word-like units using the same language-specific
    configuration as the bag-of-words feature generation.
    If a language-specific tokenizer is available (e.g., jieba for zh,
    Sudachi for ja, KoNLPy for ko, etc.), it will be used; otherwise we
    fall back to a simple regex-based word tokenizer.
    """
    if lang:
        tokenizer_fn, _, _ = get_language_config(lang)
        if tokenizer_fn is not None:
            return [t.lower() for t in tokenizer_fn(text)]
    # Fallback: simple word-based tokenization for languages without a custom tokenizer
    return re.findall(r"\w+", str(text).lower())

def extract_keywords_from_text(
    text: str,
    stopwords: set[str] | None = None,
    max_per_text: int | None = None,
    lang: str | None = None,
) -> list[str]:
    toks = tokenize(text, lang=lang)
    stopwords = get_stopwords_for_lang(lang or "")
    if stopwords is None:
        stopwords = set()
    content = [t for t in toks if t not in stopwords]
    seen = set()
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


def save_qa_keywords(qa_keywords, path="./cooc_features/qa_keywords.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert tuple keys to strings
    serializable = {f"{qid}_{lang}": kws for (qid, lang), kws in qa_keywords.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_qa_keywords(path="./cooc_features/qa_keywords.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert back to tuple keys
    return {tuple(k.split("_", 1)): v for k, v in data.items()}




def sliding_windows(tokens, window_size: int):
    if len(tokens) <= window_size:
        if tokens:
            yield tokens
        return
    for i in range(0, len(tokens) - window_size + 1):
        yield tokens[i:i + window_size]

def compute_cooc_for_language(
    lang_code: str,
    keyword_vocab: set[str],
    window_size: int = 20,
    max_docs: int = 100,
):
    if lang_code == "he":
        lang_code = "iw"  # Dataset uses 'iw' for Hebrew
    # ds = load_dataset("allenai/c4", lang_code, streaming=True)
    # # ds = load_dataset("mc4", lang_code, streaming=True)
    # train_iter = iter(ds["train"])


    train_iter = load_dataset("bertin-project/mc4-sampling", lang_code, split="train", streaming=True, sampling_method="random")

    key_set = set(keyword_vocab)
    unigram_counts = Counter()
    cooc_counts = Counter()
    total_windows = 0

    from tqdm import tqdm
    for idx, ex in enumerate(tqdm(train_iter, total=max_docs, desc=f"{lang_code} docs")):
        if idx >= max_docs:
            break

        text = ex.get("text", "")
        toks = tokenize(text, lang=lang_code)
        if not toks:
            continue

        for win in sliding_windows(toks, window_size):
            win_keywords = [w for w in win if w in key_set]
            if len(win_keywords) < 2:
                continue

            total_windows += 1
            unigram_counts.update(win_keywords)

            unique_kws = sorted(set(win_keywords))
            for w1, w2 in itertools.combinations(unique_kws, 2):
                cooc_counts[(w1, w2)] += 1

    return unigram_counts, cooc_counts, total_windows

def compute_cooc_for_language_multi(
    lang_code: str,
    keyword_vocab: set[str],
    window_sizes: list[int],
    max_docs: int = 100,
):
    if lang_code == "he":
        lang_code = "iw"  # Dataset uses 'iw' for Hebrew

    train_iter = load_dataset("bertin-project/mc4-sampling", lang_code, split="train", streaming=True, sampling_method="random")

    key_set = set(keyword_vocab)
    # For each window size, track unigram, cooc, and total_windows
    stats = {
        w: {
            "unigram_counts": Counter(),
            "cooc_counts": Counter(),
            "total_windows": 0,
        }
        for w in window_sizes
    }

    from tqdm import tqdm
    for idx, ex in enumerate(tqdm(train_iter, total=max_docs, desc=f"{lang_code} docs")):
        if idx >= max_docs:
            break

        text = ex.get("text", "")
        toks = tokenize(text, lang=lang_code)
        if not toks:
            continue

        # For each window size, compute windows and update corresponding stats
        for w in window_sizes:
            s = stats[w]
            for win in sliding_windows(toks, w):
                win_keywords = [token for token in win if token in key_set]
                if len(win_keywords) < 2:
                    continue

                s["total_windows"] += 1
                s["unigram_counts"].update(win_keywords)

                unique_kws = sorted(set(win_keywords))
                for w1, w2 in itertools.combinations(unique_kws, 2):
                    s["cooc_counts"][(w1, w2)] += 1

    # Return mapping from window size to (unigram_counts, cooc_counts, total_windows)
    return {
        w: (
            stats[w]["unigram_counts"],
            stats[w]["cooc_counts"],
            stats[w]["total_windows"],
        )
        for w in window_sizes
    }

def compute_pmi_matrix(unigram_counts, cooc_counts, total_windows, min_cooc=1):
    """
    Compute PPMI instead of PMI:
        PPMI = max(PMI, 0)
    """
    import math
    pmi = {}
    for (w1, w2), cooc_count in cooc_counts.items():
        if cooc_count < min_cooc:
            continue

        uni_w1 = unigram_counts.get(w1, 0)
        uni_w2 = unigram_counts.get(w2, 0)
        if uni_w1 == 0 or uni_w2 == 0:
            continue

        p_w1 = uni_w1 / total_windows
        p_w2 = uni_w2 / total_windows
        p_w1_w2 = cooc_count / total_windows

        if p_w1_w2 <= 0 or p_w1 <= 0 or p_w2 <= 0:
            continue

        # Standard PMI
        pmi_val = math.log(p_w1_w2 / (p_w1 * p_w2))

        # Convert to PPMI
        ppmi_val = max(pmi_val, 0.0)

        pmi[(w1, w2)] = ppmi_val

    return pmi


def save_qa_pair_details(q_id, lang, pairs_info, out_dir="data/intermediate/qa_cooc"):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{lang}_{q_id}.json"
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs_info, f, ensure_ascii=False, indent=2)

import numpy as np

def build_qa_cooc_features(
    df,
    qa_keywords,
    per_qa_out_dir="./cooc_features/qa_cooc",
    window_sizes=None,
    max_docs=100000,
    min_cooc=1,
    iterative_out_csv: str | None = None,
):
    if window_sizes is None:
        window_sizes = [50]
    elif isinstance(window_sizes, int):
        window_sizes = [window_sizes]
    else:
        window_sizes = list(window_sizes)
    window_sizes = sorted(set(window_sizes))

    if iterative_out_csv is not None:
        iterative_out_csv = str(iterative_out_csv)

    feature_rows = []
    output_rows = []

    batch_rows = []
    batch_size = 12

    for _, row in df.iterrows():
        qid = row["q_id"]
        lang = row["language"]
        kws = qa_keywords.get((qid, lang), [])
        unique_kws = list(set(kws))

        if len(unique_kws) < 2:
            feature_row = {
                "q_id": qid,
                "language": lang,
            }
            default_w = window_sizes[0]
            for w in window_sizes:
                suffix = f"_w{w}" if len(window_sizes) > 1 else ""
                feature_row[f"cooc_num_pairs{suffix}"] = 0
                feature_row[f"cooc_total_pairs{suffix}"] = 0
                feature_row[f"cooc_coverage_ratio{suffix}"] = np.nan
                feature_row[f"cooc_unseen_keywords_count{suffix}"] = 0
                feature_row[f"cooc_unseen_keywords_ratio{suffix}"] = np.nan
                feature_row[f"cooc_avg_pmi{suffix}"] = np.nan
                feature_row[f"cooc_max_pmi{suffix}"] = np.nan
                feature_row[f"cooc_min_pmi{suffix}"] = np.nan
                feature_row[f"cooc_std_pmi{suffix}"] = np.nan

            if len(window_sizes) > 1:
                feature_row["cooc_num_pairs"] = feature_row[f"cooc_num_pairs_w{default_w}"]
                feature_row["cooc_total_pairs"] = feature_row[f"cooc_total_pairs_w{default_w}"]
                feature_row["cooc_coverage_ratio"] = feature_row[f"cooc_coverage_ratio_w{default_w}"]
                feature_row["cooc_unseen_keywords_count"] = feature_row[f"cooc_unseen_keywords_count_w{default_w}"]
                feature_row["cooc_unseen_keywords_ratio"] = feature_row[f"cooc_unseen_keywords_ratio_w{default_w}"]
                feature_row["cooc_avg_pmi"] = feature_row[f"cooc_avg_pmi_w{default_w}"]
                feature_row["cooc_max_pmi"] = feature_row[f"cooc_max_pmi_w{default_w}"]
                feature_row["cooc_min_pmi"] = feature_row[f"cooc_min_pmi_w{default_w}"]
                feature_row["cooc_std_pmi"] = feature_row[f"cooc_std_pmi_w{default_w}"]

            base_info = {
                "question": row["question"],
                "answer": row["answer"],
                "language": lang,
                "q_id": qid,
            }
            out_row = {**base_info, **feature_row}
            feature_rows.append(feature_row)
            output_rows.append(out_row)
            batch_rows.append(out_row)
            if iterative_out_csv is not None and len(batch_rows) >= batch_size:
                out_df = pd.DataFrame(batch_rows)
                if not os.path.exists(iterative_out_csv):
                    out_df.to_csv(iterative_out_csv, index=False)
                else:
                    out_df.to_csv(iterative_out_csv, mode="a", header=False, index=False)
                batch_rows = []
            continue

        # Compute co-occurrence stats for one or many window sizes
        stats_per_w = {}
        if len(window_sizes) == 1:
            w = window_sizes[0]
            unigram_counts, cooc_counts, total_windows = compute_cooc_for_language(
                lang_code=lang,
                keyword_vocab=set(unique_kws),
                window_size=w,
                max_docs=max_docs,
            )
            pmi_dict = compute_pmi_matrix(unigram_counts, cooc_counts, total_windows, min_cooc=min_cooc)
            stats_per_w[w] = (unigram_counts, cooc_counts, total_windows, pmi_dict)
        else:
            multi_stats = compute_cooc_for_language_multi(
                lang_code=lang,
                keyword_vocab=set(unique_kws),
                window_sizes=window_sizes,
                max_docs=max_docs,
            )
            for w, (unigram_counts, cooc_counts, total_windows) in multi_stats.items():
                pmi_dict = compute_pmi_matrix(unigram_counts, cooc_counts, total_windows, min_cooc=min_cooc)
                stats_per_w[w] = (unigram_counts, cooc_counts, total_windows, pmi_dict)

        feature_row = {
            "q_id": qid,
            "language": lang,
        }

        # For backward compatibility, treat the first window size as "default"
        default_w = window_sizes[0]

        # For each window size, compute PPMI-based summary features
        for w in window_sizes:
            unigram_counts, cooc_counts, total_windows, pmi_dict = stats_per_w[w]

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

                    cooc_count = cooc_counts.get((w1, w2))
                    uni_w1 = unigram_counts.get(w1)
                    uni_w2 = unigram_counts.get(w2)

                    pmi_vals.append(pmi_val)
                    pair_details.append({
                        "w1": w1,
                        "w2": w2,
                        "pmi": pmi_val,
                        "cooc_count": cooc_count,
                        "unigram_w1": uni_w1,
                        "unigram_w2": uni_w2,
                        "window_size": w,
                    })

            num_pairs = len(pmi_vals)
            n_k = len(unique_kws)
            total_pairs = n_k * (n_k - 1) / 2 if n_k > 1 else 0
            coverage_ratio = (num_pairs / total_pairs) if total_pairs > 0 else np.nan

            # Only save per-QA pair details for the default window size to avoid explosion in files
            if w == default_w and pair_details:
                save_qa_pair_details(qid, lang, pair_details, per_qa_out_dir)

            if pmi_vals:
                arr = np.array(pmi_vals, dtype=float)
                avg_pmi = arr.mean()
                max_pmi = arr.max()
                min_pmi = arr.min()
                std_pmi = arr.std()
            else:
                avg_pmi = max_pmi = min_pmi = std_pmi = np.nan

            # Add suffix for window-specific features if more than one window size is used
            suffix = f"_w{w}" if len(window_sizes) > 1 else ""

            feature_row[f"cooc_num_pairs{suffix}"] = num_pairs
            feature_row[f"cooc_total_pairs{suffix}"] = total_pairs
            feature_row[f"cooc_coverage_ratio{suffix}"] = coverage_ratio
            feature_row[f"cooc_unseen_keywords_count{suffix}"] = num_unseen
            feature_row[f"cooc_unseen_keywords_ratio{suffix}"] = unseen_ratio
            feature_row[f"cooc_avg_pmi{suffix}"] = avg_pmi
            feature_row[f"cooc_max_pmi{suffix}"] = max_pmi
            feature_row[f"cooc_min_pmi{suffix}"] = min_pmi
            feature_row[f"cooc_std_pmi{suffix}"] = std_pmi

        # For backward compatibility, also expose the default window size without suffix
        if len(window_sizes) > 1:
            feature_row["cooc_num_pairs"] = feature_row[f"cooc_num_pairs_w{default_w}"]
            feature_row["cooc_total_pairs"] = feature_row[f"cooc_total_pairs_w{default_w}"]
            feature_row["cooc_coverage_ratio"] = feature_row[f"cooc_coverage_ratio_w{default_w}"]
            feature_row["cooc_unseen_keywords_count"] = feature_row[f"cooc_unseen_keywords_count_w{default_w}"]
            feature_row["cooc_unseen_keywords_ratio"] = feature_row[f"cooc_unseen_keywords_ratio_w{default_w}"]
            feature_row["cooc_avg_pmi"] = feature_row[f"cooc_avg_pmi_w{default_w}"]
            feature_row["cooc_max_pmi"] = feature_row[f"cooc_max_pmi_w{default_w}"]
            feature_row["cooc_min_pmi"] = feature_row[f"cooc_min_pmi_w{default_w}"]
            feature_row["cooc_std_pmi"] = feature_row[f"cooc_std_pmi_w{default_w}"]

        base_info = {
            "question": row["question"],
            "answer": row["answer"],
            "language": lang,
            "q_id": qid,
        }
        out_row = {**base_info, **feature_row}
        feature_rows.append(feature_row)
        output_rows.append(out_row)
        batch_rows.append(out_row)
        if iterative_out_csv is not None and len(batch_rows) >= batch_size:
            out_df = pd.DataFrame(batch_rows)
            if not os.path.exists(iterative_out_csv):
                out_df.to_csv(iterative_out_csv, index=False)
            else:
                out_df.to_csv(iterative_out_csv, mode="a", header=False, index=False)
            batch_rows = []

    if iterative_out_csv is not None and batch_rows:
        out_df = pd.DataFrame(batch_rows)
        if not os.path.exists(iterative_out_csv):
            out_df.to_csv(iterative_out_csv, index=False)
        else:
            out_df.to_csv(iterative_out_csv, mode="a", header=False, index=False)

    merged = pd.DataFrame(output_rows)
    return merged


# src/compute_keyword_cooc.py (final glue)

def main():
    parser = argparse.ArgumentParser(description="Compute QA co-occurrence features in chunks.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (inclusive) of the DataFrame slice to process.")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (exclusive) of the DataFrame slice to process.")
    parser.add_argument(
        "--window_sizes",
        type=str,
        default="50",
        help="Comma-separated list of window sizes, e.g., '20,50,100'.",
    )
    args = parser.parse_args()

    window_sizes = [int(x) for x in args.window_sizes.split(",") if x.strip()]

    project_root = Path(__file__).resolve().parents[1]

    csv_path = project_root / "data" / "processed" / "eclektic_long_subset.csv"
    df = pd.read_csv(csv_path)
    df = df.loc[:, ["question", "answer", "language", "q_id"]].drop_duplicates()

    start = args.start_idx or 0
    end = args.end_idx
    df = df.iloc[start:end].reset_index(drop=True)
    print(f"Processing rows from {start} to {end if end is not None else 'end'} (total {len(df)} rows)")

    # # show dataframe shape and first row
    # print("df.shape:", df.shape)
    # if not df.empty:
    #     print("first row:", df.iloc[0].to_dict())
    # else:
    #     print("df is empty")

    # return

    # STEP 1: extract keywords per QA
    qa_keywords = build_language_and_qa_keywords(df)
    save_qa_keywords(qa_keywords)

    print("Extracted keywords for QA pairs.")
    print(qa_keywords)

    out_csv = project_root / "data" / "processed" / "eclektic_long_with_cooc_features.csv"
    os.makedirs(out_csv.parent, exist_ok=True)

    # STEP 2: build QA-level co-occurrence features and save results incrementally
    merged = build_qa_cooc_features(
        df,
        qa_keywords=qa_keywords,
        per_qa_out_dir="./cooc_features/qa_cooc",
        window_sizes=window_sizes,
        iterative_out_csv=out_csv,
    )
    print(f"Saved QA-level co-occurrence features to {out_csv}")


if __name__ == "__main__":
    main()
