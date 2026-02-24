from __future__ import annotations
import os
import pandas as pd
from openrouter_client import OpenRouterClient, OpenAIClient, GoogleClient
from prompts import qa_user_message, judge_user_message, judge_system_message, JudgeFields
from pathlib import Path
from typing import Dict, Tuple, Protocol
import time
import csv
import re
from collections import Counter


# --- Protocol for chat clients and client selection helper ---
class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str: ...


def _get_client_for_model(model: str) -> ChatClient:
    """
    Select the appropriate client implementation based on the model name.

    Convention:
      - "openrouter/..." → OpenRouterClient
      - "gpt-..."        → OpenAIClient
      - everything else  → GoogleClient (Gemini)
    """
    if model.startswith("openrouter/"):
        return OpenRouterClient()
    if model.startswith("gpt-"):
        return OpenAIClient()
    return GoogleClient()

def answer_question(client: ChatClient, model: str, question: list, temperature: float, max_tokens: int, in_batch: bool = False) -> str:
    if in_batch:
        xx
    else:
        messages = [qa_user_message(question)]
        return client.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)

# word recall score metric (simple proxy for correctness/quality without needing a separate judge call)

# Rough CJK detection (covers Chinese chars; also works reasonably for mixed CJK text)
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')  # CJK Unified Ideographs (common Han)

def is_cjk_char(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))

def tokenize(text: str, lang: str) -> list[str]:
    """
    Tokenization:
    - For zh/ja: split into (1) single CJK chars, and (2) non-CJK chunks split by whitespace/punct.
    - Else: split on word boundaries.
    """
    if text is None:
        return []

    text = text.strip()
    if not text:
        return []

    if lang in {"zh", "ja"}:
        toks = []
        buf = []
        for ch in text:
            if is_cjk_char(ch):
                # flush buffered non-CJK chunk
                if buf:
                    chunk = "".join(buf)
                    # split chunk into word-like pieces
                    toks.extend(re.findall(r"\w+", chunk))
                    buf = []
                toks.append(ch)  # each CJK char as a token
            else:
                buf.append(ch)
        if buf:
            chunk = "".join(buf)
            toks.extend(re.findall(r"\w+", chunk))
        return [t for t in toks if t]  # drop empties
    else:
        # word-level tokens (letters/digits/underscore). Adjust if you want to keep hyphens etc.
        return re.findall(r"\w+", text.lower())

def word_recall_score(gold: str, prediction: str, lang: str, *, multiset: bool = False) -> float:
    """
    R(p,g): portion of gold tokens appearing in prediction.

    multiset=False (recommended default):
      - Each unique gold token counted once (set-style).
    multiset=True:
      - Counts duplicates in gold, capped by prediction frequency (bag-of-words).
    """
    g = tokenize(gold, lang)
    p = tokenize(prediction, lang)

    if not g:
        return 0.0

    if not multiset:
        g_set = set(g)
        p_set = set(p)
        print(f"Gold tokens (unique): {g_set}")
        print(f"Prediction tokens (unique): {p_set}")
        return len(g_set & p_set) / len(g_set)
    else:
        g_cnt = Counter(g)
        p_cnt = Counter(p)
        hits = sum(min(g_cnt[t], p_cnt[t]) for t in g_cnt)
        total = sum(g_cnt.values())
        return hits / total

# ---- Helpers for retries and atomic/resumable I/O --------------------------

def _call_with_retry(fn, *args, retries: int = 3, backoff: float = 1.5, **kwargs):
    """
    Call fn with simple retry/backoff; re-raise last error.
    """
    last = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                raise
    raise last

def _load_source_cache(path: Path) -> pd.DataFrame:
    # schema: {q_id, q_src, a_src, word_recall_score_source}
    if not path.exists():
        return pd.DataFrame(columns=["q_id", "q_src", "a_src", "word_recall_score_source"])
    return pd.read_csv(path, dtype={"q_id": str})

def _save_source_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["q_id"] = out["q_id"].astype(str)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, path)

def _load_target_preds(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "q_id","source_lang","target_lang","q_src","q_tgt",
                "a_src","a_tgt","correct_source","correct_target"
            ]
        )
    return pd.read_csv(path, dtype={"q_id": str})

def _save_target_preds(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["q_id"] = out["q_id"].astype(str)
    out = out.drop_duplicates(subset=["q_id"], keep="last")
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, path)


def run_pairwise_eval(
    df: pd.DataFrame,
    source_lang: str,
    target_lang: str,
    tested_model: str,
    temperature: float,
    max_tokens: int,
    outdir: str,
) -> pd.DataFrame:
    """
    Resumable evaluation:
      - Reuses cached source answers/correctness from {outdir}/{source_lang}_source_answers.csv
      - Skips q_id already in {outdir}/{target_lang}_predictions.csv
      - Appends new predictions and dedupes by q_id on save
      - If source_lang == target_lang, avoids redundant target calls by reusing the source result
    """

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tested_client = _get_client_for_model(tested_model)

    # Normalize identifiers early
    df = df.copy()
    df["q_id"] = df["q_id"].astype(str)

    # Build pair table (one row per q_id with both source & target info)
    base = df[df["original_lang"] == source_lang]
    src = (
        base[base["language"] == source_lang][["q_id", "question", "content"]]
        .rename(columns={"question": "q_src", "content": "c_src"})
    )
    tgt = (
        base[base["language"] == target_lang][["q_id", "question", "content"]]
        .rename(columns={"question": "q_tgt", "content": "c_tgt"})
    )
    same_lang = (source_lang == target_lang)
    if same_lang:
        # Avoid self-join duplicates; reuse source text for target columns.
        pairs = src.copy()
        pairs["q_tgt"] = pairs["q_src"]
        pairs["c_tgt"] = pairs["c_src"]
    else:
        pairs = src.merge(tgt, on="q_id", how="inner")
    pairs = pairs.dropna(subset=["q_id", "q_src", "c_src", "q_tgt", "c_tgt"])
    if pairs.empty:
        raise ValueError(
            f"No aligned pairs for original_lang={source_lang}, "
            f"source={source_lang}→target={target_lang}."
        )

    # Load caches/files (part of resumability)
    source_file = out_dir / f"{source_lang}_source_answers.csv"
    cache_df = _load_source_cache(source_file)
    cache_df["q_id"] = cache_df["q_id"].astype(str)
    source_cache: Dict[str, Tuple[str, float]] = {
        str(r["q_id"]): (r["a_src"], float(r.get("word_recall_score_source", 0.0)))
        for _, r in cache_df.iterrows()
    }

    target_file = out_dir / f"{target_lang}_predictions.csv"
    existing_preds = _load_target_preds(target_file)
    existing_preds["q_id"] = existing_preds["q_id"].astype(str)
    already_done = set(existing_preds["q_id"].unique())

    # Worklist = only q_ids not already completed
    to_process = pairs[~pairs["q_id"].isin(already_done)]
    if to_process.empty:
        return existing_preds.reset_index(drop=True)

    new_records = []
    for _, row in to_process.iterrows():
        qid = row["q_id"]

        # 1) Source answer (reuse cache if present)
        if qid in source_cache:
            a_src, word_recall_score_s = source_cache[qid]
        else:
            a_src = _call_with_retry(
                answer_question, tested_client, tested_model, row["q_src"], temperature, max_tokens, in_batch=True
            )
            word_recall_score_s = word_recall_score(row["original_answer"], a_src, source_lang)
            # correct_s = _call_with_retry(
            #     judge_correct,
            #     judge_client,
            #     context=row["c_src"],
            #     question=row["q_src"],
            #     answer=a_src,
            # )
            source_cache[qid] = (a_src, word_recall_score_s)
            cache_df = pd.concat(
                [cache_df, pd.DataFrame([{
                    "q_id": qid,
                    "q_src": row["q_src"],
                    "a_src": a_src,
                    "word_recall_score_source": word_recall_score_s,
                }])],
                ignore_index=True,
            )

        # 2) Target part
        if same_lang:
            # Source and target are identical; reuse the computed source answer and judgment.
            a_tgt = a_src
            word_recall_score_t = word_recall_score_s
        else:
            a_tgt = _call_with_retry(
                answer_question, tested_client, tested_model, row["q_tgt"], temperature, max_tokens, in_batch=True
            )
            # 3) Judge target (do NOT re-judge source)
            word_recall_score_t = word_recall_score(row["answer"], a_tgt, target_lang)
            # correct_t = _call_with_retry(
            #     judge_correct,
            #     judge_client,
            #     context=row["c_tgt"],
            #     question=row["q_tgt"],
            #     answer=a_tgt,
            # )

        new_records.append(
            {
                "q_id": qid,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "q_src": row["q_src"],
                "q_tgt": row["q_tgt"],
                "c_src": row["c_src"],
                "c_tgt": row["c_tgt"],
                "gold_answer_src": row["original_answer"],
                "gold_answer_tgt": row["answer"],
                "a_src": a_src,
                "a_tgt": a_tgt,
                "word_recall_score_source": word_recall_score_s,
                "word_recall_score_target": word_recall_score_t,
                "correct_source": word_recall_score_s >= 0.5,  # example threshold for correctness
                "correct_target": word_recall_score_t >= 0.5,  
                "translated": row["translated"],
                "title": row["title"],
                "url": row["url"]
            }
        )

        # incremental checkpoint every 50 examples
        if len(new_records) % 50 == 0:
            checkpoint_df = pd.concat([existing_preds, pd.DataFrame(new_records)], ignore_index=True)
            _save_target_preds(checkpoint_df, target_file)
            _save_source_cache(cache_df, source_file)

    # Merge existing + new, dedupe by q_id, and save
    out_df = pd.concat([existing_preds, pd.DataFrame(new_records)], ignore_index=True)
    _save_target_preds(out_df, target_file)
    _save_source_cache(cache_df, source_file)

    return out_df.reset_index(drop=True)




def run_pairwise_eval_klar(
    df: pd.DataFrame,
    source_lang: str,
    target_lang: str,
    tested_model: str,
    temperature: float,
    max_tokens: int,
    outdir: str,
) -> pd.DataFrame:
    
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tested_client = _get_client_for_model(tested_model)

    # Normalize identifiers early
    df = df.copy()
    df["q_id"] = df["q_id"].astype(str)

    # Load caches/files (part of resumability)
    # TODO: KLAR-specific caching if needed (e.g., by q_id or question text), similar to the eclectic version. For now, we just skip already done q_ids based on the target predictions file.
    target_file = out_dir / f"{target_lang}_predictions.csv"
    existing_preds = _load_target_preds(target_file)
    existing_preds["q_id"] = existing_preds["q_id"].astype(str)
    already_done = set(existing_preds["q_id"].unique()) # assuming q_id is the unique identifier for examples
    new_records = []
    for _, row in df.iterrows():
        qid = row["q_id"]
        if qid in already_done:
            continue  # skip already done
        a_tgt = _call_with_retry(
            answer_question, tested_client, tested_model, row["question"], temperature, max_tokens, in_batch=True
        )
        # 3) Judge target (do NOT re-judge source)
        word_recall_score_t = word_recall_score(row["answer"], a_tgt, target_lang)
        # correct_t = _call_with_retry(
        #     judge_correct,
        #     judge_client,
        #     context=row["c_tgt"],
        #     question=row["q_tgt"],
        #     answer=a_tgt,
        # )

        new_records.append(
            {
                "q_id": qid,
                "source_lang": None, # KLAR doesn't have a distinct source vs target question/answer; all languages have the same set of questions. We can populate this if needed, but for now we just mark source_lang as None to reflect the different structure.
                "target_lang": target_lang,
                "q_tgt": row["question"],
                "gold_answer_tgt": row["answer"],
                "a_tgt": a_tgt,
                "word_recall_score_target": word_recall_score_t,
                "correct_target": word_recall_score_t >= 0.5,  
                "relation": row["relation"],
                "index": row["index"],
                "prompt_id": row["prompt_id"]
            }
        )

        # incremental checkpoint every 50 examples
        if len(new_records) % 50 == 0:
            checkpoint_df = pd.concat([existing_preds, pd.DataFrame(new_records)], ignore_index=True)
            _save_target_preds(checkpoint_df, target_file)

    # Merge existing + new, dedupe by q_id, and save
    out_df = pd.concat([existing_preds, pd.DataFrame(new_records)], ignore_index=True)
    _save_target_preds(out_df, target_file)

    return out_df.reset_index(drop=True)


