from __future__ import annotations
import os
import pandas as pd
from openrouter_client import OpenRouterClient, OpenAIClient
from prompts import qa_user_message, judge_user_message, judge_system_message, JudgeFields
from pathlib import Path
from typing import Dict, Tuple
import time
import csv

def answer_question(client: OpenRouterClient, model: str, question: str, temperature: float, max_tokens: int) -> str:
    messages = [qa_user_message(question)]
    return client.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)

def judge_correct(
    client: OpenRouterClient|OpenAIClient,
    judge_model: str,
    context: str,
    question: str,
    answer: str,
) -> bool:
    messages = [judge_system_message(), judge_user_message(JudgeFields(context=context, question=question, answer=answer))]
    out = client.chat(model=judge_model, messages=messages, temperature=0.0, max_tokens=4)
    return out.strip().upper().startswith("Y")  # YES → True, else False

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
    if not path.exists():
        return pd.DataFrame(columns=["q_id", "q_src", "a_src", "correct_source"])
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
    judge_model: str,
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

    # client = OpenRouterClient()
    client = OpenAIClient()

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
    source_cache: Dict[str, Tuple[str, bool]] = {
        r["q_id"]: (r["a_src"], bool(r["correct_source"])) for _, r in cache_df.iterrows()
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
            a_src, correct_s = source_cache[qid]
        else:
            a_src = _call_with_retry(
                answer_question, client, tested_model, row["q_src"], temperature, max_tokens
            )
            correct_s = _call_with_retry(
                judge_correct,
                client,
                judge_model,
                context=row["c_src"],
                question=row["q_src"],
                answer=a_src,
            )
            source_cache[qid] = (a_src, bool(correct_s))
            cache_df = pd.concat(
                [cache_df, pd.DataFrame([{
                    "q_id": qid,
                    "q_src": row["q_src"],
                    "a_src": a_src,
                    "correct_source": bool(correct_s),
                }])],
                ignore_index=True,
            )

        # 2) Target part
        if same_lang:
            # Source and target are identical; reuse the computed source answer and judgment.
            a_tgt = a_src
            correct_t = correct_s
        else:
            a_tgt = _call_with_retry(
                answer_question, client, tested_model, row["q_tgt"], temperature, max_tokens
            )
            # 3) Judge target (do NOT re-judge source)
            correct_t = _call_with_retry(
                judge_correct,
                client,
                judge_model,
                context=row["c_tgt"],
                question=row["q_tgt"],
                answer=a_tgt,
            )

        new_records.append(
            {
                "q_id": qid,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "q_src": row["q_src"],
                "q_tgt": row["q_tgt"],
                "a_src": a_src,
                "a_tgt": a_tgt,
                "correct_source": bool(correct_s),
                "correct_target": bool(correct_t),
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
