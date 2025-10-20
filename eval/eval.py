from __future__ import annotations
import os
import pandas as pd
from openrouter_client import OpenRouterClient, OpenAIClient
from prompts import qa_user_message, judge_user_message, judge_system_message, JudgeFields

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

    # client = OpenRouterClient()
    client = OpenAIClient()

    # Build pair table: one row per q_id with both source & target info
    src = df[(df["original_lang"] == source_lang) & (df["language"] == source_lang)][["q_id", "question", "content"]].rename(
        columns={"question": "q_src", "content": "c_src"}
    )
    tgt = df[(df["original_lang"] == source_lang) & (df["language"] == target_lang)][["q_id", "question", "content"]].rename(
        columns={"question": "q_tgt", "content": "c_tgt"}
    )
    pairs = src.merge(tgt, on="q_id", how="inner")
    if pairs.empty:
        raise ValueError(f"No (source={source_lang}, target={target_lang}) pairs found. Check your CSV and languages.")

    records = []
    for _, row in pairs.iterrows():
        qid = row["q_id"]
        # 1) Answer source question
        a_src = answer_question(client, tested_model, row["q_src"], temperature, max_tokens)
        # 2) Answer target question
        a_tgt = answer_question(client, tested_model, row["q_tgt"], temperature, max_tokens)
        # 3) Judge in-language using the aligned contexts
        correct_s = judge_correct(client, judge_model, context=row["c_src"], question=row["q_src"], answer=a_src)
        correct_t = judge_correct(client, judge_model, context=row["c_tgt"], question=row["q_tgt"], answer=a_tgt)

        records.append(
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

    out_df = pd.DataFrame(records)
    os.makedirs(outdir, exist_ok=True)
    out_df.to_csv(os.path.join(outdir, "predictions.csv"), index=False, encoding="utf-8-sig")
    return out_df
