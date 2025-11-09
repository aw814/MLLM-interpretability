"""Utility for deriving topic features for QA pairs using zero-shot classification."""

from __future__ import annotations

import argparse
from typing import Iterable, List

import pandas as pd
from transformers import pipeline

# Default candidate topics covering broad encyclopedic domains.
DEFAULT_TOPICS = [
    "science",
    "technology",
    "history",
    "culture",
    "geography",
    "politics",
    "economics",
    "sports",
    "arts",
    "health",
    "education",
    "law",
    "environment",
]


def _parse_topic_list(raw: str | None) -> List[str]:
    if not raw:
        return DEFAULT_TOPICS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _compose_sequence(row: pd.Series) -> str:
    parts: List[str] = []
    for key in ("title", "question", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if not parts:  # Safe fallback to original fields
        for key in ("original_question", "original_answer"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return " \n".join(parts)


def _batched(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def assign_topics(
    df: pd.DataFrame,
    candidate_topics: List[str],
    model_name: str,
    hypothesis_template: str,
    batch_size: int,
    multi_label: bool,
) -> pd.DataFrame:

    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    classifier = pipeline("zero-shot-classification", model=model_name, device=device)

    sequences = df.apply(_compose_sequence, axis=1).tolist()
    total = len(sequences)
    if total == 0:
        print("No sequences found for topic assignment.")
        return df

    print(
        f"Running zero-shot topic assignment on {total} rows using {model_name} ({device})."
    )

    labels: List[str] = []
    scores: List[float] = []
    ranked: List[str] = []

    processed = 0
    for batch in _batched(sequences, batch_size):
        results = classifier(
            batch,
            candidate_labels=candidate_topics,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label,
        )

        if isinstance(results, dict):  # When batch_size == 1 Transformers returns dict
            results = [results]

        for res in results:
            top_label = res["labels"][0]
            top_score = res["scores"][0]
            formatted = ";".join(
                f"{label}:{score:.4f}"
                for label, score in zip(res["labels"], res["scores"])
            )
            labels.append(top_label)
            scores.append(top_score)
            ranked.append(formatted)

        processed += len(batch)
        print(f"Processed {processed}/{total} rows", flush=True)

    df = df.copy()
    df["qa_topic"] = labels
    df["qa_topic_score"] = scores
    df["qa_topic_ranked_candidates"] = ranked
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/eclektic_long_subset.csv",
        help="Path to the processed ECLeKTic long-form CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/eclektic_long_qa_topics.csv",
        help="Destination CSV for rows augmented with topic features.",
    )
    parser.add_argument(
        "--candidate-topics",
        help="Comma-separated list of topic labels to use for zero-shot classification.",
    )
    parser.add_argument(
        "--model",
        default="facebook/bart-large-mnli",
        help="Hugging Face model identifier for zero-shot classification.",
    )
    parser.add_argument(
        "--hypothesis",
        default="This text is about {}.",
        help="Hypothesis template passed to the zero-shot classifier.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of sequences to process per inference batch.",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Enable multi-label inference and keep highest-scoring topic as feature.",
    )

    args = parser.parse_args()

    topics = _parse_topic_list(args.candidate_topics)

    df = pd.read_csv(args.input)

    enriched_df = assign_topics(
        df,
        candidate_topics=topics,
        model_name=args.model,
        hypothesis_template=args.hypothesis,
        batch_size=args.batch_size,
        multi_label=args.multi_label,
    )

    output_cols = [
        "q_id",
        "original_content",
        "original_question",
        "qa_topic",
        "qa_topic_score",
        "qa_topic_ranked_candidates",
    ]

    for col in output_cols:
        if col not in enriched_df.columns:
            enriched_df[col] = None

    enriched_df.to_csv(args.output, columns=output_cols, index=False)
    print(f"Saved topic annotations for {len(enriched_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
