import pandas as pd
import argparse
from typing import List, Dict
import json
# from transformers import pipeline
from dotenv import load_dotenv

import os

load_dotenv()
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if OPENAI_API_KEY:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE)
    else:
        print("Warning: OPENAI_API_KEY not found in environment. LLM classification will fall back to heuristic.")
        _openai_client = None
except ImportError:
    print("Warning: openai library not installed. LLM classification will fall back to heuristic.")
    _openai_client = None

QUESTION_TYPES = [
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "DATE_TIME",
    "NUMERIC",
    "DEFINITION",
    "ENTITY_OBJECT",
    "EVENT",
    "WORK_CREATION",
    "SCIENTIFIC_TECHNICAL",
    "OTHER",
]

#### Usage
# ```bash
# python src/add_question_type.py \
#     --input data/processed/eclektic_long_subset.csv \
#     --output data/processed/eclektic_long_subset_with_question_type.csv
# ```
# `classify_question_type`:
# - Get feature `question_type` with detailed categories based on the `language` field
# - Categories: what, when, where, which, who, whom, whose, why, how, yes_no, other
# - Uses BART-large-mnli for zero-shot classification

# Note: So far only supports English source questions (`language` == 'en')

# Initialize BART classifier globally (disabled in favor of LLM-based classifier)
# print("Loading BART-large-mnli model...")
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# print("Model loaded successfully!")


def classify_question_type(question: str) -> str:
    """
    Classify an English factoid question into factual types using an LLM.

    Args:
        question: English question text

    Returns:
        One of the factual question types:
        PERSON, LOCATION, ORGANIZATION, DATE_TIME, NUMERIC, DEFINITION,
        ENTITY_OBJECT, EVENT, WORK_CREATION, SCIENTIFIC_TECHNICAL, OTHER.
    """
    if not isinstance(question, str) or not question.strip():
        return 'other'
    
    question = question.strip()
    
    # NOTE:
    # The original implementation used BART-large-mnli via HuggingFace for
    # zero-shot classification over WH-question types. That code is now
    # commented out below. We instead use an LLM (OpenAI) to classify into
    # factual answer-type categories. If the OpenAI client is not available,
    # we fall back to a simple WH-word heuristic and return lowercase labels.
    if _openai_client is None:
        # Very simple heuristic fallback based on leading WH-word
        q_low = question.lower()
        if q_low.startswith("who"):
            return "person"
        if q_low.startswith("where"):
            return "location"
        if q_low.startswith("when"):
            return "date_time"
        if q_low.startswith("how many") or q_low.startswith("how much"):
            return "numeric"
        if q_low.startswith("what is") or q_low.startswith("what are"):
            return "definition"
        # Default fallback
        return "other"
    
    # ------------------------------------------------------------------
    # Previous BART-based zero-shot classification (disabled)
    # ------------------------------------------------------------------
    # # Define all detailed candidate labels
    # # Use descriptive labels to give BART more context for accurate classification
    # # Then map to clean output labels via label_map
    # candidate_labels = [
    #     'what question asking about things, definitions, or identity',
    #     'when question asking about time or temporal information',
    #     'where question asking about location or place',
    #     'which question asking for selection or choice among options',
    #     'who question asking about people or agents',
    #     'whom question asking about people as grammatical objects',
    #     'whose question asking about possession or ownership',
    #     'why question asking about reasons or causes',
    #     'how question asking about manner, method, or process',
    #     'yes-no question requiring true or false answer',
    #     'other type of question or statement'
    # ]
    #
    # result = classifier(question, candidate_labels)
    #
    # # Map descriptive labels back to clean output types
    # label_map = {
    #     'what question asking about things, definitions, or identity': 'what',
    #     'when question asking about time or temporal information': 'when',
    #     'where question asking about location or place': 'where',
    #     'which question asking for selection or choice among options': 'which',
    #     'who question asking about people or agents': 'who',
    #     'whom question asking about people as grammatical objects': 'whom',
    #     'whose question asking about possession or ownership': 'whose',
    #     'why question asking about reasons or causes': 'why',
    #     'how question asking about manner, method, or process': 'how',
    #     'yes-no question requiring true or false answer': 'yes_no',
    #     'other type of question or statement': 'other'
    # }
    #
    # top_label = result['labels'][0]
    # return label_map[top_label]

    system_msg = (
        "You are an expert annotator for factual question classification. "
        "Given a question, you must assign exactly ONE label from the list:\n"
        f"{', '.join(QUESTION_TYPES)}.\n\n"
        "Definitions:\n"
        "- PERSON: asks for a human individual's name or identity.\n"
        "- LOCATION: asks for a geographic place (city, country, region, etc.).\n"
        "- ORGANIZATION: asks for a group, company, institution, or agency.\n"
        "- DATE_TIME: asks for a temporal expression (date, year, era, period).\n"
        "- NUMERIC: asks for a number, count, or measurement.\n"
        "- DEFINITION: asks for the meaning or concept of a term.\n"
        "- ENTITY_OBJECT: asks for a specific entity or object (including 'capital of X').\n"
        "- EVENT: asks about an event, occurrence, or phenomenon.\n"
        "- WORK_CREATION: asks about a creative or intellectual work (book, movie, painting, invention).\n"
        "- SCIENTIFIC_TECHNICAL: asks for a scientific or technical fact.\n"
        "- OTHER: factual but does not clearly fit above types.\n\n"
        "Output MUST be a compact JSON object with keys 'question_type' and 'justification'. "
        "The 'question_type' must be one of the listed labels."
    )

    user_msg = f"Question: {question}"

    try:
        response = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content
    except Exception as e:
        # On any API failure, fall back to a generic label
        return "other"

    try:
        parsed = json.loads(content)
        label = parsed.get("question_type", "OTHER").strip().upper()
    except json.JSONDecodeError:
        # If the model did not return valid JSON, try a very simple parse:
        # assume the label is the first uppercase token in the response.
        text = content.strip()
        # naive fallback parsing
        for t in QUESTION_TYPES:
            if t in text:
                label = t
                break
        else:
            label = "OTHER"

    if label not in QUESTION_TYPES:
        label = "OTHER"

    # Return lowercase to roughly match previous convention ('other', 'what', etc.)
    return label.lower()


def add_question_type_column(input_path: str, output_path: str):
    """
    Read CSV, classify English questions into factual types, propagate labels by q_id, and save.
    
    For each row where language == 'en', we run classify_question_type on language.
    The resulting question_type is then assigned to all rows that share the same q_id.
    """
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Check if required columns exist
    if 'language' not in df.columns:
        raise ValueError("Column 'language' not found in input CSV")
    
    if 'language' not in df.columns:
        raise ValueError("Column 'language' not found in input CSV")

    if 'q_id' not in df.columns:
        raise ValueError("Column 'q_id' not found in input CSV")

    # Only classify English source questions and propagate labels by q_id
    en_mask = df['language'] == 'en'
    num_en = en_mask.sum()

    if num_en == 0:
        raise ValueError("No rows with language == 'en' found. At least one English question is required.")

    print(f"Found {len(df)} total rows; {num_en} rows have English source language (language == 'en').")
    print("Classifying English questions using LLM-based factual types...")
    print(f"Total English questions to classify: {num_en}")
    
    # Classify only English questions and propagate labels to all rows with the same q_id
    print("\nClassifying question types for English questions...")
    en_df = df[en_mask].copy()
    en_df['question_type'] = en_df['question'].apply(classify_question_type)

    # Build a mapping from q_id -> question_type (first non-null per q_id)
    qtype_by_qid = (
        en_df[['q_id', 'question_type']]
        .dropna()
        .drop_duplicates(subset='q_id')
    )
    qid_to_type = dict(zip(qtype_by_qid['q_id'], qtype_by_qid['question_type']))

    # Apply the mapped question_type to all rows sharing the same q_id
    df['question_type'] = df['q_id'].map(qid_to_type)
    
    # Print distribution
    print("\n" + "="*60)
    print("Question Type Distribution:")
    print("="*60)
    print(df['question_type'].value_counts(dropna=True))
    print(f"\nPercentages:")
    print(df['question_type'].value_counts(normalize=True, dropna=True) * 100)
    
    # Save to output
    print(f"\nSaving results to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done!")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add question type classification to ECLeKTic dataset using BART-large-mnli (English source questions only)"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    
    args = parser.parse_args()
    
    try:
        add_question_type_column(args.input, args.output)
    except AssertionError as e:
        print(f"\n{e}")
        print("\nIf you need to classify questions in other languages, please wait for the release of a multilingual version of this script.")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)


if __name__ == "__main__":
    main()