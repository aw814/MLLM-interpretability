import pandas as pd
import argparse
from typing import List, Dict
from transformers import pipeline

#### Usage
# ```bash
# python src/add_question_type.py \
#     --input data/processed/eclektic_long_subset.csv \
#     --output data/processed/eclektic_long_subset_with_question_type.csv
# ```
# `classify_question_type`:
# - Get feature `question_type` with detailed categories based on the `original_question` field
# - Categories: what, when, where, which, who, whom, whose, why, how, yes_no, other
# - Uses BART-large-mnli for zero-shot classification

# Note: So far only supports English source questions (`original_lang` == 'en')

# Initialize BART classifier globally
print("Loading BART-large-mnli model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("Model loaded successfully!")


def classify_question_type(question: str) -> str:
    """
    Classify English question into detailed types using BART zero-shot classification.
    
    Args:
        question: English question text

    Returns:
        Detailed question type: 'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why',
        'how', 'yes_no', or 'other'
    """
    if not isinstance(question, str) or not question.strip():
        return 'other'
    
    question = question.strip()
    
    # Define all detailed candidate labels
    # Use descriptive labels to give BART more context for accurate classification
    # Then map to clean output labels via label_map
    candidate_labels = [
        'what question asking about things, definitions, or identity',
        'when question asking about time or temporal information',
        'where question asking about location or place',
        'which question asking for selection or choice among options',
        'who question asking about people or agents',
        'whom question asking about people as grammatical objects',
        'whose question asking about possession or ownership',
        'why question asking about reasons or causes',
        'how question asking about manner, method, or process',
        'yes-no question requiring true or false answer',
        'other type of question or statement'
    ]
    
    result = classifier(question, candidate_labels)
    
    # Map descriptive labels back to clean output types
    label_map = {
        'what question asking about things, definitions, or identity': 'what',
        'when question asking about time or temporal information': 'when',
        'where question asking about location or place': 'where',
        'which question asking for selection or choice among options': 'which',
        'who question asking about people or agents': 'who',
        'whom question asking about people as grammatical objects': 'whom',
        'whose question asking about possession or ownership': 'whose',
        'why question asking about reasons or causes': 'why',
        'how question asking about manner, method, or process': 'how',
        'yes-no question requiring true or false answer': 'yes_no',
        'other type of question or statement': 'other'
    }
    
    top_label = result['labels'][0]
    return label_map[top_label]


def add_question_type_column(input_path: str, output_path: str):
    """
    Read CSV, add question_type column based on original_question, and save.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        
    Raises:
        ValueError: If original_question or original_lang columns are missing
        AssertionError: If source language is not English
    """
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Check if required columns exist
    if 'original_question' not in df.columns:
        raise ValueError("Column 'original_question' not found in input CSV")
    
    if 'original_lang' not in df.columns:
        raise ValueError("Column 'original_lang' not found in input CSV")
    
    # Assert that all source languages are English
    unique_source_langs = df['original_lang'].unique()
    assert len(unique_source_langs) == 1 and unique_source_langs[0] == 'en', \
        f"Error: This script only supports English source questions. " \
        f"Found source language(s): {unique_source_langs}. " \
        f"All rows must have original_lang='en'."
    
    print(f"✓ Verified: All {len(df)} questions have English source language")
    print(f"Classifying questions using BART-large-mnli...")
    print(f"Total questions to classify: {len(df)}")
    
    # Classify each question with detailed categories
    # Note: This may take some time for large datasets
    print("\nClassifying question types...")
    df['question_type'] = df['original_question'].apply(classify_question_type)
    
    # Print distribution
    print("\n" + "="*60)
    print("Question Type Distribution:")
    print("="*60)
    print(df['question_type'].value_counts())
    print(f"\nPercentages:")
    print(df['question_type'].value_counts(normalize=True) * 100)
    
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