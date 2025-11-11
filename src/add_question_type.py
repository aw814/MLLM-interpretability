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
# - Get feature `question_type` as 'wh', 'how', 'yes_no', 'imperative', or 'other' based on the `original_question` field.
# - Uses BART-large-mnli for zero-shot classification

# `classify_question_type_detail()`: 
# - Get feature `question_type_detail`
# - Splits wh into: what, when, where, which, who, whom, whose, why
# - Splits imperative into: name, list, describe, explain, give, provide, identify, state, mention, tell
# - Keeps how, yes_no, and other the same
# - Uses BART-large-mnli for zero-shot classification

# Note: So far only supports English source questions (`original_lang` == 'en')

# Initialize BART classifier globally
print("Loading BART-large-mnli model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("Model loaded successfully!")


def classify_question_type(question: str) -> str:
    """
    Classify English question into types using BART zero-shot classification.
    
    Args:
        question: English question text

    Returns:
        Question type: 'wh', 'how', 'yes_no', 'imperative', or 'other'
    """
    if not isinstance(question, str) or not question.strip():
        return 'other'
    
    question = question.strip()
    
    # Define candidate labels for general classification
    candidate_labels = [
        'wh-question asking what, when, where, which, who, whom, whose, or why',
        'how question',
        'yes-no question requiring true or false answer',
        'imperative command requesting to name, list, describe, explain, give, provide, identify, state, mention, or tell',
        'other type of question or statement'
    ]
    
    result = classifier(question, candidate_labels)
    
    # Map labels back to simple types
    label_map = {
        'wh-question asking what, when, where, which, who, whom, whose, or why': 'wh',
        'how question': 'how',
        'yes-no question requiring true or false answer': 'yes_no',
        'imperative command requesting to name, list, describe, explain, give, provide, identify, state, mention, or tell': 'imperative',
        'other type of question or statement': 'other'
    }
    
    top_label = result['labels'][0]
    return label_map[top_label]


def classify_question_type_detail(question: str) -> str:
    """
    Classify English question into detailed types using BART zero-shot classification.
    
    Args:
        question: English question text

    Returns:
        Detailed question type: 'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why',
        'how', 'yes_no', 'name', 'list', 'describe', 'explain', 'give', 'provide', 
        'identify', 'state', 'mention', 'tell', or 'other'
    """
    if not isinstance(question, str) or not question.strip():
        return 'other'
    
    question = question.strip()
    
    # Define all detailed candidate labels
    candidate_labels = [
        'what question',
        'when question',
        'where question',
        'which question',
        'who question',
        'whom question',
        'whose question',
        'why question',
        'how question',
        'yes-no question',
        'command to name something',
        'command to list items',
        'command to describe something',
        'command to explain something',
        'command to give information',
        'command to provide information',
        'command to identify something',
        'command to state information',
        'command to mention something',
        'command to tell something',
        'other type of question or statement'
    ]
    
    result = classifier(question, candidate_labels)
    
    # Map labels back to simple types
    label_map = {
        'what question': 'what',
        'when question': 'when',
        'where question': 'where',
        'which question': 'which',
        'who question': 'who',
        'whom question': 'whom',
        'whose question': 'whose',
        'why question': 'why',
        'how question': 'how',
        'yes-no question': 'yes_no',
        'command to name something': 'name',
        'command to list items': 'list',
        'command to describe something': 'describe',
        'command to explain something': 'explain',
        'command to give information': 'give',
        'command to provide information': 'provide',
        'command to identify something': 'identify',
        'command to state information': 'state',
        'command to mention something': 'mention',
        'command to tell something': 'tell',
        'other type of question or statement': 'other'
    }
    
    top_label = result['labels'][0]
    return label_map[top_label]


def add_question_type_column(input_path: str, output_path: str):
    """
    Read CSV, add question_type and question_type_detail columns based on original_question, and save.
    
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
    
    # Classify each question (both general and detailed)
    # Note: This may take some time for large datasets
    print("\nClassifying general question types...")
    df['question_type'] = df['original_question'].apply(classify_question_type)
    
    print("Classifying detailed question types...")
    df['question_type_detail'] = df['original_question'].apply(classify_question_type_detail)
    
    # Print distribution for general type
    print("\n" + "="*60)
    print("Question Type Distribution (General):")
    print("="*60)
    print(df['question_type'].value_counts())
    print(f"\nPercentages:")
    print(df['question_type'].value_counts(normalize=True) * 100)
    
    # Print distribution for detailed type
    print("\n" + "="*60)
    print("Question Type Distribution (Detailed):")
    print("="*60)
    print(df['question_type_detail'].value_counts())
    print(f"\nPercentages:")
    print(df['question_type_detail'].value_counts(normalize=True) * 100)
    
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