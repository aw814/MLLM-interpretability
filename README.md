# MLLM-interpretability

## 1. Project Overview

This README includes steps for preprocessing, data processing, feature engineering, and large language model evaluation. The main sections are:

- Data Processing  
- Evaluation  
- Feature Engineering  

## 2. Data Processing

### Raw Data Source

- Raw data source: ECLeKTic: https://www.kaggle.com/datasets/googleai/eclektic?resource=download

### Language Filtering & Data Cleaning

- Script Location: /src/data_processing.py

- Load the full ECLeKTic dataset (.jsonl format).

- Filter to a predefined subset of languages (default: English, French, Hebrew, Chinese).

- Reshape the data from wide multilingual format to a long format suitable for modeling.

- Export a clean, reproducible subset in .csv under /data/processed/.

### Output Data Structure (long format)

| Column | Description |
|:-------|:-------------|
| **q_id** | Unique question identifier |
| **original_lang** | Source/original language of the QA pair |
| **original_content** | Original passage/content text |
| **original_question** | Original question text |
| **original_answer** | Original answer text |
| **content** | Target-language content text |
| **question** | Target-language question text |
| **answer** | Target-language answer text |
| **language** | Target language code (e.g., `en`, `fr`, `he`, `zh`) |
| **translated** | Flag (1 = translated, 0 = original) |
| **title**, **url** | Metadata retained from the original dataset |

## 3. Evaluation

The `eval` folder provides the code for evaluation.

First, edit `config.yaml` as needed. Defaults use `en`→`fr` and up to 50 examples.

Then, run
```
python run_eval.py --config config.yaml
```

Artifacts will be written to `eval/artifacts/`:
- `predictions.csv`
- `metrics.json`
- `run_config.resolved.yaml`

You can extend to more languages or richer prompts by adding modules in `prompts.py` and extending `eval.py` loops.

## 4. Feature Engineering

### Wikipedia Language Edition Counts

If you want to focus on feature engineering (e.g., extracting QA topic features), proceed as below.

To generate the language counts locally:
```
pip install -r requirements.txt
python src/fetch_wikipedia_language_versions.py \
    --input data/processed/eclektic_long_subset.csv \
    --output data/processed/eclektic_long_subset_with_lang_counts.csv
```
The output CSV keeps `q_id`, `title`, `url`, and `language_version_count` for downstream features.

### QA Topic Extraction

If you want to focus on feature engineering (e.g., extracting QA topic features), proceed as below.

To generate topic annotations locally:
```
pip install -r requirements.txt
python src/qa_topic_extraction.py --output data/processed/eclektic_long_topics.csv
```
The script reads `data/processed/eclektic_long_subset.csv` and writes a compact CSV with the topic features.

### Question Type Feature

This feature classifies English source questions into linguistic categories based on their structure.

#### Categories

| Type | Description | Example |
|------|-------------|---------|
| **wh** | Information/fact questions starting with What, When, Where, Which, Who, Whom, Whose, Why | *What is the capital of France?* |
| **how** | Process/explanation questions | *How did this event happen?* |
| **yes_no** | Confirmation questions starting with auxiliary verbs (Is, Are, Do, Can, etc.) | *Is Paris the capital of France?* |
| **imperative** | Commands/requests for enumeration or description (Name, List, Describe, etc.) | *Name two rivers in France.* |
| **other** | Questions that don't fit the above patterns | *France capital?* |

#### Usage
```bash

python src/add_question_type.py \
    --input data/processed/eclektic_long_subset.csv \
    --output data/processed/eclektic_long_subset_with_question_type.csv
```
`classify_question_type`:

- Get feature `question_type` as 'wh', 'how', 'yes_no', 'imperative', or 'other' based on the `original_question` field.

`classify_question_type_detail()`: 

- Get feature `question_type_detail`
- Splits wh into: what, when, where, which, who, whom, whose, why
- Splits imperative into: name, list, describe, explain, give, provide, identify, state, mention, tell
- Keeps how, yes_no, and other the same
