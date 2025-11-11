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

This feature classifies English source questions into linguistic and semantic categories using **Large Language Model (LLM) prompting**.
- **Model**: Uses OpenAI GPT model through API prompting (configurable via `.env` file)
- **Prompting logic**: The LLM is prompted with the question text and a predefined taxonomy of question types, returning the most semantically appropriate label.
- **Scope**: Runs classification only for rows with `language == 'en'`, and applies the resulting `question_type` to all rows with the same `qid`.

#### Usage

```bash
python src/add_question_type.py \
    --input data/processed/eclektic_long_subset.csv \
    --output data/processed/eclektic_long_subset_with_question_type.csv
```

**Output:**
Adds a new column `question_type` to the dataset.  
- ⚠️ **English only**: Only processes rows where `language == 'en'`; propagates the label to all related rows sharing the same `qid`.  
- Supports graceful fallback to heuristic classification if the OpenAI API key is missing.

#### Categories
The question_type feature provides 11 detailed semantic categories identified by the LLM:
	•	PERSON – asks for a human individual’s name or identity.
	•	LOCATION – asks for a geographic place (city, country, region, etc.).
	•	ORGANIZATION – asks for a group, company, institution, or agency.
	•	DATE_TIME – asks for a temporal expression (date, year, era, or period).
	•	NUMERIC – asks for a number, count, or measurement.
	•	DEFINITION – asks for the meaning or conceptual explanation of a term.
	•	ENTITY_OBJECT – asks for a specific entity or object (e.g., What is the capital of France?).
	•	EVENT – asks about an event, occurrence, or phenomenon.
	•	WORK_CREATION – asks about a creative or intellectual work (book, movie, painting, invention).
	•	SCIENTIFIC_TECHNICAL – asks for a scientific or technical fact.
	•	OTHER – factual but does not clearly fit the above types.

    
### Bag-of-Words Features

Branch: `feat/bagofwords`

* This branch implements **Bag-of-Words (BoW) feature extraction** for the ECLeKTic multilingual Q&A dataset.
* The module generates BoW text features from question texts using scikit-learn's `CountVectorizer` with language-specific tokenization.
* Each feature row corresponds to a unique `q_id` and `language` combination.

Usage:
```bash
# Generate features
python src/generate_bagofwords_features.py \
    --input data/processed/eclektic_long_subset.csv \
    --output data/processed/bow/

# Load
## for mixed results
vectorizer, features = joblib.load("./data/bow/target_mix_features.pkl")
## for language-specific results
vectorizer, features = joblib.load("./data/bow/target_zh_features.pkl")
metadata = pd.read_csv("./data/bow/metadata.csv")
zh_metadata = metadata[metadata["language"] == "zh"].reset_index(drop=True)
```

Output

Generates vocabularies (`.json`) and feature matrices (`.pkl`) for:
- **Source mix**: All source texts combined
- **Source language-specific**: Per `original_lang` (e.g., `source_en`)
- **Target mix**: All target texts combined  
- **Target language-specific**: Per `language` (e.g., `target_zh`, `target_ja`)
- **Metadata**: `metadata.csv` with `q_id`, languages, title, and URL for alignment

```
bow/
 ├── source_mix_vocab.json        # Combined vocabulary for all source questions and answers
 ├── source_mix_features.pkl      # (vectorizer, matrix) for all source texts combined
 ├── source_en_vocab.json         # Vocabulary for English-only source questions and answers
 ├── source_en_features.pkl       # (vectorizer, matrix) for English-only source texts
 ├── target_mix_vocab.json        # Combined vocabulary for all target questions across languages
 ├── target_mix_features.pkl      # (vectorizer, matrix) for all target texts combined
 ├── target_en_vocab.json         # Vocabulary for English-only target questions
 ├── target_en_features.pkl       # (vectorizer, matrix) for English-only target texts
 ├── target_zh_vocab.json         # Vocabulary for Chinese-only target questions
 ├── target_zh_features.pkl       # (vectorizer, matrix) for Chinese-only target texts
 ├── ...                          # Additional language-specific vocab/features if present (e.g., fr, es, etc.)
 └── metadata.csv                 # Contains q_id, original_lang, language, title, and url for alignment
```

* Each `.json` file contains a **vocabulary dictionary** mapping tokens → feature indices.
* Each `.pkl` file stores a serialized tuple `(vectorizer, feature_matrix)` for fast reloading and reuse.
* `metadata.csv` aligns all samples with their IDs and language info, ensuring easy downstream merging.


---
# 🧩 Feature Engineering — Syntactic Complexity

This module extracts **syntactic complexity features** from the ECLeKTic multilingual QA dataset.  
It provides implementations based on **Stanza** — to compute comparable structural indicators across languages.


## 📘 Overview

We quantify a question’s **syntactic complexity** (as a proxy for reasoning difficulty and cross-lingual transfer robustness) with three dependency-based features:

| Feature | Description |
|----------|--------------|
| **`avg_dep_depth`** | Average dependency path length from each token to the sentence root (overall structural depth). |
| **`max_tree_depth`** | Maximum dependency tree depth (deepest syntactic nesting). |
| **`num_clauses`** | Count of subordinate / relative clauses (labels: `ccomp`, `advcl`, `relcl`, `acl`). |

The output CSV can be merged with other features (topic, frequency, BoW, question type) for downstream modeling.

---

## ⚙️ Implementation A — spaCy (`syntactic_features_spacy.py`)

**Pros:** Fast, lightweight; good for English-only baselines.  
**Cons:** Not multilingual by default.
### Language Filtering & Data Cleaning

- Script Location: /src/data_processing.py

- Load the full ECLeKTic dataset (.jsonl format).

- Filter to a predefined subset of languages (default: English, French, Hebrew, Chinese).

- Reshape the data from wide multilingual format to a long format suitable for modeling.

- Export a clean, reproducible subset in .csv under /data/processed/.

### Output Data Structure (long format)


## ⚙️ Implementation B — Stanza (syntactic_features_stanza.py)

**Pros:** Fully multilingual — supports 60+ languages via Universal Dependencies.

**Cons:** Cross-lingual structural comparability ensured by UD framework. Slower; requires downloading per-language models.

## References

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D. Manning.
“Stanza: A Python Natural Language Processing Toolkit for Many Human Languages.” ACL 2020.

