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
* Each feature row corresponds to a unique combination of `q_id`, `original_lang`, and `language`.

Usage:
```bash
# Generate features
python src/generate_bagofwords_features.py \
    --input data/processed/eclektic_long_subset.csv \
    --output data/processed/bow/

# Load
## for mixed results
df_mix = pd.read_csv("./data/bow/target_mix_features.csv")
features = df_mix.iloc[:, 5:].values  # Skip metadata columns (q_id, original_lang, language, title, url)
metadata = df_mix.iloc[:, :5]         # Extract metadata

## for language-specific results (e.g., for zh)
df_zh = pd.read_csv("./data/bow/target_zh_features.csv")
features = df_zh.iloc[:, 5:].values   # Skip metadata columns
metadata = df_zh.iloc[:, :5]          # Extract metadata
```

Output

Generates vocabularies (`.json`) and feature matrices (`.csv`) for:
- **Source mix**: All source texts combined
- **Source language-specific**: Per `original_lang` (e.g., `source_en`)
- **Target mix**: All target texts combined  
- **Target language-specific**: Per `language` (e.g., `target_zh`, `target_ja`)

Each CSV file contains metadata columns (`q_id`, `original_lang`, `language`, `title`, `url`) followed by feature columns (vocabulary words with L2-normalized counts).
```
bow/
 ├── source_mix_vocab.json        # Combined vocabulary for all source questions
 ├── source_mix_features.csv      # Metadata + feature vectors for all source texts combined
 ├── source_en_vocab.json         # Vocabulary for English-only source questions
 ├── source_en_features.csv       # Metadata + feature vectors for English-only source texts
 ├── target_mix_vocab.json        # Combined vocabulary for all target questions across languages
 ├── target_mix_features.csv      # Metadata + feature vectors for all target texts combined
 ├── target_en_vocab.json         # Vocabulary for English-only target questions
 ├── target_en_features.csv       # Metadata + feature vectors for English-only target texts
 ├── target_zh_vocab.json         # Vocabulary for Chinese-only target questions
 ├── target_zh_features.csv       # Metadata + feature vectors for Chinese-only target texts
 └── ...                          # Additional language-specific vocab/features (e.g., fr, es, ja, ko, etc.)
```

**File Format Details:**

* Each `.json` file contains a **vocabulary dictionary** mapping tokens → feature indices.
* Each `.pkl` file stores a serialized tuple `(vectorizer, feature_matrix)` for fast reloading and reuse.
* `metadata.csv` aligns all samples with their IDs and language info, ensuring easy downstream merging.

* Each `.csv` file contains:
  - **Metadata columns**: `q_id`, `original_lang`, `language`, `title`, `url`
  - **Feature columns**: One column per vocabulary word with L2-normalized BoW counts
  - **Rows**: Each row represents a unique combination of q_id + original_lang + language

**Example CSV structure:**
```csv
q_id,original_lang,language,title,url,word1,word2,word3,...
33,en,en,AFN Bremerhaven,https://en.wikipedia.org/wiki/AFN_Bremerhaven,0.023,0.045,0.012,...
33,en,fr,AFN Bremerhaven,https://en.wikipedia.org/wiki/AFN_Bremerhaven,0.031,0.028,0.019,...
33,en,he,AFN Bremerhaven,https://en.wikipedia.org/wiki/AFN_Bremerhaven,0.041,0.033,0.007,...
```

---
### Feature Engineering — Syntactic Complexity

This module extracts **syntactic complexity features** from the ECLeKTic multilingual QA dataset.  
It provides implementations based on **Stanza** — to compute comparable structural indicators across languages.


#### 📘 Overview

We quantify a question’s **syntactic complexity** (as a proxy for reasoning difficulty and cross-lingual transfer robustness) with three dependency-based features:

| Feature | Description |
|----------|--------------|
| **`avg_dep_depth`** | Average dependency path length from each token to the sentence root (overall structural depth). |
| **`max_tree_depth`** | Maximum dependency tree depth (deepest syntactic nesting). |
| **`num_clauses`** | Count of subordinate / relative clauses (labels: `ccomp`, `advcl`, `relcl`, `acl`). |

The output CSV can be merged with other features (topic, frequency, BoW, question type) for downstream modeling.

---

#### ⚙️ Implementation A — spaCy (`syntactic_features_spacy.py`)

**Pros:** Fast, lightweight; good for English-only baselines.  
**Cons:** Not multilingual by default.
#### Language Filtering & Data Cleaning

- Script Location: /src/data_processing.py

- Load the full ECLeKTic dataset (.jsonl format).

- Filter to a predefined subset of languages (default: English, French, Hebrew, Chinese).

- Reshape the data from wide multilingual format to a long format suitable for modeling.

- Export a clean, reproducible subset in .csv under /data/processed/.

#### Output Data Structure (long format)


#### ⚙️ Implementation B — Stanza (syntactic_features_stanza.py)

**Pros:** Fully multilingual — supports 60+ languages via Universal Dependencies.

**Cons:** Cross-lingual structural comparability ensured by UD framework. Slower; requires downloading per-language models.

#### References

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D. Manning.
“Stanza: A Python Natural Language Processing Toolkit for Many Human Languages.” ACL 2020.



## 5. QA Co-Occurrence Feature Extraction (Chunked Execution)

This module (`src/cooc_features.py`) computes multilingual keyword co-occurrence and PMI-based features for each QA pair in `data/processed/eclektic_long_subset.csv`. It streams documents from a large text corpus via Hugging Face `datasets`, processes a sliding window over tokens, and aggregates statistics per `(q_id, language)` pair. It supports chunked processing and iterative saving, so you can run it on different slices of the data and accumulate results in a single CSV.

### Key Features
- Extracts keywords per QA pair using language-aware tokenization (`build_language_and_qa_keywords`).
- Streams corpus documents per language and computes co-occurrence counts within a fixed-size sliding window (default `window_size=50`, `max_docs=100000`).
- Computes PMI-based statistics for observed keyword pairs and aggregates them per QA pair.
- Tracks keywords that never appear in the sampled corpus (`cooc_unseen_keywords_count`, `cooc_unseen_keywords_ratio`).
- Saves results incrementally every 12 rows to `data/processed/eclektic_long_with_cooc_features.csv`.
- Writes per-QA pair co-occurrence details to `cooc_features/qa_cooc/` as JSON files.

### Usage
Run in chunks to process parts of `data/processed/eclektic_long_subset.csv` sequentially. Each run reads the same input CSV, slices rows by index, and appends features to a shared output file:

```bash
# First 100 rows
python src/cooc_features.py --start_idx 0 --end_idx 100

# Next 100 rows
python src/cooc_features.py --start_idx 100 --end_idx 200

# Continue to the end
python src/cooc_features.py --start_idx 200
```

On the first run with `--start_idx 0`, any existing `data/processed/eclektic_long_with_cooc_features.csv` is removed and recreated. Subsequent runs with `--start_idx > 0` append new rows to the same CSV.

Output columns include: `question`, `answer`, `language`, `q_id`, `cooc_num_pairs`, `cooc_total_pairs`, `cooc_unseen_keywords_count`, `cooc_unseen_keywords_ratio`, and aggregated PMI metrics (`cooc_avg_pmi`, `cooc_max_pmi`, `cooc_min_pmi`, `cooc_std_pmi`).


## Training Simple Classifiers

This module trains a set of baseline classifiers to predict `correct_target` from different feature groups. It loads `data/processed/training_data.csv`, automatically detects numeric vs. categorical columns, and applies a scikit-learn preprocessing pipeline (median imputation + scaling for numeric features, most-frequent imputation + one-hot encoding for categorical features). For each predefined feature set (`qa_topic`, `language_version_count`, `question_type`, `cooc`, `syntactic`, `linguistic`, `wiki_size`, and `all`), it evaluates multiple models (Logistic Regression, Random Forest, SVM, LDA, KNN) using 5×2 repeated stratified cross-validation with accuracy, balanced accuracy, macro-F1, and ROC-AUC (OVR) as metrics. Aggregated results are saved as `model_feature_results.csv` in the script directory and printed as a formatted table.

**Usage:**
```bash
python src/train_simple_classifiers.py
```