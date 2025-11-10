
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
