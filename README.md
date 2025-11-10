# Bag-of-Words Features for ECLeKTic Dataset

Branch: `feat/bagofwords`

## Overview

* This branch implements **Bag-of-Words (BoW) feature extraction** for the ECLeKTic multilingual Q&A dataset.
* The module generates BoW text features from question texts using scikit-learn's `CountVectorizer` with language-specific tokenization.
* Each feature row corresponds to a unique `q_id` and `language` combination.
* **Reference:** Graff, M., Moctezuma, D., & Téllez, E. S. (2025). *Bag-of-Words is Not Dead: A Performance Analysis on a Myriad of Text Classification Challenges.* *Natural Language Processing Journal*, 100154.

## Key Features

* **Multilingual tokenization** supporting 12+ languages (Chinese, Japanese, Korean, Hindi, Hebrew, English, French, German, Spanish, Italian, Portuguese, Indonesian)
* **Language-specific tokenizers**: jieba (Chinese), Sudachi (Japanese), KoNLPy (Korean), with regex fallbacks for others
* **Mixed and language-specific vocabularies** for both source (original) and target (translated) texts
* **Configurable vocabulary size** (default: 5000 most frequent tokens via `max_features`)
* **L2-normalized feature matrices** for consistent downstream use

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Generate features
python src/generate_bagofwords_features.py \
    --input data/processed/eclektic_long_subset.csv \
    --output data/processed/bow/
```

## Output

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


# Data:

- Raw data source: ECLeKTic: https://www.kaggle.com/datasets/googleai/eclektic?resource=download

- Language Filtering & Data Cleaning: 

   -  Script Location: /src/data_processing.py

    - Load the full ECLeKTic dataset (.jsonl format).

    - Filter to a predefined subset of languages (default: English, French, Hebrew, Chinese).

   -  Reshape the data from wide multilingual format to a long format suitable for modeling.

    - Export a clean, reproducible subset in .csv under /data/processed/.

## Data Structure:

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

# Evaluation:

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



