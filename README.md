# MLLM-interpretability

Task 1:

- Raw data source: ECLeKTic: https://www.kaggle.com/datasets/googleai/eclektic?resource=download

- Language Filtering & Data Cleaning: 

   -  Script Location: /src/data_processing.py

    - Load the full ECLeKTic dataset (.jsonl format).

    - Filter to a predefined subset of languages (default: English, French, Hebrew, Chinese).

   -  Reshape the data from wide multilingual format to a long format suitable for modeling.

    - Export a clean, reproducible subset in .csv under /data/processed/.

- Output Data Structure (long format)

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


The `eval` folder provides the code for evaluation.

First, edit `config.yaml` as needed. Defaults use `en`→`fr` and up to 50 examples.

Then, run
```
python run_eval.py --config config.yaml
```

Artifacts will be written to `./artifacts/`:
- `predictions.csv`
- `metrics.json`
- `run_config.resolved.yaml`

You can extend to more languages or richer prompts by adding modules in `prompts.py` and extending `eval.py` loops.



