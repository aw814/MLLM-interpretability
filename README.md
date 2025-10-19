# MLLM-interpretability

Task 1:

- Raw data source: ECLeKTic: https://www.kaggle.com/datasets/googleai/eclektic?resource=download

- Language Filtering & Data Cleaning: 

   -  Script Location: /src/data_processing.py

    - Load the full ECLeKTic dataset (.jsonl format).

    - Filter to a predefined subset of languages (default: English, French, Hebrew, Chinese).

   -  Reshape the data from wide multilingual format to a long format suitable for modeling.

    - Export a clean, reproducible subset in .csv under /data/processed/.