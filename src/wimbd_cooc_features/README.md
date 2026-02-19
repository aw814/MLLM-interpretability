# WIMBD Elasticsearch Guide

This directory contains tools for managing a local Elasticsearch server and indexing the mC4 dataset for use with the `wimbd` library.

## 1. Starting the Server

The server is installed in `elasticsearch-8.17.0` at the project root.

```bash
# From the project root:
bash src/wimbd_cooc_features/run_elsticsearch_server.sh
```

**Note:** The server is configured without security (no HTTPS/Auth) for local development on port 9200.

---

## 2. Managing Indices (curl)

Use these commands to monitor the status of your data.

### Check Server Health
```bash
curl -X GET "http://localhost:9200/"
```

### List All Indices
Shows names, document counts, and disk usage.
```bash
curl -s "http://localhost:9200/_cat/indices?v"
```

### Count Documents in a Specific Index
```bash
curl -s -X GET "http://localhost:9200/mc4_de/_count"
```

### Delete an Index
**Warning:** This permanently removes the data.
```bash
curl -X DELETE "http://localhost:9200/mc4_de"
```

---

## 3. Indexing mC4 Data

Use `index_mc4.py` to index Arrow files from `multilingual_datasets/mC4/`.

### Run Indexing for a Language
```bash
uv run python src/wimbd_cooc_features/index_mc4.py --lang de --batch_size 2000
```

### Run in Background (nohup)
For large datasets, run in the background to ensure completion even if the terminal closes.
```bash
nohup uv run python src/wimbd_cooc_features/index_mc4.py --lang de --batch_size 2000 --quiet > index_de.log 2>&1 &
```

---

## 4. Querying Data with `WimbdHelper`

The `WimbdHelper` class in `wimbd_helper.py` provides a clean abstraction over the `wimbd` library's functional core.

### Basic Usage
```python
from wimbd_helper import WimbdHelper

# Initialize the helper (defaults to http://localhost:9200)
helper = WimbdHelper()

# Count documents containing a phrase (AND logic)
count = helper.count_phrases("mc4_de", ["künstliche intelligenz"], all_phrases=True)
print(f"Found {count} documents.")

# Query multiple languages at once
res = helper.count_phrases("mc4_de,mc4_fr,mc4_es", ["data science"], all_phrases=True)

# List all mc4 indices
print(helper.list_mc4_indices())
```

### Retrieving Documents
```python
# Get up to 5 document hits
docs = helper.get_documents("mc4_de", ["klimawandel"], num_documents=5)
for doc in docs:
    print(doc["_source"]["text"][:200])
```

---

## 5. Calculating Co-occurrences and PPMI

The `calculate_co_occurrences.py` script provides a function to calculate co-occurrence statistics and Positive Pointwise Mutual Information (PPMI).

### Example Usage
```python
from calculate_co_occurrences import co_occure

# Calculate stats for two terms in one index
stats = co_occure("science", "data", "mc4_de")

# Example output:
# {
#     'str1': 'science', 
#     'str2': 'data', 
#     'indices': 'mc4_de', 
#     'co_occurrence_count': 5384, 
#     'count_str1': 60086, 
#     'count_str2': 88625, 
#     'total_documents': 8643085, 
#     'ppmi': 3.1274102488272395
# }

# Calculate stats across multiple indices
stats = co_occure("AI", "intelligence", ["mc4_de", "mc4_fr", "mc4_zh"])

print(f"Co-occurrence Count: {stats['co_occurrence_count']}")
print(f"PPMI Score: {stats['ppmi']}")
```

**Returned Dictionary Schema:**
* `str1`, `str2`: The terms compared.
* `indices`: The index or indices searched.
* `co_occurrence_count`: Documents containing both terms ($N_{1,2}$).
* `count_str1`, `count_str2`: Total documents containing each term individually ($N_1$, $N_2$).
* `total_documents`: Total number of documents across the searched indices ($N_{total}$).
* `ppmi`: The calculated PPMI score.

---

## Troubleshooting

### Version Compatibility
This project requires the `elasticsearch` Python client version **8.x**. If you encounter `media_type_header_exception` or `BadRequestError(400, 'None')`, ensure you are not using client version 9.x.

```bash
uv add "elasticsearch<9"
```

### Timeouts
If `refresh` or `bulk` operations time out, the script is configured with a 60-second `request_timeout`. For extremely large indices, you may see a `ConnectionTimeout` at the very end of indexing during the final refresh; this usually means the data is there, but the server is still busy merging segments.
