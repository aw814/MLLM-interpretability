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

## 4. Querying Data (Python)

Use the `wimbd` library to search across the indices.

### Basic Usage with `WimbdSearch`
```python
from wimbd.es import WimbdSearch

# Connect to local server
ws = WimbdSearch(es_url="http://localhost:9200", api_key=None)

# Count documents containing a phrase (AND logic)
count = ws.count_documents_containing_phrases(
    index="mc4_de", 
    phrases=["künstliche intelligenz"], 
    all_phrases=True
)
print(f"Found {count} documents.")

# Query multiple languages at once
res = ws.count_documents_containing_phrases(
    index="mc4_de,mc4_fr,mc4_es", 
    phrases=["data science"], 
    all_phrases=True
)
```

### Low-level Python Client
For custom management tasks:
```python
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Get list of all index names
indices = es.indices.get_alias().keys()
print(list(indices))
```

---

## Troubleshooting

### Version Compatibility
This project requires the `elasticsearch` Python client version **8.x**. If you encounter `media_type_header_exception` or `BadRequestError(400, 'None')`, ensure you are not using client version 9.x.

```bash
uv add "elasticsearch<9"
```

### Timeouts
If `refresh` or `bulk` operations time out, the script is configured with a 60-second `request_timeout`. For extremely large indices, you may see a `ConnectionTimeout` at the very end of indexing during the final refresh; this usually means the data is there, but the server is still busy merging segments.
