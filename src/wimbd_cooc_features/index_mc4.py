import os
import argparse
import logging
import pyarrow as pa
import pyarrow.ipc as ipc
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_index(es, index_name):
    """Create the index with appropriate mappings if it doesn't exist."""
    try:
        es.indices.get(index=index_name)
        logger.info(f"Index {index_name} already exists.")
        return
    except Exception:
        # If it doesn't exist, it will raise an error
        logger.info(f"Index {index_name} does not exist or error checking it. Proceeding with creation.")

    mappings = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "url": {"type": "keyword"},
                "timestamp": {"type": "date"},
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1"  # Disable refresh during bulk indexing
        }
    }
    try:
        es.indices.create(index=index_name, body=mappings)
        logger.info(f"Created index {index_name}.")
    except Exception as e:
        logger.error(f"Error creating index: {e}")

def doc_generator(index_name, arrow_files):
    """Generator for bulk indexing from multiple arrow files."""
    for file_path in arrow_files:
        logger.info(f"Processing {file_path}")
        try:
            with pa.memory_map(file_path, "rb") as source:
                reader = ipc.open_stream(source)
                for batch in reader:
                    pylist = batch.to_pylist()
                    for doc in pylist:
                        yield {
                            "_index": index_name,
                            "_source": {
                                "text": doc.get("text"),
                                "url": doc.get("url"),
                                "timestamp": doc.get("timestamp"),
                            }
                        }
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

def find_arrow_files(lang_dir):
    """Recursively find all .arrow files in a language directory."""
    arrow_files = []
    for root, _, files in os.walk(lang_dir):
        for file in files:
            if file.endswith(".arrow") and "train" in file:
                arrow_files.append(os.path.join(root, file))
    return sorted(arrow_files)

def main():
    parser = argparse.ArgumentParser(description="Index mC4 arrow files into Elasticsearch.")
    parser.add_argument("--lang", required=True, help="Language code (e.g., de, fr, zh)")
    parser.add_argument("--es_url", default="http://localhost:9200", help="Elasticsearch URL")
    parser.add_argument("--batch_size", type=int, default=1000, help="Bulk indexing batch size")
    parser.add_argument("--limit_files", type=int, default=None, help="Limit number of arrow files to index (for testing)")
    parser.add_argument("--quiet", action="store_true", help="Minimize logging output")
    
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("elastic_transport").setLevel(logging.WARNING)
        logger.setLevel(logging.INFO)
    
    base_dir = "multilingual_datasets/mC4"
    lang_dir = os.path.join(base_dir, args.lang)
    
    if not os.path.exists(lang_dir):
        logger.error(f"Directory {lang_dir} does not exist.")
        return

    arrow_files = find_arrow_files(lang_dir)
    if not arrow_files:
        logger.error(f"No arrow files found in {lang_dir}")
        return

    if args.limit_files:
        arrow_files = arrow_files[:args.limit_files]
        logger.info(f"Limited to first {args.limit_files} files.")

    index_name = f"mc4_{args.lang}"
    es = Elasticsearch(
        args.es_url,
        request_timeout=60  # Increase timeout to 60 seconds
    )
    
    create_index(es, index_name)
    
    logger.info(f"Starting indexing for {len(arrow_files)} files into {index_name}...")
    
    # Use streaming_bulk helper
    success, failed = 0, 0
    progress = tqdm(unit="docs")
    
    from elasticsearch.helpers import streaming_bulk
    for ok, item in streaming_bulk(es, doc_generator(index_name, arrow_files), chunk_size=args.batch_size, raise_on_error=False):
        if ok:
            success += 1
        else:
            failed += 1
        progress.update(1)
    
    progress.close()
    
    # Re-enable refresh interval
    es.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "1s"}})
    es.indices.refresh(index=index_name)
    
    logger.info(f"Indexing complete. Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    main()
