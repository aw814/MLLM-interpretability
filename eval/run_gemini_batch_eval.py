"""
Run Gemini Batch Evaluation

This script:
1. Uploads a JSONL batch request file
2. Submits a Gemini Batch job
3. Waits until the job finishes
4. Downloads the results

Requirements:
pip install google-genai

Before running:
export GEMINI_API_KEY="your_key"
"""

import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ================================
# Configuration
# ================================
DATASET_NAME = "KLAR"
MODEL_NAME = "gemini-2.5-flash"
INPUT_JSONL = f"/Users/anniewang/Desktop/MLLM-interpretability/MLLM-interpretability/data/{DATASET_NAME}/raw/{DATASET_NAME}_batch_requests.jsonl"
OUTPUT_JSONL = f"/Users/anniewang/Desktop/MLLM-interpretability/MLLM-interpretability/eval/artifacts/{MODEL_NAME}/{DATASET_NAME}_batch_results.jsonl"

# INPUT_JSONL = "test_batch_requests.jsonl"
# OUTPUT_JSONL = "test_results.jsonl"


# polling interval in seconds
POLL_INTERVAL = 45


# ================================
# Initialize client
# ================================

print("Initializing Gemini client...")
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)   # reads GEMINI_API_KEY from environment

# ================================
# Batch job checkpoint file
# ================================

JOB_ID_FILE = "batch_job_id.txt"


# ================================
# Step 1 — Upload JSONL file
# ================================

print("Uploading batch request file...")

uploaded_file = client.files.upload(
    file=INPUT_JSONL,
    config=types.UploadFileConfig(display_name=f'{DATASET_NAME}-batch-requests', mime_type='jsonl')
)

print("Uploaded file name:", uploaded_file.name)


# ================================
# Step 2 — Create or Resume batch job
# ================================

if os.path.exists(JOB_ID_FILE):

    # resume existing job
    with open(JOB_ID_FILE, "r") as f:
        batch_job_name = f.read().strip()

    print("Resuming existing batch job:", batch_job_name)

    batch_job = client.batches.get(name=batch_job_name)

else:

    print("Submitting new batch job...")

    batch_job = client.batches.create(
        model=MODEL_NAME,
        src=uploaded_file.name,
    )

    print("Batch job created:", batch_job.name)

    # save job ID so we can resume later
    with open(JOB_ID_FILE, "w") as f:
        f.write(batch_job.name)



# ================================
# Step 3 — Wait for job to finish
# ================================

print("Waiting for batch job to complete...")

start_time = time.time()

while True:

    job = client.batches.get(name=batch_job.name)

    elapsed = int(time.time() - start_time)

    print(f"[{elapsed}s] Current job state:", job.state)

    if job.state in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
        break

    time.sleep(POLL_INTERVAL)

# ================================
# Step 4 — Handle results
# ================================

if job.state == "JOB_STATE_SUCCEEDED":

    print("Batch job succeeded. Downloading results...")

    output_file_name = job.dest.file_name

    result_bytes = client.files.download(file=output_file_name)


    with open(OUTPUT_JSONL, "wb") as f:
        f.write(result_bytes)
    
    # remove checkpoint after success
    if os.path.exists(JOB_ID_FILE):
        os.remove(JOB_ID_FILE)

    print("Results saved to:", OUTPUT_JSONL)

else:

    print("Batch job did not succeed.")
    print("Final state:", job.state)