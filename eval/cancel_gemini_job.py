"""
Cancel a Gemini Batch Job

Usage examples:

1) Cancel using checkpoint file:
python cancel_gemini_batch.py

2) Cancel using job id:
python cancel_gemini_batch.py batches/abc123

Requirements:
pip install google-genai

Environment variable required:
export GOOGLE_API_KEY="your_key"
"""

import sys
import os
from google import genai
from dotenv import load_dotenv

# ================================
# Initialize client
# ================================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)


# ================================
# Get job ID
# ================================

if len(sys.argv) > 1:
    job_id = sys.argv[1]
else:
    checkpoint_file = "batch_job_id.txt"

    if not os.path.exists(checkpoint_file):
        raise RuntimeError(
            "No job id provided and batch_job_id.txt not found"
        )

    with open(checkpoint_file) as f:
        job_id = f.read().strip()


print(f"Attempting to cancel job: {job_id}")


# ================================
# Cancel job
# ================================

try:
    client.batches.cancel(name=job_id)
    print("Cancellation request sent.")

except Exception as e:
    print("Error cancelling job:", e)
    sys.exit(1)


# ================================
# Verify state
# ================================

job = client.batches.get(name=job_id)

print("Current job state:", job.state)

if job.state == "JOB_STATE_CANCELLED":
    print("Job successfully cancelled.")
else:
    print("Job may still be cancelling...")