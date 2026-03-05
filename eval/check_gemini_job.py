from google import genai
import os
import time
from dotenv import load_dotenv
load_dotenv()
BATCH_NAME = "batches/bg1sbpm4x3d6uxpj0x6v0cizx9ao81oxdc0e"

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

while True:
    try:
        job = client.batches.get(name=BATCH_NAME)
        print("state:", job.state)
        if str(job.state).endswith("SUCCEEDED") or str(job.state).endswith("FAILED") or str(job.state).endswith("CANCELLED"):
            print(job)
            break
        time.sleep(30)
    except Exception as e:
        # network/DNS hiccup: keep going instead of crashing
        print("poll error:", repr(e))
        time.sleep(30)