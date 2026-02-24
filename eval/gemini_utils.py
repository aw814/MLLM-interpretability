from typing import List, Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import ClientError
import time

__CACHE__ = {
    "genai_client": None,
}

def message2gemini_request(
    metadata: Dict[Any, Any],
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 256,
    model: str = 'gemini-2.5-flash',
) -> dict:
    """
    Build a Gemini batch GenerateContentRequest.
    - System message is OPTIONAL (detected by role == 'system').
    - temperature and max_tokens are per-request.
    """

    system_msg = None
    contents = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            system_msg = content
            continue

        if role == "assistant":
            role = "model"
        elif role not in ("user", "model"):
            role = "user"

        contents.append(
            {
                "role": role,
                "parts": [{"text": content}],
            }
        )

    config = {
        "temperature": temperature,
        "response_modalities": ["text"],
        "max_output_tokens": max_tokens,
    }

    if system_msg is not None:
        config["system_instruction"] = system_msg

    return {
        "model": model,
        "contents": contents,
        "config": config,
        "metadata": metadata,
    }
    
def call_gemini_one_by_one_api(
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 256,
    model: str = 'gemini-2.5-flash',
) -> types.GenerateContentResponse:

    client = __CACHE__.get("genai_client", None)
    if not client:
        client = genai.Client()
        __CACHE__["genai_client"] = client

    system_msg = None
    contents = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            system_msg = content
            continue

        if role == "assistant":
            role = "model"
        elif role not in ("user", "model"):
            role = "user"

        contents.append({
            "role": role,
            "parts": [{"text": content}],
        })

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_msg,
    )

    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

def submit_gemini_job(
    requests: List[Dict[str, Any]],
    model: str = 'gemini-2.5-flash'
) -> types.BatchJob:
    """
    Submit a Gemini batch job with the given requests. Return the job information.
    """
    client = __CACHE__.get("genai_client", None)
    if not client:
        client = genai.Client()
        __CACHE__["genai_client"] = client
    try:
        batch_job = client.batches.create(
            model=model,
            src=requests,
        )
    except ClientError as err:
        if err.code == 429:
            print(err)
            print("Sleep for 10 minutes and retry...")
            time.sleep(600)
            batch_job = submit_gemini_job(
                requests=requests,
                model=model
            )
            return batch_job
        else:
            raise err
    return batch_job

def checkback(job_name: str) -> List[Dict[str, Any]]:
    """
    Checkback a Gemini batch job and return the list of responses.
    Args:
        job_name: the name of the Gemini batch job returned by submit_gemini_job
    Returns:
        A list of dicts containing the responses for each request.
    """
    client = __CACHE__.get("genai_client", None)
    if not client:
        client = genai.Client()
        __CACHE__["genai_client"] = client
    job = client.batches.get(name=job_name)
    if job.state == types.JobState.JOB_STATE_SUCCEEDED:
        responses = []
        for response in job.dest.inlined_responses:
            responses.append(response)
        return responses
    elif job.state in {types.JobState.JOB_STATE_CANCELLED, types.JobState.JOB_STATE_PAUSED, types.JobState.JOB_STATE_FAILED}:
        raise RuntimeError(f"Gemini batch job {job_name} failed with state {job.state}.")
    else:
        raise RuntimeError(f"Gemini batch job {job_name} is not completed yet. Current state: {job.state}.")