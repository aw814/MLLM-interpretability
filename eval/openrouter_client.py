import os
import requests
from dotenv import load_dotenv
from typing import Optional, Tuple
import re
from google import genai
from google.genai import types
from gemini_utils import submit_gemini_job, checkback, message2gemini_request
import time

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_BASE = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _extract_retry_after(error_obj: dict) -> float | None:
    """
    Parse retry delay from Google error payloads.
    Supports detail.retryDelay like '48s' or '650ms' and message text 'retry in Xms/Xs'.
    """
    try:
        details = error_obj.get("details", [])
        for detail in details:
            retry = detail.get("retryDelay")
            if isinstance(retry, str):
                if retry.endswith("s"):
                    return float(retry[:-1])
                if retry.endswith("ms"):
                    return float(retry[:-2]) / 1000.0
    except Exception:
        pass

    msg = error_obj.get("message", "")
    if isinstance(msg, str):
        m = re.search(r"retry in ([0-9.]+)ms", msg)
        if m:
            return float(m.group(1)) / 1000.0
        m = re.search(r"retry in ([0-9.]+)s", msg)
        if m:
            return float(m.group(1))
    return None

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = base_url or OPENROUTER_BASE
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY (set it in .env)")

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Malformed OpenRouter response: {data}") from e


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY (set it in .env)")

    def chat(self, model: str, messages: list[dict], temperature: float = 0.3, max_tokens: int = 256) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            # "temperature": temperature,
            # "max_tokens": max_tokens,
            "messages": messages,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Malformed OpenAI response: {data}") from e

class GoogleClient:
    """
    Minimal Gemini client using the official google-genai SDK.

    This keeps the same interface as the previous HTTP-based version:
    - Reads GOOGLE_API_KEY from the environment by default.
    - Exposes a .chat(model, messages, temperature, max_tokens) method.
    - Uses simple ChatML-style messages: [{"role": "user"|"assistant", "content": "..."}]
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # Preserve the same env-based behavior as before.
        # NOTE: base_url is no longer needed when using the SDK; we keep it in the
        # signature for compatibility but do not use it.
        self.api_key = api_key or GOOGLE_API_KEY
        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY (set it in .env)")

        # google-genai client; it will handle the underlying REST details.
        self.client = genai.Client(api_key=self.api_key)

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> str:
        """
        Simple non-streaming chat wrapper.

        `messages` is a ChatML-style list:
            [{"role": "user"|"assistant", "content": "..."}, ...]
        """

        # Convert ChatML-style messages to Gemini "contents" format.
        contents = []
        for m in messages:
            role = m.get("role", "user")
            # Map 'assistant' -> 'model' for Gemini
            if role == "assistant":
                role = "model"
            elif role not in ("user", "model"):
                # Fallback – Gemini only expects 'user' or 'model'
                role = "user"

            contents.append(
                {
                    "role": role,
                    "parts": [{"text": m.get("content", "")}],
                }
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            resp = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            # Preserve your RateLimitError behavior as best-effort.
            msg = str(e) or e.__class__.__name__
            lowered = msg.lower()
            if "429" in msg or "rate limit" in lowered or "quota" in lowered:
                # The SDK exceptions do not expose a standard retry-after yet,
                # so we propagate None here. Callers can implement generic backoff.
                raise RateLimitError(f"Google Gemini rate limit: {msg}", retry_after=None)
            raise RuntimeError(f"Google Gemini error via google-genai: {msg}") from e

        # The google-genai SDK exposes a convenience .text property.
        try:
            text = (resp.text or "").strip()
            if not text:
                raise ValueError("Empty response text")
            return text
        except Exception as e:
            # Fall back to raw structure for debugging:
            raise RuntimeError(f"Malformed Google Gemini response: {resp}") from e

    def batch_chat(
        self,
        messages_list: list[list[dict]],
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 256,
        batch_size: int = 1000,
        poll_interval: float = 1.0,
        max_polls: int = 60,
        timeout_seconds: float = 120.0,
    ) -> list[str]:
        """
        Batch chat using Gemini batch API.

        Args:
            messages_list: list of ChatML message-lists (one per example), e.g.
                [
                [{"role": "user", "content": "Q1..."}],
                [{"role": "user", "content": "Q2..."}],
                ]
            model: Gemini model name.
            temperature: Sampling temperature.
            max_tokens: (kept for API consistency; gemini_utils doesn't currently pass it through)
            batch_size: number of examples per batch job.
            poll_interval: seconds between polling attempts.
            max_polls: max polling attempts per job.
            timeout_seconds: wall-clock timeout in seconds for polling each batch job.

        Returns:
            List[str] responses aligned to the same order as `messages_list`.
        """
        import time  # ensure time is available

        def _extract_text(resp: object) -> str:
            # dict-like response
            try:
                if isinstance(resp, dict):
                    cands = resp.get("candidates") or []
                    if cands:
                        content = cands[0].get("content") or {}
                        parts = content.get("parts") or []
                        if parts and isinstance(parts[0], dict):
                            return (parts[0].get("text") or "").strip()
                    if "text" in resp:
                        return str(resp["text"]).strip()
            except Exception:
                pass

            # proto-like response (google.genai types)
            try:
                cands = getattr(resp, "candidates", None)
                if cands:
                    content = getattr(cands[0], "content", None)
                    if content:
                        parts = getattr(content, "parts", None)
                        if parts and len(parts) > 0:
                            return (getattr(parts[0], "text", "") or "").strip()
                txt = getattr(resp, "text", None)
                if isinstance(txt, str):
                    return txt.strip()
            except Exception:
                pass

            return str(resp).strip()

        if not messages_list:
            return []

        outputs: list[str] = []

        # Submit in chunks to avoid oversized jobs.
        for start in range(0, len(messages_list), batch_size):
            chunk = messages_list[start : start + batch_size]

            batch_requests = []
            for i, chatml in enumerate(chunk):
                gemini_msgs = chatml
                # Useful for debugging; we still return outputs in request order.
                meta = {"batch_index": str(start + i)}
                batch_requests.append(
                    message2gemini_request(
                        metadata=meta,
                        messages=gemini_msgs,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        model=model,
                    )
                )

            # Submit job
            job = submit_gemini_job(requests=batch_requests, model=model)

            # Poll until complete (bounded by max_polls AND wall-clock timeout)
            responses = None
            deadline = time.time() + float(timeout_seconds)
            last_status_msg = None

            for attempt in range(max_polls):
                try:
                    responses = checkback(job.name)
                    break
                except RuntimeError as e:
                    msg = str(e)
                    last_status_msg = msg
                    # gemini_utils.checkback raises this when not completed yet
                    if "not completed yet" in msg:
                        if time.time() >= deadline:
                            break
                        # lightweight progress print every ~10 polls
                        if attempt % 10 == 0:
                            print(f"[Gemini batch] waiting on job {job.name} (attempt {attempt}/{max_polls})")
                        time.sleep(poll_interval)
                        continue
                    raise

            if responses is None:
                raise RuntimeError(
                    f"Gemini batch job {job.name} did not complete (timeout={timeout_seconds}s, polls={max_polls}). "
                    f"Last status: {last_status_msg}"
                )

            # Responses are aligned to request order.
            for resp in responses:
                outputs.append(_extract_text(resp))

        return outputs


# Example usage:
if __name__ == "__main__":
    messages = [{"role": "user", "content": "Hello, who are you?"}]

    msgs_batch = [
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Who won the World Cup in 2018?"}],
        [{"role": "user", "content": "What is the largest mammal?"}],
        [{"role": "user", "content": "What is the boiling point of water?"}],
        [{"role": "user", "content": "Who is the president of the United States?"}],
        [{"role": "user", "content": "What is the speed of light?"}]
    ]
    # openai_client = OpenAIClient()

    # reply = openai_client.chat("gpt-4o-mini", messages)
    # print("OpenAI:", reply)

    # openrouter_client = OpenRouterClient()
    # reply2 = openrouter_client.chat("meta-llama/llama-3.1-70b-instruct", messages)
    # print("OpenRouter:", reply2)
    google_client = GoogleClient()
    replies = google_client.batch_chat(
            msgs_batch,
            model="gemini-2.5-flash",
            temperature=0.0,
            max_tokens=64,
            batch_size=3,          # force multiple jobs to test chunking
            poll_interval=2.0,
            max_polls=120,
            timeout_seconds=600.0, # give it real time to complete
        )

    for q, r in zip(msgs_batch, replies):
        print("Q:", q[0]["content"])
        print("A:", r)
        print("-" * 50)
