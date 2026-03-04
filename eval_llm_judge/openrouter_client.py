import os
import requests
from dotenv import load_dotenv
from typing import Optional, Tuple
import re
from google import genai
from google.genai import types

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_BASE = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")


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


# Example usage:
if __name__ == "__main__":
    messages = [{"role": "user", "content": "Hello, who are you?"}]

    # openai_client = OpenAIClient()

    # reply = openai_client.chat("gpt-4o-mini", messages)
    # print("OpenAI:", reply)

    # openrouter_client = OpenRouterClient()
    # reply2 = openrouter_client.chat("meta-llama/llama-3.1-70b-instruct", messages)
    # print("OpenRouter:", reply2)

    google_client = GoogleClient()
    reply3 = google_client.chat("gemini-2.5-flash", messages)
    print("Google Gemini:", reply3)
