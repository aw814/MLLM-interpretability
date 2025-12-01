import os
import re
import requests
from dotenv import load_dotenv
from typing import Optional, Tuple

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
    Minimal Gemini client using the Google Generative Language API.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or GOOGLE_API_KEY
        self.base_url = base_url or GOOGLE_BASE
        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY (set it in .env)")

    def chat(self, model: str, messages: list[dict], temperature: float = 0.3, max_tokens: int = 256) -> str:
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        # Flatten simple ChatML-style messages into Gemini "contents"
        contents = []
        for m in messages:
            role = "user" if m.get("role") == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code != 200:
            retry_after = None
            try:
                data_err = r.json().get("error", {})
                retry_after = _extract_retry_after(data_err)
            except Exception:
                pass
            if r.status_code == 429:
                # If service doesn't return a delay, wait at least 1 second to avoid hammering.
                if retry_after is None:
                    retry_after = 1.0
                raise RateLimitError(f"Google Gemini rate limit {r.status_code}: {r.text}", retry_after=retry_after)
            raise RuntimeError(f"Google Gemini error {r.status_code}: {r.text}")
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            raise RuntimeError(f"Malformed Google Gemini response: {data}") from e


# Example usage:
if __name__ == "__main__":
    openai_client = OpenAIClient()
    messages = [{"role": "user", "content": "Hello, who are you?"}]
    reply = openai_client.chat("gpt-4o-mini", messages)
    print("OpenAI:", reply)

    openrouter_client = OpenRouterClient()
    reply2 = openrouter_client.chat("meta-llama/llama-3.1-70b-instruct", messages)
    print("OpenRouter:", reply2)
