import os
import requests
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


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


# Example usage:
if __name__ == "__main__":
    openai_client = OpenAIClient()
    messages = [{"role": "user", "content": "Hello, who are you?"}]
    reply = openai_client.chat("gpt-4o-mini", messages)
    print("OpenAI:", reply)

    openrouter_client = OpenRouterClient()
    reply2 = openrouter_client.chat("meta-llama/llama-3.1-70b-instruct", messages)
    print("OpenRouter:", reply2)
