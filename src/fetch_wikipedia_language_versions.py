#!/usr/bin/env python3
"""Fetch the number of Wikipedia language editions for articles in the dataset.

The script reads the processed Ecklectic dataset, collects the unique article
URLs, queries the Wikipedia API for each article, and counts how many language
editions exist for that page. The resulting counts are joined back to the
original dataset and saved as a new CSV file.

Usage (from repository root):
    python src/fetch_wikipedia_language_versions.py \
        --input data/processed/eclektic_long_subset.csv \
        --output data/processed/eclektic_long_article_language_counts.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

API_URL_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
REQUEST_TIMEOUT = 15  # seconds
USER_AGENT = "MLLM-interpretability/0.1 (github.com/aw814)"
SLEEP_BETWEEN_CALLS = 0.2  # seconds, friendly to the API


@dataclass
class ArticleLanguageCount:
    """Holds the language count metadata for a single article."""

    url: str
    title: Optional[str]
    language_version_count: Optional[int]
    error: Optional[str] = None


def extract_lang_and_title(article_url: str) -> tuple[str, str]:
    """Return the base language code and page title from a Wikipedia URL."""

    parsed = urlparse(article_url)
    if not parsed.netloc:
        raise ValueError("URL is missing a hostname")

    host_parts = parsed.netloc.split(".")
    if not host_parts or host_parts[0] == "www":
        raise ValueError("URL does not include a language subdomain")

    base_lang = host_parts[0]
    if not parsed.path.startswith("/wiki/"):
        raise ValueError("URL path is not a standard /wiki/ article")

    title = unquote(parsed.path[len("/wiki/") :])
    if not title:
        raise ValueError("URL does not include an article title")

    return base_lang, title


def fetch_language_count(
    base_lang: str,
    title: str,
    session: requests.Session,
    max_retries: int = 3,
) -> int:
    """Return how many language editions exist for the given article."""

    params: Dict[str, str] = {
        "action": "query",
        "format": "json",
        "prop": "langlinks",
        "titles": title,
        "lllimit": "500",
        "redirects": "1",
    }

    total_links = 0
    continuation: Optional[Dict[str, str]] = None

    while True:
        request_params = dict(params)
        if continuation:
            request_params.update(continuation)

        for attempt in range(max_retries):
            try:
                response = session.get(
                    API_URL_TEMPLATE.format(lang=base_lang),
                    params=request_params,
                    headers={"User-Agent": USER_AGENT},
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                is_last_attempt = attempt == max_retries - 1
                if is_last_attempt:
                    raise RuntimeError(str(exc)) from exc
                time.sleep(2**attempt)
        else:
            raise RuntimeError("Exceeded maximum retries without a response")

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})

        if "missing" in page:
            return 0

        langlinks = page.get("langlinks", [])
        total_links += len(langlinks)

        continuation = data.get("continue")
        if not continuation:
            break

    # The langlinks list excludes the article's own language, so add one.
    return total_links + 1


def collect_language_counts(urls: pd.Series) -> Dict[str, ArticleLanguageCount]:
    """Query Wikipedia for each article and return their language counts."""

    results: Dict[str, ArticleLanguageCount] = {}
    session = requests.Session()

    for idx, article_url in enumerate(urls, start=1):
        try:
            base_lang, title = extract_lang_and_title(article_url)
            language_count = fetch_language_count(base_lang, title, session)
            results[article_url] = ArticleLanguageCount(
                url=article_url,
                title=title,
                language_version_count=language_count,
            )
            print(f"[{idx}/{len(urls)}] {title} -> {language_count}")
            time.sleep(SLEEP_BETWEEN_CALLS)
        except Exception as exc:  # noqa: BLE001 - log and continue
            results[article_url] = ArticleLanguageCount(
                url=article_url,
                title=None,
                language_version_count=None,
                error=str(exc),
            )
            print(f"[{idx}/{len(urls)}] {article_url} -> ERROR: {exc}", file=sys.stderr)
            time.sleep(SLEEP_BETWEEN_CALLS)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia language counts for Ecklectic articles.",
    )
    parser.add_argument(
        "--input",
        default="data/processed/eclektic_long_subset.csv",
        help="Path to the Ecklectic dataset (CSV).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/eclektic_long_article_language_counts.csv",
        help="Where to save the enriched dataset (CSV).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if "url" not in df.columns:
        raise KeyError("The input dataset must include a 'url' column.")

    unique_urls = (
        df["url"].dropna().astype(str).drop_duplicates().reset_index(drop=True)
    )
    if unique_urls.empty:
        raise ValueError("No article URLs found in the input dataset.")

    counts = collect_language_counts(unique_urls)

    counts_df = pd.DataFrame(
        {
            "url": [entry.url for entry in counts.values()],
            "title": [entry.title for entry in counts.values()],
            "language_version_count": [
                entry.language_version_count for entry in counts.values()
            ],
            "error": [entry.error for entry in counts.values()],
        }
    )

    merged_df = df.merge(
        counts_df[["url", "language_version_count"]], on="url", how="left"
    )

    required_columns = ["q_id", "title", "url", "language_version_count"]
    missing_required = [col for col in required_columns if col not in merged_df.columns]
    if missing_required:
        raise KeyError(
            "Missing required columns in merged dataset: " + ", ".join(missing_required)
        )

    final_df = merged_df[required_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"Saved merged dataset with language counts to {output_path}")


if __name__ == "__main__":
    main()
