from __future__ import annotations
import os
import yaml
import pandas as pd
from dataclasses import dataclass

DATA_COLUMNS = [
    "q_id",
    "original_lang",
    "language",
    "question",
    "answer",
    "content",
    "original_question",
    "original_answer",
    "original_content",
    "title",
    "url",
]

@dataclass
class Config:
    csv_path: str
    max_examples: int | None
    source_lang: str
    target_lang: str
    tested_model: str
    judge_model: str
    temperature: float
    max_tokens: int
    artifacts_dir: str

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = cfg.get("data", {})
    eval_ = cfg.get("eval", {})
    models = cfg.get("models", {})
    decode = cfg.get("decode", {})
    outdir = cfg.get("artifacts_dir", "./artifacts")

    os.makedirs(outdir, exist_ok=True)

    resolved = {
        "csv_path": data.get("csv_path"),
        "max_examples": data.get("max_examples", None),
        "source_lang": eval_.get("source_lang", "en"),
        "target_lang": eval_.get("target_lang", "fr"),
        "tested_model": models.get("tested_model"),
        "judge_model": models.get("judge_model"),
        "temperature": float(decode.get("temperature", 0.0)),
        "max_tokens": int(decode.get("max_tokens", 128)),
        "artifacts_dir": outdir,
    }

    # persist resolved config for provenance
    with open(os.path.join(outdir, "run_config.resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    return Config(**resolved)  # type: ignore

def load_long_csv(csv_path: str, max_examples: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        # Allow partial schema: only strictly required columns for this pipeline
        req = ["q_id", "original_lang", "language", "question", "content"]
        req_missing = [c for c in req if c not in df.columns]
        if req_missing:
            raise ValueError(f"CSV missing required columns: {req_missing}")
    if max_examples:
        # sample by unique q_id to ensure both languages are likely present
        unique_ids = df["q_id"].dropna().unique().tolist()[:max_examples]
        df = df[df["q_id"].isin(unique_ids)]
    return df
