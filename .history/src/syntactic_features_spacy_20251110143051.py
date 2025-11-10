"""
src/feature_engineering/syntactic_features.py
Extract syntactic complexity features from ECLeKTic QA dataset.
"""
import spacy
import os
import pandas as pd
import spacy_stanza
from tqdm import tqdm
# ---- temporary fix for PyTorch 2.6+ weight-only load issue ----
import torch
torch.serialization.add_safe_globals([__import__("numpy").core.multiarray._reconstruct])

# ---------------------------------------------------------------

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_PATH = os.path.join(BASE_DIR, "MLLM-interpretability/data/processed/eclektic_long_subset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "MLLM-interpretability/data/processed/syntactic_complexity_multilang.csv")
SUPPORTED_LANGS = ["en", "fr", "zh", "he"]  

# -------------------------------------------------------------
# Core functions
# -------------------------------------------------------------
def get_parser(lang_code):
    """Load or initialize spaCy-Stanza pipeline for a given language."""
    print(f"🔹 Loading spaCy-Stanza model for: {lang_code}")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"❌ Failed to load {lang_code}. Try: stanza.download('{lang_code}')")
        raise e
    return nlp


def extract_syntactic_features(text, nlp):
    """Compute syntactic complexity metrics for one question."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"avg_dep_depth": 0, "max_tree_depth": 0, "num_clauses": 0}

    doc = nlp(text)
    depths = [len(list(tok.ancestors)) for tok in doc if tok.dep_ != "punct"]
    if not depths:
        return {"avg_dep_depth": 0, "max_tree_depth": 0, "num_clauses": 0}

    avg_depth = sum(depths) / len(depths)
    max_depth = max(depths)
    num_clauses = sum(tok.dep_ in ["ccomp", "advcl", "relcl", "acl"] for tok in doc)

    return {
        "avg_dep_depth": avg_depth,
        "max_tree_depth": max_depth,
        "num_clauses": num_clauses,
    }


def process_language_subset(df, lang_code):
    """Process all questions for one target language."""
    nlp = get_parser(lang_code)
    tqdm.pandas(desc=f"Parsing {lang_code}")
    synt_feats = df["question"].progress_apply(lambda x: extract_syntactic_features(x, nlp))
    synt_df = pd.DataFrame(list(synt_feats))
    merged = pd.concat([df.reset_index(drop=True), synt_df], axis=1)
    return merged


# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------
def main():
    os.makedirs(os.path.join(BASE_DIR, "features"), exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    print(f"✅ Loaded {len(df)} QA samples from {INPUT_PATH}")

    all_results = []
    for lang in SUPPORTED_LANGS:
        subset = df[df["language"] == lang]
        if subset.empty:
            print(f"⚠️ No data for language {lang}, skipping.")
            continue
        print(f"\n=== Processing {lang} ({len(subset)} samples) ===")
        processed = process_language_subset(subset, lang)
        all_results.append(processed)

    final = pd.concat(all_results)
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n🎉 Syntactic complexity features saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
