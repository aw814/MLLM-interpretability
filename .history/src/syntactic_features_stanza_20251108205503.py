"""
src/feature_engineering/syntactic_features.py
Extract syntactic complexity features from ECLeKTic QA dataset.
"""
import spacy
import os
os.environ["STANZA_RESOURCES_DIR"] = "/project/6101776/xzhan576/stanza_resources"

import pandas as pd
import spacy_stanza
from tqdm import tqdm
# ---- temporary fix for PyTorch 2.6+ weight-only load issue ----
import torch
torch.serialization.add_safe_globals([__import__("numpy").core.multiarray._reconstruct])

import stanza

# Download once (only the first time)
stanza.download('en', model_dir=os.environ["STANZA_RESOURCES_DIR"])
# ---------------------------------------------------------------

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_PATH = os.path.join(BASE_DIR, "MLLM-interpretability/data/processed/eclektic_long_subset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "MLLM-interpretability/features/syntactic_complexity_multilang.csv")
SUPPORTED_LANGS = ["en", "fr", "zh", "he"]  

# -------------------------------------------------------------
# Core functions
# -------------------------------------------------------------
def get_parser(lang_code):

    import stanza, torch, os

    os.environ["STANZA_RESOURCES_DIR"] = "/project/6101776/xzhan576/stanza_resources"

    _torch_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _torch_load(*args, **kwargs)
    torch.load = patched_load

    try:
        print(f"🔹 Loading Stanza model for {lang_code} ...")
        stanza.download(lang_code, model_dir=os.environ["STANZA_RESOURCES_DIR"])
        nlp = stanza.Pipeline(
            lang_code,
            use_gpu=True,
            model_dir=os.environ["STANZA_RESOURCES_DIR"]
        )
    except Exception as e:
        print(f"❌ Failed to load {lang_code}. Try stanza.download('{lang_code}') manually.")
        raise e
    finally:
        torch.load = _torch_load  

    return nlp


def extract_syntactic_features(text, nlp):
    """Compute syntactic complexity metrics for one question using Stanza."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"avg_dep_depth": 0, "max_tree_depth": 0, "num_clauses": 0}

    doc = nlp(text)

    depths = []
    num_clauses = 0

    for sent in doc.sentences:
        for w in sent.words:

            depth = 0
            head = w.head
            while head > 0:
                head = sent.words[head - 1].head
                depth += 1
            depths.append(depth)

            # 判断是否为从句关系
            if w.deprel in ["ccomp", "advcl", "relcl", "acl"]:
                num_clauses += 1

    if not depths:
        return {"avg_dep_depth": 0, "max_tree_depth": 0, "num_clauses": 0}

    avg_depth = sum(depths) / len(depths)
    max_depth = max(depths)

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
