"""
src/feature_engineering/syntactic_features_stanza.py
Extract syntactic complexity features for both source and target questions
from ECLeKTic QA dataset (multilingual).
"""
import os
import pandas as pd
from tqdm import tqdm
import torch
import stanza

# -------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------
os.environ["STANZA_RESOURCES_DIR"] = "/project/6101776/xzhan576/stanza_resources"

# Fix for PyTorch 2.6+ weights-only load issue
torch.serialization.add_safe_globals([__import__("numpy").core.multiarray._reconstruct])

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_PATH = os.path.join(BASE_DIR, "MLLM-interpretability/data/processed/eclektic_long_subset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "MLLM-interpretability/data/processed/syntactic_complexity.csv")

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def get_parser(lang_code: str):
    """Load or initialize a Stanza pipeline for the given language."""
    _torch_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _torch_load(*args, **kwargs)

    torch.load = patched_load

    try:
        print(f"Loading Stanza model for {lang_code} ...")
        stanza.download(lang_code, model_dir=os.environ["STANZA_RESOURCES_DIR"])
        nlp = stanza.Pipeline(
            lang_code,
            use_gpu=True,
            model_dir=os.environ["STANZA_RESOURCES_DIR"],
            processors="tokenize,pos,lemma,depparse"
        )
    except Exception as e:
        print(f"Failed to load {lang_code}. Try stanza.download('{lang_code}') manually.")
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
            # Compute dependency depth
            depth = 0
            head = w.head
            while head > 0:
                head = sent.words[head - 1].head
                depth += 1
            depths.append(depth)

            # Count clause-like dependencies
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
    """Process all questions (original + target) for one target language."""
    nlp = get_parser(lang_code)
    tqdm.pandas(desc=f"Parsing {lang_code}")

    # Compute features for target question
    synt_feats_target = df["question"].progress_apply(lambda x: extract_syntactic_features(x, nlp))
    synt_df_target = pd.DataFrame(list(synt_feats_target)).add_prefix("target_")

    # Compute features for original question
    synt_feats_source = df["original_question"].progress_apply(lambda x: extract_syntactic_features(x, nlp))
    synt_df_source = pd.DataFrame(list(synt_feats_source)).add_prefix("source_")

    merged = pd.concat([df.reset_index(drop=True), synt_df_target, synt_df_source], axis=1)
    return merged


# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------
def main():
    os.makedirs(os.path.join(BASE_DIR, "features"), exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} QA samples from {INPUT_PATH}")

    # Automatically detect all languages present in the dataset
    SUPPORTED_LANGS = sorted(df["language"].dropna().unique().tolist())
    print(f"Detected languages: {SUPPORTED_LANGS}")

    all_results = []
    for lang in SUPPORTED_LANGS:
        subset = df[df["language"] == lang]
        if subset.empty:
            print(f"No data for language {lang}, skipping.")
            continue
        print(f"\n=== Processing {lang} ({len(subset)} samples) ===")
        processed = process_language_subset(subset, lang)
        all_results.append(processed)

    final = pd.concat(all_results)
    final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Syntactic complexity features (source + target) saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
