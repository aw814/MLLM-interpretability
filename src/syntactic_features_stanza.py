"""
src/feature_engineering/syntactic_features_stanza.py
Extract syntactic complexity features for both source and target questions
from ECLeKTic QA dataset (multilingual).

Metrics implemented based on Universal Dependencies:
1. Mean Dependency Distance (MDD) - Linear processing burden (Liu, 2008)
2. Max Tree Depth (Tree Height) - Hierarchical embedding complexity
3. Crossing Dependencies Rate - Measure of word order freedom/non-projectivity
"""
import os
import pandas as pd
from tqdm import tqdm
import torch
import stanza
import numpy as np

# -------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------
# Ensure this path is correct for your environment
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
    # Patching torch.load to avoid weights_only error in newer PyTorch versions
    _torch_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _torch_load(*args, **kwargs)

    torch.load = patched_load

    try:
        print(f"Loading Stanza model for language: {lang_code} ...")
        # Ensure model is downloaded
        stanza.download(lang_code, model_dir=os.environ["STANZA_RESOURCES_DIR"])
        
        # Initialize pipeline
        nlp = stanza.Pipeline(
            lang_code,
            use_gpu=True,
            model_dir=os.environ["STANZA_RESOURCES_DIR"],
            processors="tokenize,pos,lemma,depparse",
            verbose=False 
        )
    except Exception as e:
        print(f"Failed to load {lang_code}. Try stanza.download('{lang_code}') manually.")
        raise e
    finally:
        torch.load = _torch_load

    return nlp


def calculate_tree_depth(sentence):
    """
    Calculate the depth of each node in the dependency tree.
    Returns: A dictionary {word_id: depth}, where Root depth is 0.
    """
    # Build adjacency list: id -> head
    heads = {word.id: word.head for word in sentence.words}
    depths = {}
    
    # Stanza IDs are 1-based, 0 is Root
    # We use dynamic programming (memoization) to avoid redundant traversals
    def get_depth(node_id):
        if node_id == 0:
            return 0
        if node_id in depths:
            return depths[node_id]
        
        # Recursively find depth of parent
        parent_depth = get_depth(heads[node_id])
        depths[node_id] = parent_depth + 1
        return depths[node_id]

    for word in sentence.words:
        # Protect against potential cycles (though unlikely in valid UD trees)
        try:
            get_depth(word.id)
        except RecursionError:
            depths[word.id] = 0 # Fallback for cycle
            
    return depths.values()


def calculate_crossings(sentence):
    """
    Calculate the number of crossing dependencies (non-projectivity).
    An arc (i, j) crosses (k, l) if min(i,j) < min(k,l) < max(i,j) < max(k,l).
    """
    arcs = []
    # Collect all dependency arcs excluding Root connections
    for word in sentence.words:
        if word.head == 0:
            continue
        # Store as (start, end) where start < end
        start, end = sorted((word.id, word.head))
        arcs.append((start, end))
    
    crossings = 0
    n_arcs = len(arcs)
    if n_arcs < 2:
        return 0
        
    for i in range(n_arcs):
        for j in range(i + 1, n_arcs):
            s1, e1 = arcs[i]
            s2, e2 = arcs[j]
            # Check crossing condition
            if (s1 < s2 < e1 < e2) or (s2 < s1 < e2 < e1):
                crossings += 1
                
    return crossings


def extract_syntactic_features(text, nlp):
    """
    Compute Universal Dependencies based syntactic complexity metrics.
    Metrics:
    1. MDD (Mean Dependency Distance)
    2. Max Tree Depth (Tree Height)
    3. Crossing Rate (Structural Complexity)
    """
    # 1. Basic validation
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "mdd": 0, "max_tree_depth": 0, "crossing_rate": 0, "sentence_count": 0
        }

    try:
        doc = nlp(text)
    except Exception:
        return {
            "mdd": 0, "max_tree_depth": 0, "crossing_rate": 0, "sentence_count": 0
        }

    # 2. Initialize aggregation lists
    all_distances = []
    all_max_depths = []
    total_arcs = 0
    total_crossings = 0
    
    for sent in doc.sentences:
        # --- Metric 1: Dependency Distance (MDD) ---
        # Distance = |Dependent_Index - Head_Index|
        # Exclude Root connections (head=0) from MDD calculation as per standard practice
        dists = [abs(w.id - w.head) for w in sent.words if w.head != 0]
        all_distances.extend(dists)
        
        # --- Metric 2: Tree Depth ---
        # Calculate max depth for this sentence
        sent_depths = calculate_tree_depth(sent)
        if sent_depths:
            all_max_depths.append(max(sent_depths))
        else:
            all_max_depths.append(0)
            
        # --- Metric 3: Crossing Dependencies ---
        # Measures non-projectivity (common in free word order languages)
        n_cross = calculate_crossings(sent)
        n_arcs = len(dists) # Number of valid dependency arcs
        
        total_crossings += n_cross
        total_arcs += n_arcs

    # 3. Final Aggregation
    mdd = np.mean(all_distances) if all_distances else 0
    # For text with multiple sentences, we can take the mean of the max depths, 
    # or the absolute max depth. Mean of max depths represents average sentence complexity.
    avg_max_tree_depth = np.mean(all_max_depths) if all_max_depths else 0
    
    # Crossing rate: crossings per dependency arc
    crossing_rate = total_crossings / total_arcs if total_arcs > 0 else 0

    return {
        "mdd": mdd,                            # 平均依存距离 (Linear Complexity)
        "max_tree_depth": avg_max_tree_depth,  # 平均最大树深 (Hierarchical Complexity)
        "crossing_rate": crossing_rate,        # 交叉依存率 (Word Order Complexity)
        "sentence_count": len(doc.sentences)
    }


def process_language_subset(df, target_lang_code):
    """
    Process all questions for one target language group.
    """
    # 1. Load target language model
    nlp_target = get_parser(target_lang_code)
    
    # 2. Identify source language (assuming consistent within chunk)
    source_lang_code = df["original_lang"].iloc[0]
    
    # Reuse model if source == target
    if source_lang_code == target_lang_code:
        print(f"Source and Target are both {target_lang_code}. Reusing model.")
        nlp_source = nlp_target
    else:
        print(f"Source ({source_lang_code}) != Target ({target_lang_code}). Loading source model...")
        nlp_source = get_parser(source_lang_code)

    # Process Target Questions
    tqdm.pandas(desc=f"Parsing Target ({target_lang_code})")
    synt_feats_target = df["question"].progress_apply(lambda x: extract_syntactic_features(x, nlp_target))
    synt_df_target = pd.DataFrame(list(synt_feats_target)).add_prefix("target_")

    # Process Source Questions
    tqdm.pandas(desc=f"Parsing Source ({source_lang_code})")
    synt_feats_source = df["original_question"].progress_apply(lambda x: extract_syntactic_features(x, nlp_source))
    synt_df_source = pd.DataFrame(list(synt_feats_source)).add_prefix("source_")

    # Merge results
    merged = pd.concat([df.reset_index(drop=True), synt_df_target, synt_df_source], axis=1)
    
    return merged


# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} QA samples from {INPUT_PATH}")

    if "language" not in df.columns:
        raise ValueError("Input CSV must have a 'language' column.")
        
    SUPPORTED_LANGS = sorted(df["language"].dropna().unique().tolist())
    print(f"Detected target languages: {SUPPORTED_LANGS}")

    all_results = []
    
    for lang in SUPPORTED_LANGS:
        subset = df[df["language"] == lang].copy()
        if subset.empty:
            continue
            
        print(f"\n==============================================")
        print(f"Processing Subset: Target Language = {lang} ({len(subset)} rows)")
        print(f"==============================================")
        
        try:
            processed = process_language_subset(subset, lang)
            all_results.append(processed)
        except Exception as e:
            print(f"Error processing language {lang}: {e}")
            continue

    if all_results:
        final = pd.concat(all_results)
        final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"\nSuccess! Syntactic complexity features saved to: {OUTPUT_PATH}")
        print(f"Total rows processed: {len(final)}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()