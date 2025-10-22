from __future__ import annotations
import argparse
import os
import json
from io_utils import load_config, load_long_csv
from eval import run_pairwise_eval
from metrics import compute_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_long_csv(cfg.csv_path, cfg.max_examples)

    # Informative log for same-language runs (source == target)
    same_lang = (cfg.source_lang == cfg.target_lang)
    if same_lang:
        print(f"[info] Same-language evaluation detected: {cfg.source_lang} → {cfg.target_lang}. "
              f"Target answers/judgments will be reused from source.")

    preds = run_pairwise_eval(
        df=df,
        source_lang=cfg.source_lang,
        target_lang=cfg.target_lang,
        tested_model=cfg.tested_model,
        judge_model=cfg.judge_model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        outdir=cfg.artifacts_dir,
    )
    # Note: run_pairwise_eval() is resumable — it skips any q_id already completed in previous runs,
    # so preds may contain both previously saved and newly generated results.
    try:
        uniq = preds["q_id"].nunique()
        print(f"[info] Predictions ready: rows={len(preds)} | unique q_id={uniq}")
    except Exception:
        print(f"[info] Predictions ready: rows={len(preds)}")
    print(f"[info] Predictions file: {os.path.join(cfg.artifacts_dir, f'{cfg.target_lang}_predictions.csv')}")

    metrics = compute_metrics(preds)

    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    with open(os.path.join(cfg.artifacts_dir, f"{cfg.target_lang}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", metrics)
    print("Artifacts in:", os.path.abspath(cfg.artifacts_dir))

if __name__ == "__main__":
    main()
