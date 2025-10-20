from __future__ import annotations
import argparse
import os
from io_utils import load_config, load_long_csv
from eval import run_pairwise_eval
from metrics import compute_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_long_csv(cfg.csv_path, cfg.max_examples)

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

    metrics = compute_metrics(preds, cfg.artifacts_dir)
    print("Saved metrics:", metrics)
    print("Artifacts in:", os.path.abspath(cfg.artifacts_dir))

if __name__ == "__main__":
    main()
