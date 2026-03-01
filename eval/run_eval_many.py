from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
import json

from io_utils import load_config, load_long_csv, load_klar_df
from eval import run_pairwise_eval, run_pairwise_eval_klar
from metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pairwise eval for many target_langs concurrently (from YAML).")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--workers", type=int, default=None, help="Max concurrent targets (default=min(4, len(targets)))")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    dataset_name = cfg.name
    if dataset_name == "eclektic":
        print("[info] Detected dataset: ECLEKTIC. Using ECLEKTIC-specific schema.")
        csv_path = "../data/eclektic/processed/eclektic_long_subset.csv"
        df = load_long_csv(csv_path, cfg.max_examples)
        # Read targets from YAML. Fallback to single target if list not provided.
        targets = cfg.target_lang or []
        if not targets:
            single = cfg.target_lang
            if not single:
                raise ValueError("No targets provided. Add `eval.target_langs: [..]` or `eval.target_lang: 'xx'` in YAML.")
            targets = [single]

        source = cfg.source_lang
        workers = args.workers or min(4, len(targets))
        os.makedirs(cfg.artifacts_dir, exist_ok=True)

        print(f"[info] Source: {source}")
        print(f"[info] Targets: {targets}")
        print(f"[info] Concurrency: {workers}")
        print(f"[info] Artifacts dir: {os.path.join(cfg.artifacts_dir, cfg.tested_model)}")

        # Submit all targets concurrently
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for tgt in targets:
                futures[pool.submit(
                    _run_one_target_ecklectic,
                    df=df,
                    source=source,
                    target=tgt,
                    tested_model=cfg.tested_model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    outdir=os.path.join(cfg.artifacts_dir, cfg.tested_model),
                )] = tgt

            # Collect results
            for fut in as_completed(futures):
                tgt = futures[fut]
                try:
                    rows, uniq = fut.result()
                    print(f"[ok] {source} → {tgt}: rows={rows}, unique q_id={uniq} "
                        f"| preds={os.path.join(cfg.artifacts_dir, f'{tgt}_predictions.csv')} "
                        f"| metrics={os.path.join(cfg.artifacts_dir, f'{tgt}_metrics.json')}")
                except Exception as e:
                    print(f"[error] {source} → {tgt}: {e}")

    elif dataset_name == "klar":
        print("[info] Detected dataset: KLAR. Using KLAR-specific schema.")
        folder = "../data/KLAR/raw/"
        # Read targets from YAML. Fallback to single target if list not provided.
        targets = cfg.target_lang or []
        if not targets:
            single = cfg.target_lang
            if not single:
                raise ValueError("No targets provided. Add `eval.target_langs: [..]` or `eval.target_lang: 'xx'` in YAML.")
            targets = [single]

        source = cfg.source_lang
        workers = args.workers or min(4, len(targets))
        os.makedirs(cfg.artifacts_dir, exist_ok=True)

        print(f"[info] Source: {source}, not meaningful for KLAR, as all languages have the same set of questions.")
        print(f"[info] Targets: {targets}")
        print(f"[info] Concurrency: {workers}")
        print(f"[info] Artifacts dir: {os.path.join(cfg.artifacts_dir, cfg.tested_model)}")

        # Submit all targets concurrently
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for tgt in targets:
                df = load_klar_df(dir, tgt, cfg.max_examples)
                futures[pool.submit(
                    _run_one_target_klar,
                    df=df,
                    source=source,
                    target=tgt,
                    tested_model=cfg.tested_model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    outdir=os.path.join(cfg.artifacts_dir, cfg.tested_model),
                )] = tgt

            # Collect results
            for fut in as_completed(futures):
                tgt = futures[fut]
                try:
                    rows, uniq = fut.result()
                    print(f"[ok] {source} → {tgt}: rows={rows}, unique q_id={uniq} "
                        f"| preds={os.path.join(cfg.artifacts_dir, f'{tgt}_predictions.csv')} "
                        f"| metrics={os.path.join(cfg.artifacts_dir, f'{tgt}_metrics.json')}")
                except Exception as e:
                    print(f"[error] {source} → {tgt}: {e}")

        
    

    
def _run_one_target_ecklectic(
    df,
    source: str,
    target: str,
    tested_model: str,
    temperature: float,
    max_tokens: int,
    outdir: str,
):
    # Informative log for same-language runs (source == target)
    if source == target:
        print(f"[info] Same-language eval: {source} → {target}. Reusing source answers/judgments for target.")

    preds = run_pairwise_eval(
        df=df,
        source_lang=source,
        target_lang=target,
        tested_model=tested_model,
        temperature=temperature,
        max_tokens=max_tokens,
        outdir=outdir,
    )

    # Resumable note: preds may include previously saved + newly generated rows
    try:
        uniq = preds["q_id"].nunique()
    except Exception:
        uniq = len(preds)

    # Compute & save metrics per target
    metrics = compute_metrics(preds)
    metrics_path = os.path.join(outdir, f"{target}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return len(preds), uniq


    
def _run_one_target_klar(
    df,
    source: str,
    target: str,
    tested_model: str,
    temperature: float,
    max_tokens: int,
    outdir: str,
):
   
    preds = run_pairwise_eval_klar(
        df=df,
        source_lang=source,
        target_lang=target,
        tested_model=tested_model,
        temperature=temperature,
        max_tokens=max_tokens,
        outdir=outdir,
    )

    # Resumable note: preds may include previously saved + newly generated rows
    try:
        uniq = preds["q_id"].nunique()
    except Exception:
        uniq = len(preds)

    # Compute & save metrics per target
    metrics = compute_metrics(preds)
    metrics_path = os.path.join(outdir, f"{target}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return len(preds), uniq



if __name__ == "__main__":
    main()