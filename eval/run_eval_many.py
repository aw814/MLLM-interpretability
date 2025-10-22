from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
import json

from io_utils import load_config, load_long_csv
from eval import run_pairwise_eval
from metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pairwise eval for many target_langs concurrently (from YAML).")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--workers", type=int, default=None, help="Max concurrent targets (default=min(4, len(targets)))")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    df = load_long_csv(cfg.csv_path, cfg.max_examples)

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
    print(f"[info] Artifacts dir: {os.path.abspath(cfg.artifacts_dir)}")

    # Submit all targets concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for tgt in targets:
            futures[pool.submit(
                _run_one_target,
                df=df,
                source=source,
                target=tgt,
                tested_model=cfg.tested_model,
                judge_model=cfg.judge_model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                outdir=cfg.artifacts_dir,
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


def _run_one_target(
    df,
    source: str,
    target: str,
    tested_model: str,
    judge_model: str,
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
        judge_model=judge_model,
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