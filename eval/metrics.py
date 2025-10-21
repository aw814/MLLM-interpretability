from __future__ import annotations
import json
import os
import pandas as pd

def compute_metrics(preds_df: pd.DataFrame) -> dict:
    g = preds_df.groupby("q_id").agg(
        correct_source=("correct_source", "max"),
        correct_target=("correct_target", "max"),
    ).reset_index()

    overall_success = (g["correct_source"] & g["correct_target"]).mean()

    mask = g["correct_source"]
    transfer = (g.loc[mask, "correct_target"].mean()) if mask.any() else 0.0

    out = {
        "overall_success": float(overall_success),
        "transfer": float(transfer),
        "n_items": int(g.shape[0]),
    }
    return out
