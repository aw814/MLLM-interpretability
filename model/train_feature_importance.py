#!/usr/bin/env python
"""
More detailed analysis of feature importance for interpretable models.

- Reuses data loading + preprocessing from train.py
- Fits LogisticRegression, RandomForest, and LDA per feature set
- Extracts coefficients / importances for transformed features
- Heuristically maps encoded features back to original feature names
- Writes a single CSV with all importances:
    model_feature_importances.csv
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Import shared config + helpers from train.py
from train import (
    DATA_PATH,
    TARGET_COLUMN,
    ID_COLUMNS,
    FEATURE_SETS,
    load_data,
    filter_rows,
    detect_feature_types,
    build_preprocessor,
)

# Flattened list of base feature column names (excluding the 'all' meta-set)
BASE_FEATURE_COLS = sorted(
    {
        col
        for set_name, cols in FEATURE_SETS.items()
        if set_name != "all"
        for col in cols
    },
    key=len,
    reverse=True,
)


def get_interpretable_models() -> Dict[str, Any]:
    """
    Subset of models for which we can extract reasonably interpretable
    feature importance:
      - LogisticRegression: coefficients
      - RandomForestClassifier: feature_importances_
      - LinearDiscriminantAnalysis: coefficients
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "LDA": LinearDiscriminantAnalysis(),
    }
    return models


def collapse_multiclass_coefs(coef: np.ndarray) -> np.ndarray:
    """
    Handle coef_ shapes for multi-class models.

    - Binary (2 classes): use the coefficients for the positive class
    - Multi-class (>2 classes): use the L2 norm across classes for each feature
    """
    if coef.ndim == 1:
        return coef

    # shape (n_classes, n_features)
    n_classes, n_features = coef.shape
    if n_classes == 2:
        # take coefficients for positive class
        return coef[1]
    else:
        # L2 norm across classes per feature
        return np.linalg.norm(coef, axis=0)


def infer_original_feature_name(encoded_name: str) -> str:
    """Infer the original feature column name from an encoded (possibly OHE) feature.

    One-hot encoded names often look like:
        'question_type_scientific_technical'
        'target_genus_Malayo-Polynesian'

    This function:
    - checks all base columns in BASE_FEATURE_COLS
    - returns the longest column name that is a prefix of encoded_name
    - falls back to splitting on the first underscore if nothing matches.
    """
    # Prefer longest-prefix matches first (BASE_FEATURE_COLS is sorted by length desc)
    for col in BASE_FEATURE_COLS:
        if encoded_name == col or encoded_name.startswith(col + "_"):
            return col

    # Fallback: previous simple heuristic
    if "_" in encoded_name:
        return encoded_name.split("_", 1)[0]

    return encoded_name


def infer_original_feature_set(orig_name: str) -> str:
    """Infer which feature set a base feature column belongs to.

    Takes the original feature column name (e.g., 'question_type', 'target_genus')
    and returns the name of the feature set in FEATURE_SETS (excluding 'all') that
    contains it. Returns 'unknown' if no match is found.
    """
    for set_name, cols in FEATURE_SETS.items():
        if set_name == "all":
            continue
        if orig_name in cols:
            return set_name

    # Optional loose fallback: if orig_name partially matches any column name
    for set_name, cols in FEATURE_SETS.items():
        if set_name == "all":
            continue
        for col in cols:
            if col in orig_name or orig_name in col:
                return set_name

    return "unknown"


def extract_importances_for_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    present_cols: List[str],
    feature_set_name: str,
) -> List[Dict[str, Any]]:
    """
    Fit a pipeline (preprocessor + model) and extract feature importances.

    Returns a list of dicts, one per transformed feature.
    """
    if not present_cols:
        return []

    # Detect feature types and build preprocessor as in train.py
    numeric_cols, cat_cols = detect_feature_types(X, present_cols)
    pre = build_preprocessor(numeric_cols, cat_cols)

    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
    pipe.fit(X[present_cols], y)

    pre_step = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # Get names for transformed features
    # This reflects scaling + one-hot encoded columns
    feature_names = list(pre_step.get_feature_names_out(present_cols))

    # Determine raw importance depending on model type
    if hasattr(clf, "coef_"):
        # LogisticRegression / LDA
        coef = clf.coef_
        coef_collapsed = collapse_multiclass_coefs(coef)
        raw_importances = np.asarray(coef_collapsed, dtype=float)
    elif hasattr(clf, "feature_importances_"):
        # RandomForest
        raw_importances = np.asarray(clf.feature_importances_, dtype=float)
    else:
        # No straightforward importance (e.g., SVM, KNN) -> skip
        return []

    if raw_importances.shape[0] != len(feature_names):
        # Something is inconsistent; skip to be safe
        print(
            f"[WARN] Importance length mismatch for model {model_name}, "
            f"feature_set {feature_set_name}: "
            f"{raw_importances.shape[0]} vs {len(feature_names)}"
        )
        return []

    rows: List[Dict[str, Any]] = []
    for fname, imp_raw in zip(feature_names, raw_importances):
        orig_name = infer_original_feature_name(fname)
        orig_set = infer_original_feature_set(orig_name)
        rows.append(
            {
                "feature_set": feature_set_name,
                "model": model_name,
                "encoded_feature": fname,
                "orig_feature": orig_name,
                "orig_feature_set": orig_set,
                "raw_importance": float(imp_raw),
                "abs_importance": float(abs(imp_raw)),
            }
        )

    return rows


def main():
    # Load data (same hygiene as train.py)
    df = load_data(DATA_PATH)
    df = filter_rows(df)

    for c in ID_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=[c])

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"TARGET_COLUMN '{TARGET_COLUMN}' not found in data. "
            f"Available columns: {list(df.columns)[:20]} ..."
        )

    # Drop rows with missing labels
    df = df[~df[TARGET_COLUMN].isna()].copy()

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    models = get_interpretable_models()

    all_rows: List[Dict[str, Any]] = []

    for feature_set_name, cols in FEATURE_SETS.items():
        present = [c for c in cols if c in X.columns]
        missing = sorted(set(cols) - set(present))
        if not present:
            print(
                f"[WARN] Feature set '{feature_set_name}' has no columns present in data. Skipping."
            )
            continue
        if missing:
            print(
                f"[INFO] Feature set '{feature_set_name}': missing columns ignored -> {missing}"
            )

        for model_name, model in models.items():
            print(
                f"[INFO] Fitting {model_name} for feature set '{feature_set_name}' "
                f"with {len(present)} input columns..."
            )
            rows = extract_importances_for_model(
                model_name=model_name,
                model=model,
                X=X,
                y=y,
                present_cols=present,
                feature_set_name=feature_set_name,
            )
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(
            "No feature importance rows produced. "
            "Check that your feature columns exist and models expose importances."
        )

    results = pd.DataFrame(all_rows)

    # Sort for convenience: by feature_set, model, and descending abs importance
    results = results.sort_values(
        ["feature_set", "model", "abs_importance"], ascending=[True, True, False]
    )

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model_feature_importances.csv",
    )
    results.to_csv(out_path, index=False)

    # Pretty print some toplines
    with pd.option_context(
        "display.max_rows", 50,
        "display.max_columns", None,
        "display.width", 120,
    ):
        print("\n=== Top Feature Importances (head) ===")
        print(results.head(50).to_string(index=False))
        print(f"\nSaved detailed feature importances to: {out_path}")


if __name__ == "__main__":
    main()