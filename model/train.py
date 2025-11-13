import os
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# Silence some benign warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


DATA_PATH = "/home/ubuntu/MLLM-interpretability/data/processed/training_data.csv"
TARGET_COLUMN = "correct_target"
ID_COLUMNS = []

def filter_rows(df):
    return df


qa_topic_feat = "qa_topic"
language_version_count_feat = "language_version_count"
question_type_feat = "question_type"
cooc_features = ["cooc_num_pairs", "cooc_total_pairs", "cooc_coverage_ratio",
                 "cooc_unseen_keywords_count", "cooc_unseen_keywords_ratio",
                 "cooc_avg_pmi", "cooc_max_pmi", "cooc_min_pmi", "cooc_std_pmi"]
syntactic_features = ["target_avg_dep_depth", "target_max_tree_depth", "target_num_clauses",
                      "source_avg_dep_depth", "source_max_tree_depth", "source_num_clauses"]
linguistic_features = ["source_family", "source_genus", "target_family", "target_genus",
                       "source_script", "source_syllables", "target_script", "target_syllables"]
wiki_size_features = ["source_wiki_size", "target_wiki_size"]

FEATURE_SETS = {
    "qa_topic": [qa_topic_feat],
    "language_version_count": [language_version_count_feat],
    "question_type": [question_type_feat],
    "cooc": cooc_features,
    "syntactic": syntactic_features,
    "linguistic": linguistic_features,
    "wiki_size": wiki_size_features,
}

# Also include a union "all" for convenience
FEATURE_SETS["all"] = sorted(
    set(FEATURE_SETS["qa_topic"]
        + FEATURE_SETS["language_version_count"]
        + FEATURE_SETS["question_type"]
        + FEATURE_SETS["cooc"]
        + FEATURE_SETS["syntactic"]
        + FEATURE_SETS["linguistic"]
        + FEATURE_SETS["wiki_size"])
)

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_PATH does not exist: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        # Fall back to CSV
        df = pd.read_csv(path)
    return df

def detect_feature_types(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    """Split selected columns into categorical vs numeric by dtype/uniques heuristic."""
    numeric_cols, cat_cols = [], []
    for c in columns:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            # Heuristic: treat low-cardinality ints as categorical; strings/objects as categorical
            # If dtype is object or category OR a numeric with few unique values relative to rows -> categorical.
            if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) > max(20, int(0.02 * len(df))):
                numeric_cols.append(c)
            else:
                cat_cols.append(c)
    return numeric_cols, cat_cols

def build_preprocessor(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        n_jobs=None,
        verbose_feature_names_out=False
    )
    return pre

def get_models() -> dict[str, object]:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    }
    return models

def evaluate_models(X: pd.DataFrame, y: pd.Series, features: list[str], feature_set_name: str) -> list[dict]:
    # Keep only columns present
    present = [c for c in features if c in X.columns]
    if not present:
        print(f"[WARN] Feature set '{feature_set_name}' has no columns present in data. Skipping.")
        return []
    missing = sorted(set(features) - set(present))
    if missing:
        print(f"[INFO] Feature set '{feature_set_name}': missing columns ignored -> {missing}")

    # Detect feature types
    numeric_cols, cat_cols = detect_feature_types(X, present)

    pre = build_preprocessor(numeric_cols, cat_cols)
    models = get_models()

    # Repeated stratified CV for stability
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    # Scoring metrics (macro F1 handles multiclass; ROC AUC uses OVR for multi)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "roc_auc_ovr": "roc_auc_ovr",
    }

    rows = []
    for model_name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
        scores = cross_validate(
            pipe,
            X[present],
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            error_score="raise"
        )
        row = {
            "feature_set": feature_set_name,
            "n_features": len(present),
            "model": model_name,
            "accuracy_mean": np.mean(scores["test_accuracy"]),
            "accuracy_std": np.std(scores["test_accuracy"]),
            "balanced_acc_mean": np.mean(scores["test_balanced_accuracy"]),
            "balanced_acc_std": np.std(scores["test_balanced_accuracy"]),
            "f1_macro_mean": np.mean(scores["test_f1_macro"]),
            "f1_macro_std": np.std(scores["test_f1_macro"]),
            "roc_auc_ovr_mean": np.mean(scores["test_roc_auc_ovr"]),
            "roc_auc_ovr_std": np.std(scores["test_roc_auc_ovr"]),
        }
        rows.append(row)
    return rows

def main():
    # Load
    df = load_data(DATA_PATH)
    df = filter_rows(df)

    # Basic hygiene
    for c in ID_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=[c])

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found in data. Available columns: {list(df.columns)[:20]} ...")

    # Drop rows with missing labels
    df = df[~df[TARGET_COLUMN].isna()].copy()

    # Separate features/target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Evaluate each feature set
    all_rows = []
    for name, cols in FEATURE_SETS.items():
        rows = evaluate_models(X, y, cols, name)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No results produced. Check that your feature columns exist in the dataset.")

    results = pd.DataFrame(all_rows)
    # Order columns nicely
    col_order = ["feature_set", "n_features", "model",
                 "accuracy_mean", "accuracy_std",
                 "balanced_acc_mean", "balanced_acc_std",
                 "f1_macro_mean", "f1_macro_std",
                 "roc_auc_ovr_mean", "roc_auc_ovr_std"]
    results = results[col_order].sort_values(["feature_set", "f1_macro_mean", "accuracy_mean"], ascending=[True, False, False])

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_feature_results.csv")
    results.to_csv(out_path, index=False)

    # Pretty print top-lines
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print("\n=== Model Performance by Feature Set ===")
        print(results.to_string(index=False))
        print(f"\nSaved detailed results to: {out_path}")

if __name__ == "__main__":
    main()
