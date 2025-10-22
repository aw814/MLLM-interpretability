import json
import pandas as pd
import os


def reshape_eclektic_long(
    input_path: str,
    output_path: str,
    select_langs: list[str] = None,
) -> pd.DataFrame:
    """
    Reshape ECLeKTic dataset into long format with original and translated QA pairs.

    Args:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to save the reshaped CSV file.
        select_langs (list[str]): List of languages to include (e.g., ["en", "fr", "he", "zh"]).

    Returns:
        pd.DataFrame: Long-format DataFrame.
    """
    if select_langs is None:
        select_langs = ["en", "fr", "he", "zh"]

    # --- Load data ---
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")

    # --- Select relevant columns ---
    base_cols = ["q_id", "original_lang", "title", "url", "content", "question", "answer"]
    all_cols = (
        base_cols
        + [f"{l}_c" for l in select_langs if f"{l}_c" in df.columns]
        + [f"{l}_q" for l in select_langs if f"{l}_q" in df.columns]
        + [f"{l}_a" for l in select_langs if f"{l}_a" in df.columns]
    )
    df = df[all_cols]
    print(f"Columns selected: {len(df.columns)}")

    # --- Reshape into long format ---
    rows = []
    for _, row in df.iterrows():
        for lang in select_langs:
            c_col, q_col, a_col = f"{lang}_c", f"{lang}_q", f"{lang}_a"
            if c_col in df.columns and pd.notna(row.get(q_col)):
                rows.append(
                    {
                        "original_lang": row["original_lang"],
                        "original_content": row["content"],
                        "original_question": row["question"],
                        "original_answer": row["answer"],
                        "content": row.get(c_col, None),
                        "question": row.get(q_col, None),
                        "answer": row.get(a_col, None),
                        "language": lang,
                        "translated": 0 if row["original_lang"] == lang else 1,
                        "q_id": row["q_id"],
                        "title": row["title"],
                        "url": row["url"],
                    }
                )

    long_df = pd.DataFrame(rows)
    print(f"Reshaped to long format: {len(long_df)} rows × {len(long_df.columns)} cols")

    # --- Filter to selected languages (safety check) ---
    long_df = long_df[long_df["language"].isin(select_langs)]

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    long_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved reshaped subset to {output_path}")

    print("\nSample rows:")
    print(long_df.head(5))
    return long_df


if __name__ == "__main__":
    INPUT_PATH = "./data/raw/eclektic_main.jsonl"
    OUTPUT_PATH = "./data/processed/eclektic_long_subset.csv"
    SELECT_LANGS = ["en", "fr", "he", "zh", "de", "es", "hi", "id", "it", "ja", "ko", "pt"]

    reshape_eclektic_long(INPUT_PATH, OUTPUT_PATH, SELECT_LANGS)
