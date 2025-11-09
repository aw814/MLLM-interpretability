# 🧩 Feature Engineering — Syntactic Complexity

This module extracts **syntactic complexity features** from the ECLeKTic multilingual QA dataset.  
It provides two implementations — one based on **spaCy** and one based on **Stanza** — to compute comparable structural indicators across languages.

---

## 📘 Overview

We quantify a question’s **syntactic complexity** (as a proxy for reasoning difficulty and cross-lingual transfer robustness) with three dependency-based features:

| Feature | Description |
|----------|--------------|
| **`avg_dep_depth`** | Average dependency path length from each token to the sentence root (overall structural depth). |
| **`max_tree_depth`** | Maximum dependency tree depth (deepest syntactic nesting). |
| **`num_clauses`** | Count of subordinate / relative clauses (labels: `ccomp`, `advcl`, `relcl`, `acl`). |

The output CSV can be merged with other features (topic, frequency, BoW, question type) for downstream modeling.

---

## ⚙️ Implementation A — spaCy (`syntactic_features_spacy.py`)

**Pros:** Fast, lightweight; good for English-only baselines.  
**Cons:** Not multilingual by default.


## ⚙️ Implementation B — Stanza (syntactic_features_stanza.py)

**Pros:** Fully multilingual — supports 60+ languages via Universal Dependencies.
**Cons:** Cross-lingual structural comparability ensured by UD framework. Slower; requires downloading per-language models.

## References

Matthew Honnibal, Ines Montani, et al. “spaCy 3: Industrial-strength Natural Language Processing in Python.” (2020).
Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D. Manning.
“Stanza: A Python Natural Language Processing Toolkit for Many Human Languages.” ACL 2020.
