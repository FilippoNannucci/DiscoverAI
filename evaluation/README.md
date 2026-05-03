# Evaluation — Quantitative Benchmark

This folder contains the *quantitative validation* of the DiscoverAI search
pipeline. It answers explicitly the questions raised by the project rules
("which metrics?", "which baseline?", "did you tune hyperparameters?").

## Files

| File | Role |
|------|------|
| `08_Evaluation.ipynb` | Main report. 11 sections: setup → main comparison → ablations (β, modules) → per-intent breakdown → MRR top-1 analysis → summarisation/entity coverage → guardrail confusion matrix → conclusions |
| `eval_set.py` | 38 hand-crafted queries with intent tags and pseudo-relevance specs |
| `eval_metrics.py` | NDCG@K, MRR, Recall@K, Precision@K, MAP@K + relevance-judgment scorer |
| `CHANGES.md` | Empirical justification for each parameter change applied to `config.py` and `search.py` |

## Methodology

Pseudo-relevance judgments are derived from keyword matching on
`product_text_base` (title + brand + features + description), with four
levels:

```
rel = 0  → no must-keyword match  OR  excluded keyword present
rel = 1  → must matched, no bonus matched
rel = 2  → must matched, ≥1 bonus matched
rel = 3  → must matched, ≥3 bonus matched AND product_avg_rating ≥ min_rating
```

**Caveat declared explicitly in the notebook**: this scoring favours lexical
systems (BM25). Numbers should be read as relative comparisons across
systems and configurations, not as absolute ground truth. A human-annotated
benchmark would be needed for ground-truth claims; this is acknowledged as
a limitation in the notebook conclusions.

## Reproducing the results

The notebook is pre-rendered with results, so it can be read as-is. To
re-execute the underlying analysis from scratch:

```bash
# from repo root
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# the analysis scripts are kept in experiments/ (gitignored scratch space).
# To re-run them you'll need to:
#   1. produce the artefacts from notebooks 01-04 into src/io/
#   2. run experiments/step1_setup.py through step6_guardrail.py in order.
```

## Headline numbers (NDCG@5 on 37 queries, k=5)

| System | NDCG@5 | MRR | Recall@5 | Notes |
|--------|--------|-----|----------|-------|
| bm25 (baseline) | 0.524 | 0.803 | 0.144 | strong on lexical queries |
| semantic (no re-rank) | 0.522 | **0.855** | 0.133 | best top-1 result |
| search (β_q=0.12, β_p=0.05 — old default) | 0.505 | 0.765 | 0.140 | popularity hurt MRR |
| search_v3 (with min_rating=3.5 — old default) | 0.489 | 0.699 | 0.121 | filter cut recall |
| **hybrid_v4** (production default) | **0.526** | 0.760 | 0.121 | best NDCG, balanced |

After the ablation-driven changes (`BETA_POPULARITY = 0`, `min_rating=None`
default), all three team-built systems improve by ~+0.02 NDCG and ~+0.02 MRR
relative to the rows above.
