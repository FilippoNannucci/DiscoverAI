# DiscoverAI — Semantic Product Search for Health & Personal Care

Review-aware semantic search system built on the Amazon Reviews 2023 dataset (Health & Personal Care). Natural-language queries are matched to products via sentence embedding similarity, quality-weighted re-ranking, and structured review insights extracted through summarization and entity recognition.

**Team:** Mean-Squared-Terrors · Deloitte × LUISS 2026

---

## Notebooks

### 01 · Data Ingestion & Cleaning

Raw Amazon data is filtered down to a usable product catalogue. Products with fewer than 10 reviews are excluded — below this threshold the review signal is too weak to generate a meaningful embedding. Reviews below 25 tokens are also dropped (they carry no semantic content: "Love it!!", "Fast shipping").

The key design decision here is **balanced rating sampling**: instead of keeping all reviews, at most `{5:5, 4:3, 3:2, 2:3, 1:3}` reviews are selected per rating bucket, prioritizing the most helpful votes. Raw five-star reviews dominate (~62 % of the corpus); after balancing this drops to ~46 %, giving the embedding model a more honest picture of each product.

Product text is constructed by concatenating title, brand, features, description, and selected `details` fields (item form, skin type, active ingredients, etc.). Fields are capped by token count so the final text fits within MPNet's 512-token limit.

**Output:** 7,647 products · 318k cleaned reviews · 57k balanced reviews.

---

### 02 · Exploratory Data Analysis

Characterizes the dataset before modelling: rating distributions (raw vs. balanced), review length, price buckets, top categories, and helpful-vote patterns. Results from the EDA directly informed the cleaning choices in NB01 (minimum review thresholds, which `details` keys carry semantic value, price bucket boundaries).

---

### 03 · Embedding

Builds the vector index that powers all downstream search.

**Two-tower architecture.** Each product gets two embeddings computed independently with `all-mpnet-base-v2` (768 dimensions, trained on 1B+ sentence pairs):

1. **Product embedding** — from the cleaned product text (title + features + metadata).
2. **Review embedding** — a weighted average of the balanced reviews, where each review's weight combines its helpful-vote score (70 %) and how far its rating deviates from neutral (30 %). This surfaces the reviews that other customers found most useful.

**Dynamic alpha blending.** The two embeddings are merged as:

```
combined = alpha * product_emb + (1 - alpha) * review_emb
```

Alpha is not fixed: it scales with the product's review signal strength. Products with few or low-quality reviews get a higher alpha (product text dominates); products with rich review evidence get a lower alpha (review voice matters more). The range is `[0.35, 0.70]`.

**Index.** All combined embeddings are normalized and loaded into a FAISS `IndexFlatIP` (exact inner-product search). Approximate indices (IVF, HNSW) were considered but rejected: at ~7.6k products, exact search adds negligible latency while guaranteeing no missed neighbours.

---

### 04 · Semantic Search

Implements and evaluates the retrieval and re-ranking pipeline.

**Base search.** A query is encoded with the same MPNet model, and FAISS retrieves the top-50 candidates by cosine similarity. Candidates are then re-ranked with:

```
score = similarity + β_quality · quality_score + β_pop · popularity_score
```

`quality_score` combines average rating, fraction of positive reviews, fraction of negative reviews, and log-helpful-votes. `popularity_score` is the log-normalized review count. Both are min-max normalized to `[0, 1]`. The betas (`0.12` and `0.05`) are small enough that a low-quality product cannot overtake a genuinely more relevant one purely on popularity.

**Three search variants developed iteratively:**

- `search` — baseline similarity + fixed re-ranking.
- `search_v2` — adds negation filtering: terms after "without", "no", "free of" are detected and used to post-filter results that mention those terms in product text.
- `search_v3` (production) — adds synonym expansion (e.g. "SPF" → "sunscreen", "omega 3" → "fish oil"), price-intent parsing ("cheap", "affordable", "budget" → price bucket filter), dosage normalization, and **adaptive beta**: when the mean similarity across the 50 retrieved candidates is below 0.65 (query is generic or ambiguous), the quality weight increases from 0.12 to 0.20, falling back on crowd wisdom rather than noisy similarity scores.

**Recommendation.** A separate function retrieves the five most similar products to a given ASIN, operating on the combined embedding matrix directly without going through the query encoder.

---

### 05 · Summarization & Entity Recognition

Extracts structured, human-readable intelligence from the raw review text.

**Summarization.** For each product, reviews are split into positive (≥4 stars) and negative (≤2 stars) buckets. Informative sentences are extracted (≥12 words, not starting with "I", <300 chars) and deduplicated. The resulting text is fed to **BART-Large-CNN** (BART fine-tuned on CNN/DailyMail summarization), which produces a concise summary plus explicit pros and cons fields. Products with fewer than 5 reviews fall back to extractive summarization (top sentences only) to avoid hallucination on thin evidence.

**Entity extraction.** Regex patterns identify domain-specific entities directly from product text and review snippets: active ingredients (retinol, niacinamide, hyaluronic acid, SPF, omega-3, …), certifications (organic, vegan, cruelty-free, paraben-free, …), and use-case targets (skin type, hair type, age group, body area). spaCy NER (`en_core_web_sm`) is run on top to extract brand mentions.

The final `product_profiles.csv` merges the embedding index, summaries, and entities into a single table consumed by the demo.

---

### 06 · Demo

Gradio web interface integrating the full pipeline. Three tabs:

- **Search** — free-text query with price bucket and minimum rating filters; results show product image, rating, price, BART summary, pros and cons (when NB05 has been run).
- **Similar products** — content-based recommendation by product name.
- **System info** — index statistics, model details, feature coverage.

---

## Source modules (`src/mean_squared_terrors/`)

| Module | Role |
|--------|------|
| `config.py` | All tunable parameters in one place: model name, review caps, alpha range, beta weights |
| `cleaning.py` | Text normalization, balanced rating sampling, product text construction |
| `eda.py` | Plotting helpers for NB02 |
| `embedding.py` | MPNet encoding, weighted review aggregation, dynamic alpha blending, FAISS index |
| `search.py` | `search`, `search_v2`, `search_v3`, negation/synonym/price parsing, recommendation |
| `summarization.py` | BART-Large-CNN summarization, spaCy entity extraction, batch processing, profile merge |

---

## Requirements

Dependencies are listed in `requirements.txt`. All parameters are centralized in `config.py` — no hardcoded values appear inside functions or notebooks.
