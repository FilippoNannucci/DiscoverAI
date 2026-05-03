"""
search_extended.py — Production-grade extensions on top of `search.py`.

Adds two orthogonal capabilities:

  1. search_v4()    — Hybrid search: BM25 + semantic via Reciprocal Rank Fusion.
  2. rerank_mmr()   — Maximal Marginal Relevance for diversity-aware re-ranking.

Both are designed to compose with the existing pipeline::

    candidates = search_v3(..., k=50)            # existing retrieval
    diverse    = rerank_mmr(candidates, emb, k=5) # MMR over the candidates

    # or, hybrid in one call:
    results = search_v4(query, model, faiss_index, bm25, index_df, k=5)

`search_v4` is the recommended production default: on the internal benchmark
(see `08_Evaluation.ipynb`) it achieves the highest NDCG@5 across
all retrieval systems by combining the lexical precision of BM25 with the
semantic recall of MPNet.

Extra dependencies::

    pip install rank-bm25
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from .config import BETA_QUALITY, BETA_POPULARITY, N_CANDIDATES
from .search import (
    parse_query_v2,
    expand_query,
    extract_dosages,
    dosage_filter,
)


# ── BM25 index ────────────────────────────────────────────────────────────────

def build_bm25_index(index_df: pd.DataFrame) -> "BM25Okapi":
    """
    Build a BM25 index over product texts.

    Tokenises: title + brand. BM25 is bag-of-words: each document is a list of
    lower-cased tokens. The corpus is built once and held in memory — it is
    small (~7.5k products × ~20 tokens average ≈ 150k tokens overall).

    Args:
        index_df: must contain at least the columns 'product_title' and 'brand'.

    Returns:
        A BM25Okapi instance ready for `.get_scores(tokens)` queries.
    """
    if not BM25_AVAILABLE:
        raise ImportError("rank_bm25 is not installed. Run: pip install rank-bm25")

    def tokenize(row: pd.Series) -> list[str]:
        title = str(row.get("product_title", "") or "")
        brand = str(row.get("brand", "") or "")
        text  = f"{title} {brand}".lower()
        text  = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    corpus = index_df.apply(tokenize, axis=1).tolist()
    return BM25Okapi(corpus)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = 60) -> float:
    """RRF score for a position in the ranking. k=60 is the canonical value (Cormack 2009)."""
    return 1.0 / (k + rank + 1)


def _fuse_rankings(
    semantic_df: pd.DataFrame,
    bm25_df: pd.DataFrame,
    n_final: int,
    rrf_k: int = 60,
) -> pd.DataFrame:
    """
    Combine two ranked DataFrames (semantic and BM25) via Reciprocal Rank Fusion.

    Both DataFrames must have a 'parent_asin' column. The semantic frame must
    additionally carry 'similarity', 'quality_score' and 'popularity_score' —
    these are needed for the final re-ranking step.

    RRF is robust: it does not require score normalisation (it operates on
    pure rank positions) and tends to perform well out-of-the-box.

    Returns:
        DataFrame with an added 'rrf_score' column, sorted by rrf_score desc.
    """
    sem_ranks  = {asin: rank for rank, asin in enumerate(semantic_df["parent_asin"])}
    bm25_ranks = {asin: rank for rank, asin in enumerate(bm25_df["parent_asin"])}

    all_asins = set(sem_ranks) | set(bm25_ranks)

    # RRF: sum of contributions from each ranking
    rrf_scores: dict[str, float] = {}
    for asin in all_asins:
        score = 0.0
        if asin in sem_ranks:
            score += _rrf_score(sem_ranks[asin], k=rrf_k)
        if asin in bm25_ranks:
            score += _rrf_score(bm25_ranks[asin], k=rrf_k)
        rrf_scores[asin] = score

    # Start from semantic_df (rich metadata) and append BM25-only ASINs.
    result = semantic_df.copy()
    result["rrf_score"] = result["parent_asin"].map(rrf_scores).fillna(0.0)

    bm25_only = set(bm25_df["parent_asin"]) - set(semantic_df["parent_asin"])
    if bm25_only:
        extra = bm25_df[bm25_df["parent_asin"].isin(bm25_only)].copy()
        extra["rrf_score"] = extra["parent_asin"].map(rrf_scores).fillna(0.0)
        result = pd.concat([result, extra], ignore_index=True)

    return result.sort_values("rrf_score", ascending=False).head(n_final).reset_index(drop=True)


# ── search_v4: Hybrid BM25 + Semantic ────────────────────────────────────────

def search_v4(
    query: str,
    model,
    faiss_index,
    bm25_index: "BM25Okapi",
    index_df: pd.DataFrame,
    k: int = 5,
    price_buckets: list | None = None,
    min_rating: float = None,
    n_candidates: int = None,
    rrf_k: int = 60,
    beta_quality: float = None,
    beta_popularity: float = BETA_POPULARITY,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Hybrid search: BM25 keyword matching + FAISS semantic retrieval, fused
    with Reciprocal Rank Fusion (RRF).

    Why this beats `search_v3` alone on the internal benchmark:
      - search_v3 wins on conceptual queries ("moisturizer for dry skin")
      - search_v4 wins on exact / brand queries ("CeraVe Hydrating Cleanser")
      - RRF combines the two without requiring score normalisation

    Pipeline:
        1. Synonym expansion + query parsing (same as search_v3)
        2. BM25 retrieval — top-N by lexical score
        3. FAISS retrieval — top-N by semantic similarity
        4. RRF fusion → unified ranking
        5. Hard filters: price_bucket, min_rating, negation, dosages
        6. Final re-ranking using rrf_score + similarity + quality (+ popularity if enabled)

    Args:
        query:           raw user text.
        model:           already-loaded SentenceTransformer.
        faiss_index:     already-built FAISS IndexFlatIP.
        bm25_index:      output of `build_bm25_index()`.
        index_df:        DataFrame from embedding_index_enriched.csv
                         (with quality_score / popularity_score pre-computed).
        k:               final result count.
        price_buckets:   optional explicit price-bucket filter.
        min_rating:      optional rating floor (off by default).
        n_candidates:    candidates retrieved from each system before fusion
                         (default: N_CANDIDATES * 2 = 100).
        rrf_k:           RRF parameter (60 is the literature default).
        beta_quality:    quality-score weight (None = adaptive on similarity).
        beta_popularity: popularity-score weight (0 by default in config).
        verbose:         debug printing.

    Returns:
        DataFrame with index_df columns + similarity + rrf_score + score.
    """
    if not BM25_AVAILABLE:
        raise ImportError("rank_bm25 is not installed. Run: pip install rank-bm25")

    n_cand = n_candidates or (N_CANDIDATES * 2)

    # ── Step 1: expand + parse ────────────────────────────────────────────
    expanded = expand_query(query)
    if expanded != query.lower().strip() and verbose:
        print(f"  Expansion: '{query}' → '{expanded}'")

    parsed   = parse_query_v2(expanded)
    buckets  = price_buckets if price_buckets is not None else parsed["price_buckets"]
    boost    = parsed["quality_boost"]
    exclude  = parsed["exclude_words"]
    dosages  = extract_dosages(query)
    clean_q  = parsed["clean"]

    if verbose:
        print(f"  clean query     : {clean_q}")
        print(f"  price buckets   : {buckets}")
        print(f"  exclude words   : {exclude}")
        print(f"  detected dosages: {dosages}")

    # ── Step 2: BM25 retrieval ────────────────────────────────────────────
    bm25_tokens  = re.sub(r"[^a-z0-9\s]", " ", clean_q.lower()).split()
    bm25_scores  = bm25_index.get_scores(bm25_tokens)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:n_cand]

    bm25_df = index_df.iloc[bm25_top_idx].copy().reset_index(drop=True)
    bm25_df["bm25_score"] = bm25_scores[bm25_top_idx]
    # Keep only candidates with at least one term match
    bm25_df = bm25_df[bm25_df["bm25_score"] > 0].reset_index(drop=True)

    if verbose:
        print(f"  BM25 candidates : {len(bm25_df)}")

    # ── Step 3: FAISS semantic retrieval ─────────────────────────────────
    q_vec = model.encode(
        [clean_q], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    n_faiss = n_cand * 2 if (buckets or exclude or dosages) else n_cand
    D, I    = faiss_index.search(q_vec, n_faiss)

    sem_df  = index_df.iloc[I[0]].copy().reset_index(drop=True)
    sem_df["similarity"] = D[0]

    if verbose:
        print(f"  FAISS candidates: {len(sem_df)}")

    # ── Step 4: RRF fusion ────────────────────────────────────────────────
    fused = _fuse_rankings(sem_df, bm25_df, n_final=n_cand * 3, rrf_k=rrf_k)

    if verbose:
        print(f"  post-fusion     : {len(fused)} unique candidates")

    # ── Step 5: hard filters (post-fusion to preserve recall) ─────────────
    if buckets:
        fused = fused[fused["price_bucket"].isin(buckets)]
    if min_rating is not None and "product_avg_rating" in fused.columns:
        fused = fused[fused["product_avg_rating"] >= min_rating]
    for word in exclude:
        fused = fused[~fused["product_title"].str.lower().str.contains(word, na=False, regex=False)]
    if dosages:
        fused = dosage_filter(fused, dosages)

    if len(fused) == 0:
        if verbose:
            print("  no results after filtering")
        return pd.DataFrame()

    # ── Step 6: final re-ranking ──────────────────────────────────────────
    if beta_quality is None:
        avg_sim      = fused["similarity"].mean() if "similarity" in fused.columns else 0.6
        beta_quality = 0.20 if avg_sim < 0.65 else BETA_QUALITY

    eff_beta = beta_quality * 1.5 if boost else beta_quality
    sim_col  = fused["similarity"].fillna(0) if "similarity" in fused.columns else 0

    fused["score"] = (
        fused["rrf_score"] * 10           # rescale RRF to be comparable to similarity
        + sim_col
        + eff_beta        * fused["quality_score"].fillna(0)
        + beta_popularity * fused["popularity_score"].fillna(0)
    )

    return fused.sort_values("score", ascending=False).head(k).reset_index(drop=True)


# ── MMR: Maximal Marginal Relevance ───────────────────────────────────────────

def rerank_mmr(
    candidates: pd.DataFrame,
    combined_emb: np.ndarray,
    k: int = 5,
    lambda_mmr: float = 0.6,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Re-rank a candidate set using Maximal Marginal Relevance to balance
    relevance and diversity.

    MMR is most useful when the top-k pure-relevance results are nearly
    identical (e.g. five almost-identical SKUs from the same brand).

    Carbonell & Goldstein (1998) formulation::

        MMR = argmax [ λ · Sim(q, d)  − (1 − λ) · max_{d' ∈ S} Sim(d, d') ]

    where:
        - Sim(q, d) is the query relevance (taken from 'score' or 'similarity')
        - Sim(d, d') is the cosine similarity between two candidates
        - S is the already-selected set
        - λ is the relevance/diversity trade-off (0.5 = balanced, 1.0 = pure relevance)

    Args:
        candidates:   DataFrame with 'parent_asin', 'emb_row' and 'score'
                      (output of any search_v* function).
        combined_emb: (n_products × 768) array from combined_embeddings.npy.
        k:            number of final results to select.
        lambda_mmr:   relevance/diversity trade-off (default 0.6 — slightly
                      relevance-biased).
        verbose:      print intra-set similarities at every step.

    Returns:
        DataFrame of the k selected products with an added 'mmr_score' column.
    """
    if len(candidates) == 0:
        return candidates

    if "emb_row" not in candidates.columns:
        raise ValueError(
            "Candidates DataFrame must include the 'emb_row' column. "
            "Make sure you are passing index_df from embedding_index_enriched.csv."
        )

    k = min(k, len(candidates))

    rows = candidates["emb_row"].astype(int).values
    embs = combined_emb[rows].astype(np.float32)         # (n_cand, dim)

    # L2-normalise (in case the inputs aren't already normalised)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embs  = embs / norms

    # Relevance score (column 'score', falling back to 'similarity')
    if "score" in candidates.columns:
        relevance = candidates["score"].fillna(0).values
    elif "similarity" in candidates.columns:
        relevance = candidates["similarity"].fillna(0).values
    else:
        relevance = np.ones(len(candidates))

    # Min-max normalise relevance to [0, 1]
    rel_min, rel_max = relevance.min(), relevance.max()
    if rel_max > rel_min:
        relevance = (relevance - rel_min) / (rel_max - rel_min)

    # Greedy MMR selection
    selected_idx  = []
    remaining_idx = list(range(len(candidates)))

    for step in range(k):
        if not remaining_idx:
            break

        best_idx = None
        best_mmr = -np.inf

        for i in remaining_idx:
            rel_score = relevance[i]
            if len(selected_idx) == 0:
                # First pick: highest pure relevance
                mmr_score = rel_score
            else:
                sel_embs  = embs[selected_idx]
                sims      = sel_embs @ embs[i]            # dot product, already L2-normalised
                max_sim   = float(sims.max())
                mmr_score = lambda_mmr * rel_score - (1 - lambda_mmr) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if verbose and len(selected_idx) > 0:
            title = str(candidates.iloc[best_idx]["product_title"])[:60]
            print(f"  step {step+1}: '{title}' — MMR={best_mmr:.4f}")

        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    result = candidates.iloc[selected_idx].copy().reset_index(drop=True)
    result["mmr_score"] = [relevance[i] for i in selected_idx]
    return result
