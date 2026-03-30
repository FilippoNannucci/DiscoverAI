"""Embedding, FAISS index building, and search utilities (notebook 03)."""

import numpy as np
import pandas as pd

from .config import ALPHA_MAX, ALPHA_MIN, W_HELPFUL, W_RATING


def compute_review_weights(reviews):
    """
    Compute per-review weights combining helpful_vote and rating distance,
    with bias correction that boosts rare reviews within each product.

    Weight formula:
        weight_i = (W_HELPFUL × log1p(helpful_vote_i) / max(log1p)
                  + W_RATING  × |rating_i - 3| / 2)
                  × bias_correction_i
    """
    hv       = reviews["helpful_vote"].fillna(0).values.astype(float)
    hv_log   = np.log1p(hv)
    hv_norm  = hv_log / hv_log.max() if hv_log.max() > 0 else hv_log
    rating_dist  = (np.abs(reviews["rating"].values - 3) / 2).astype(float)
    weights_base = np.clip(W_HELPFUL * hv_norm + W_RATING * rating_dist, 1e-6, None)

    global_dist  = reviews["rating"].value_counts(normalize=True).to_dict()
    prod_dist    = reviews.groupby("parent_asin")["rating"].value_counts(normalize=True)
    bias_weights = np.ones(len(reviews))
    for i, (_, row) in enumerate(reviews.iterrows()):
        local_pct  = prod_dist.get((row["parent_asin"], row["rating"]), 0.01)
        global_pct = global_dist.get(row["rating"], 0.01)
        bias_weights[i] = np.clip(global_pct / max(local_pct, 0.01), 0.5, 3.0)

    return np.clip(weights_base * bias_weights, 1e-6, None)


def compute_review_embedding(review_emb_raw, reviews, weights, asin_to_idx, dim):
    """
    Compute a weighted average review embedding per product.
    Returns an (n_products, dim) normalised float32 array.
    """
    n = len(asin_to_idx)
    review_emb = np.zeros((n, dim), dtype=np.float32)
    for asin, grp in reviews.groupby("parent_asin"):
        idx = asin_to_idx.get(asin)
        if idx is None:
            continue
        row_idxs   = grp.index.tolist()
        embs       = review_emb_raw[row_idxs]
        w          = weights[row_idxs]
        review_emb[idx] = (embs * w[:, None]).sum(axis=0) / w.sum()
    norms = np.linalg.norm(review_emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (review_emb / norms).astype(np.float32)


def compute_dynamic_alpha(reviews, products):
    """
    Compute per-product alpha based on review signal quality.
    High alpha → product embedding dominates (weak review signal).
    Low alpha  → review embedding matters more (strong review signal).
    Returns a dict {parent_asin: alpha}.
    """
    rev_stats = reviews.groupby("parent_asin").agg(
        n_rev       =("rating", "count"),
        max_helpful =("helpful_vote", "max"),
    ).reset_index()
    rev_stats["max_helpful"] = rev_stats["max_helpful"].fillna(0)

    n_rev_norm   = (rev_stats["n_rev"].clip(upper=16) - 3) / 13
    hv_prod_norm = rev_stats["max_helpful"].clip(upper=20) / 20
    quality      = 0.6 * n_rev_norm + 0.4 * hv_prod_norm
    alphas       = ALPHA_MAX - quality * (ALPHA_MAX - ALPHA_MIN)
    return dict(zip(rev_stats["parent_asin"], alphas))


def blend_embeddings(product_emb, review_emb, asin_alpha, products):
    """
    Blend product and review embeddings using per-product alpha:
        combined = alpha × product_emb + (1 - alpha) × review_emb
    Returns a normalised (n_products, dim) float32 array.
    """
    alpha_vec = np.array(
        [asin_alpha.get(asin, 0.5) for asin in products["parent_asin"]],
        dtype=np.float32,
    )[:, None]
    combined = alpha_vec * product_emb + (1 - alpha_vec) * review_emb
    norms    = np.linalg.norm(combined, axis=1, keepdims=True)
    norms    = np.where(norms == 0, 1, norms)
    return (combined / norms).astype(np.float32)


def search(query, model, index, index_df, k=5, price_bucket=None):
    """
    Encode query, search FAISS index, optionally filter by price_bucket.
    Returns top-k results with product metadata and cosine score.
    """
    q_vec  = model.encode([query], normalize_embeddings=True,
                           convert_to_numpy=True).astype(np.float32)
    n_cand = k * 10 if price_bucket else k
    D, I   = index.search(q_vec, n_cand)
    res    = index_df.iloc[I[0]].copy()
    res["score"] = D[0]
    if price_bucket:
        res = res[res["price_bucket"] == price_bucket]
    return res.head(k)[["product_title", "brand", "price", "price_bucket",
                          "product_avg_rating", "alpha", "score"]]
