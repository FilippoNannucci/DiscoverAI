"""EDA helper functions (notebook 02)."""

import pandas as pd


def top_tfidf_terms(tfidf, texts, n=15):
    """Transform texts with a fitted TF-IDF vectoriser and return top n terms by mean weight."""
    X = tfidf.transform(texts)
    mean_weights = X.mean(axis=0).A1
    top_idx = mean_weights.argsort()[-n:][::-1]
    terms   = tfidf.get_feature_names_out()[top_idx]
    weights = mean_weights[top_idx]
    return pd.DataFrame({"term": terms, "tfidf_weight": weights})


def top_tfidf_bigrams(tfidf_bi, texts, n=15):
    """Transform texts with a fitted bigram TF-IDF vectoriser and return top n bigrams."""
    X = tfidf_bi.transform(texts)
    mean_weights = X.mean(axis=0).A1
    top_idx = mean_weights.argsort()[-n:][::-1]
    terms   = tfidf_bi.get_feature_names_out()[top_idx]
    weights = mean_weights[top_idx]
    return pd.DataFrame({"term": terms, "tfidf_weight": weights})


def price_bucket(p):
    """Map a price to a human-readable bucket label."""
    if p < 10:  return "budget"
    if p < 25:  return "low"
    if p < 50:  return "mid"
    if p < 100: return "high"
    return "premium"
