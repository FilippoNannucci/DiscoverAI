"""Search, re-ranking, and recommendation utilities (notebook 04)."""

import re
import numpy as np
import pandas as pd

from .config import BETA_QUALITY, BETA_POPULARITY, N_CANDIDATES


# ── Price intent ──────────────────────────────────────────────────────────────
PRICE_INTENT = {
    "cheap":        ["budget", "low"],
    "affordable":   ["budget", "low"],
    "budget":       ["budget"],
    "inexpensive":  ["budget", "low"],
    "low cost":     ["budget", "low"],
    "low-cost":     ["budget", "low"],
    "value":        ["budget", "low", "mid"],
    "premium":      ["high", "premium"],
    "luxury":       ["premium"],
    "expensive":    ["high", "premium"],
    "professional": ["high", "premium"],
    "high end":     ["high", "premium"],
    "high-end":     ["high", "premium"],
    "best":         None,   # quality signal, not a price bucket
}

QUALITY_BOOST_WORDS = {"best", "top", "highest rated", "most popular"}

# ── Negation patterns ─────────────────────────────────────────────────────────
NEGATION_PATTERNS = [
    r"without\s+(\w+(?:\s+\w+)?)",
    r"\bno\s+(\w+)",
    r"free\s+from\s+(\w+(?:\s+\w+)?)",
    r"(\w+)[\-\s]free\b",
]

# ── Synonym expansion ─────────────────────────────────────────────────────────
SYNONYM_MAP = {
    "help me sleep":        "sleep aid melatonin supplement",
    "can't sleep":          "sleep aid insomnia supplement",
    "fall asleep":          "sleep supplement melatonin",
    "dry cracked skin":     "moisturizer healing ointment dry skin",
    "fix dry skin":         "moisturizer cream dry skin",
    "protect skin sun":     "sunscreen SPF UV protection",
    "sun protection":       "sunscreen SPF broad spectrum",
    "at the beach":         "sunscreen waterproof SPF",
    "stop hair loss":       "hair loss treatment DHT blocker",
    "grow hair":            "hair growth supplement biotin",
    "longer stronger hair": "hair growth vitamins biotin collagen",
    "energy boost morning": "energy supplement caffeine B12",
    "feel tired":           "energy supplement fatigue",
    "joint pain":           "joint support glucosamine supplement pain relief",
    "knee pain":            "knee brace support joint pain relief",
    "lose weight":          "weight loss fat burner supplement",
    "burn fat":             "thermogenic fat burner supplement",
    "clean teeth":          "teeth cleaning dental hygiene",
    "fresh breath":         "fresh breath mouthwash oral hygiene",
    "digestive problems":   "probiotic digestive supplement",
    "bloating":             "probiotic digestive enzyme supplement",
}

# ── Dosage regex ──────────────────────────────────────────────────────────────
DOSAGE_PATTERN = re.compile(
    r"\b(\d+\s*(?:mg|mcg|iu|g|ml|oz|lb|%|spf\s*\d+|x\d+))\b",
    re.IGNORECASE,
)


# ── Quality scores ────────────────────────────────────────────────────────────

def compute_quality_scores(index_df: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich index_df with quality_score and popularity_score, both normalised to [0, 1].

    quality_score    = 0.40 × rating_norm + 0.35 × sentiment + 0.25 × helpful_credibility
    popularity_score = log-normalised total review count (clipped at p95)

    Call once per session — never at query time.
    Returns a new DataFrame (index_df is not modified in place).
    """
    rev_agg = reviews.groupby("parent_asin").agg(
        n_reviews      = ("rating", "count"),
        avg_rating_rev = ("rating", "mean"),
        pct_positive   = ("rating", lambda x: (x >= 4).mean()),
        pct_negative   = ("rating", lambda x: (x <= 2).mean()),
        total_helpful  = ("helpful_vote", "sum"),
        max_helpful    = ("helpful_vote", "max"),
    ).reset_index()

    df = index_df.merge(rev_agg, on="parent_asin", how="left")

    rating_norm  = (df["product_avg_rating"].fillna(3.0) - 1) / 4
    sentiment    = df["pct_positive"].fillna(0.5) - 0.5 * df["pct_negative"].fillna(0.5)
    hv_log       = np.log1p(df["total_helpful"].fillna(0))
    hv_p95       = hv_log.quantile(0.95)
    helpful_cred = (hv_log / hv_p95).clip(upper=1.0)

    df["quality_score"] = (
        0.40 * rating_norm +
        0.35 * sentiment   +
        0.25 * helpful_cred
    )

    rc_log = np.log1p(df["product_rating_count"].fillna(0))
    rc_p95 = rc_log.quantile(0.95)
    df["popularity_score"] = (rc_log / rc_p95).clip(upper=1.0)

    for col in ["quality_score", "popularity_score"]:
        mn, mx = df[col].min(), df[col].max()
        df[col] = (df[col] - mn) / (mx - mn + 1e-9)

    return df


# ── Query parsing ─────────────────────────────────────────────────────────────

def parse_query(query: str) -> dict:
    """
    Extract structured intent from a free-text query.

    Strips price/quality words so only the semantic core is encoded by MPNet.
    Example: 'affordable moisturizer for sensitive skin'
             → clean='moisturizer for sensitive skin', price_buckets=['budget','low']

    Returns:
        original      : original query string
        clean         : query with price/quality words stripped (pass this to the encoder)
        price_buckets : list of acceptable price buckets, or None (no filter)
        quality_boost : True if query contains boost words ("best", "top", …)
    """
    q_lower = query.lower().strip()
    price_buckets = None
    quality_boost = False
    clean = q_lower

    for word, buckets in PRICE_INTENT.items():
        if word in q_lower:
            if buckets is not None:
                price_buckets = buckets
            else:
                quality_boost = True
            clean = clean.replace(word, "").strip()

    for word in QUALITY_BOOST_WORDS:
        if word in q_lower:
            quality_boost = True
            clean = clean.replace(word, "").strip()

    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) < 3:
        clean = query

    return {
        "original":      query,
        "clean":         clean,
        "price_buckets": price_buckets,
        "quality_boost": quality_boost,
    }


def parse_query_v2(query: str) -> dict:
    """
    Extended parser — adds negation extraction on top of parse_query.

    Handles: 'without X', 'no X', 'X-free', 'free from X'.
    Extracted terms are stored in exclude_words and used to post-filter results.
    """
    q_lower = query.lower().strip()
    price_buckets, quality_boost = None, False
    clean = q_lower
    exclude_words = []

    for pattern in NEGATION_PATTERNS:
        matches = re.findall(pattern, q_lower)
        for m in matches:
            word = m.strip() if isinstance(m, str) else (m[0] if m else "")
            if word and len(word) > 2:
                exclude_words.append(word.lower())

    for word, buckets in PRICE_INTENT.items():
        if word in q_lower:
            if buckets is not None:
                price_buckets = buckets
            else:
                quality_boost = True
            clean = clean.replace(word, "").strip()

    for word in QUALITY_BOOST_WORDS:
        if word in q_lower:
            quality_boost = True
            clean = clean.replace(word, "").strip()

    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) < 3:
        clean = query

    return {
        "original":      query,
        "clean":         clean,
        "price_buckets": price_buckets,
        "quality_boost": quality_boost,
        "exclude_words": exclude_words,
    }


# ── Synonym expansion ─────────────────────────────────────────────────────────

def expand_query(query: str) -> str:
    """
    Replace implicit consumer phrases with explicit product vocabulary via SYNONYM_MAP.
    Example: 'help me sleep' → 'sleep aid melatonin supplement'
    Returns the original query unchanged if no pattern matches.
    """
    q_lower = query.lower().strip()
    for pattern, expansion in SYNONYM_MAP.items():
        if pattern in q_lower:
            return q_lower.replace(pattern, expansion)
    return query


# ── Dosage filter ─────────────────────────────────────────────────────────────

def extract_dosages(text: str) -> list:
    """Extract dosage specs (mg, mcg, SPF, IU, %) from a query string."""
    return [m.group(0).lower().replace(" ", "") for m in DOSAGE_PATTERN.finditer(text)]


def dosage_filter(results_df: pd.DataFrame, dosages: list) -> pd.DataFrame:
    """
    Keep only rows whose product_title contains at least one queried dosage.
    Falls back to the unfiltered results if fewer than 2 products match.
    """
    if not dosages:
        return results_df
    mask = results_df["product_title"].str.lower().apply(
        lambda title: any(d in title.replace(" ", "") for d in dosages)
    )
    filtered = results_df[mask]
    return filtered if len(filtered) >= 2 else results_df


# ── Search functions ──────────────────────────────────────────────────────────

def search(
    query: str,
    model,
    index,
    index_df: pd.DataFrame,
    k: int = 5,
    price_buckets: list = None,
    min_rating: float = None,
    beta_quality: float = BETA_QUALITY,
    beta_popularity: float = BETA_POPULARITY,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Base semantic search with quality re-ranking.

    Pipeline: parse → encode clean query → FAISS top-K → hard filters → re-ranking.

    Score = similarity + beta_quality × quality_score + beta_popularity × popularity_score
    beta_quality is multiplied by 1.5 when the query contains boost words ("best", "top"…).
    """
    parsed  = parse_query(query)
    buckets = price_buckets if price_buckets is not None else parsed["price_buckets"]
    boost   = parsed["quality_boost"]

    if verbose:
        print(f"Query originale : {parsed['original']}")
        print(f"Query pulita    : {parsed['clean']}")
        print(f"Price buckets   : {buckets}")
        print(f"Quality boost   : {boost}")

    q_vec  = model.encode(
        [parsed["clean"]], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    n_cand = N_CANDIDATES * 3 if buckets else N_CANDIDATES
    D, I   = index.search(q_vec, n_cand)

    res = index_df.iloc[I[0]].copy()
    res["similarity"] = D[0]

    if buckets:
        res = res[res["price_bucket"].isin(buckets)]
    if min_rating is not None:
        res = res[res["product_avg_rating"] >= min_rating]

    eff_beta = beta_quality * 1.5 if boost else beta_quality
    res["score"] = (
        res["similarity"] +
        eff_beta        * res["quality_score"].fillna(0) +
        beta_popularity * res["popularity_score"].fillna(0)
    )
    return res.sort_values("score", ascending=False).head(k).reset_index(drop=True)


def search_v2(
    query: str,
    model,
    index,
    index_df: pd.DataFrame,
    k: int = 5,
    price_buckets: list = None,
    min_rating: float = None,
    beta_quality: float = BETA_QUALITY,
    beta_popularity: float = BETA_POPULARITY,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Search with negation filter.

    Extends search() by removing products whose title contains negated terms.
    Handles: 'without X', 'no X', 'X-free', 'free from X'.
    """
    parsed  = parse_query_v2(query)
    buckets = price_buckets if price_buckets is not None else parsed["price_buckets"]
    boost   = parsed["quality_boost"]
    exclude = parsed["exclude_words"]

    if verbose:
        print(f"Query pulita  : {parsed['clean']}")
        print(f"Price buckets : {buckets}")
        print(f"Quality boost : {boost}")
        print(f"Escludi       : {exclude}")

    q_vec  = model.encode(
        [parsed["clean"]], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    n_cand = 150 if (buckets or exclude) else N_CANDIDATES
    D, I   = index.search(q_vec, n_cand)

    res = index_df.iloc[I[0]].copy()
    res["similarity"] = D[0]

    if buckets:
        res = res[res["price_bucket"].isin(buckets)]
    if min_rating is not None:
        res = res[res["product_avg_rating"] >= min_rating]
    for word in exclude:
        res = res[~res["product_title"].str.lower().str.contains(word, na=False, regex=False)]

    eff_beta = beta_quality * 1.5 if boost else beta_quality
    res["score"] = (
        res["similarity"] +
        eff_beta        * res["quality_score"].fillna(0) +
        beta_popularity * res["popularity_score"].fillna(0)
    )
    return res.sort_values("score", ascending=False).head(k).reset_index(drop=True)


def search_v3(
    query: str,
    model,
    index,
    index_df: pd.DataFrame,
    k: int = 5,
    price_buckets: list = None,
    min_rating: float = None,
    beta_quality: float = None,
    beta_popularity: float = BETA_POPULARITY,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Search v3 — all improvements combined:
    - Synonym expansion for implicit queries
    - Dosage filter (1000mg, SPF 50, IU, …)
    - min_rating optional (off by default — caller can pass e.g. 3.5 to enable
      a hard quality cut). Trade-off: enabling it raises perceived precision
      but lowers recall by ~1.6% NDCG@5; see evaluation/08_Evaluation.ipynb.
    - Adaptive beta_quality: 0.20 when mean similarity < 0.65, 0.12 otherwise
    - Negation filter inherited from search_v2
    """
    expanded = expand_query(query)
    if expanded != query.lower().strip() and verbose:
        print(f"  Expansion: '{query}' → '{expanded}'")

    parsed  = parse_query_v2(expanded)
    buckets = price_buckets if price_buckets is not None else parsed["price_buckets"]
    boost   = parsed["quality_boost"]
    exclude = parsed["exclude_words"]
    dosages = extract_dosages(query)

    if dosages and verbose:
        print(f"  Dosaggi: {dosages}")

    q_vec  = model.encode(
        [parsed["clean"]], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    n_cand = 150 if (buckets or exclude or dosages) else 80
    D, I   = index.search(q_vec, n_cand)

    res = index_df.iloc[I[0]].copy()
    res["similarity"] = D[0]

    if buckets:
        res = res[res["price_bucket"].isin(buckets)]
    if min_rating is not None:
        res = res[res["product_avg_rating"] >= min_rating]
    for word in exclude:
        res = res[~res["product_title"].str.lower().str.contains(word, na=False, regex=False)]
    if dosages:
        res = dosage_filter(res, dosages)

    if beta_quality is None:
        avg_sim      = res["similarity"].mean() if len(res) > 0 else 0.6
        beta_quality = 0.20 if avg_sim < 0.65 else BETA_QUALITY

    if len(res) == 0:
        return pd.DataFrame()

    eff_beta = beta_quality * 1.5 if boost else beta_quality
    res["score"] = (
        res["similarity"] +
        eff_beta        * res["quality_score"].fillna(0) +
        beta_popularity * res["popularity_score"].fillna(0)
    )
    return res.sort_values("score", ascending=False).head(k).reset_index(drop=True)


# ── Recommendation ────────────────────────────────────────────────────────────

def recommend_similar(
    asin_or_title: str,
    model,
    index,
    index_df: pd.DataFrame,
    combined_emb: np.ndarray,
    k: int = 5,
    verbose: bool = False,
):
    """
    Content-based recommendation: find the K products most similar to a given product.

    Differs from search(): uses the product's own combined embedding as the FAISS query
    vector — no text encoding at retrieval time.

    Args:
        asin_or_title : exact parent_asin, or a title substring (resolved via FAISS)
        model         : SentenceTransformer (used only for title-based lookup)
        index         : FAISS IndexFlatIP
        index_df      : product index DataFrame
        combined_emb  : (n_products, dim) combined embedding matrix
        k             : number of recommendations to return
        verbose       : print source product info

    Returns:
        (source_row, recommendations_df)
    """
    if asin_or_title in index_df["parent_asin"].values:
        source = index_df[index_df["parent_asin"] == asin_or_title].iloc[0]
    else:
        q_vec = model.encode(
            [asin_or_title], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        _, I  = index.search(q_vec, 1)
        source = index_df.iloc[I[0][0]]

    if verbose:
        print(f"Prodotto sorgente: {str(source['product_title'])[:70]}")
        print(f"  ASIN: {source['parent_asin']}")

    emb_row = int(source["emb_row"])
    q_vec   = combined_emb[emb_row : emb_row + 1]
    D, I    = index.search(q_vec, k + 1)

    res = index_df.iloc[I[0]].copy()
    res["similarity"] = D[0]
    res = res[res["parent_asin"] != source["parent_asin"]].head(k).reset_index(drop=True)
    return source, res
