"""
vector_db.py — ChromaDB vector database layer.

Drop-in alternative to FAISS for the production demo. The signature of
`search_chroma()` mirrors `search_v3()` from `search.py`, so swapping the
backend requires no notebook or demo changes.

Why ChromaDB over a plain FAISS index:
  - Disk persistence: embeddings are loaded once and reused across runs,
    no re-indexing needed at startup.
  - Native metadata filtering: `price_bucket` and `avg_rating` filters are
    applied inside the database, not in Python after the fetch — more
    efficient at scale.
  - Production-ready: the same API works locally, on a remote server, or
    against ChromaDB Cloud.

Install (one-off)::

    pip install chromadb

Typical workflow::

    from mean_squared_terrors.vector_db import build_chroma_index, search_chroma

    # 1. Build the index (once, then reload from disk)
    collection = build_chroma_index(index_df, combined_emb, persist_dir="./chroma_db")

    # 2. Query
    results = search_chroma("moisturizer sensitive skin", model, collection, index_df, k=5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .config import BETA_QUALITY, BETA_POPULARITY, N_CANDIDATES
from .search import parse_query_v2, expand_query, extract_dosages, dosage_filter


# ── ChromaDB import with explicit fallback ────────────────────────────────────

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


def _check_chroma() -> None:
    if not CHROMA_AVAILABLE:
        raise ImportError(
            "ChromaDB is not installed. Run: pip install chromadb"
        )


# ── Build / load index ────────────────────────────────────────────────────────

def build_chroma_index(
    index_df: pd.DataFrame,
    combined_emb: np.ndarray,
    persist_dir: str = "./chroma_db",
    collection_name: str = "amazon_products",
    force_rebuild: bool = False,
) -> "chromadb.Collection":
    """
    Build (or load from disk) a ChromaDB collection from the combined embeddings.

    If the collection already exists in `persist_dir` and `force_rebuild` is
    False, it is loaded as-is — no re-indexing.

    Args:
        index_df:        DataFrame from embedding_index_enriched.csv. Must
                         include: parent_asin, product_title, price_bucket,
                         product_avg_rating, quality_score, popularity_score,
                         emb_row.
        combined_emb:    (n_products × 768) array from combined_embeddings.npy.
        persist_dir:     directory where ChromaDB persists its data.
        collection_name: ChromaDB collection name.
        force_rebuild:   if True, drops and rebuilds the collection from scratch.

    Returns:
        A `chromadb.Collection` ready for `search_chroma()`.
    """
    _check_chroma()

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # Reuse the existing collection unless explicitly asked to rebuild
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing and not force_rebuild:
        print(f"[ChromaDB] collection '{collection_name}' found — loaded from {persist_dir}")
        return client.get_collection(collection_name)

    # Drop the old one if force_rebuild was requested
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"[ChromaDB] collection '{collection_name}' dropped — rebuilding")

    collection = client.create_collection(
        name=collection_name,
        # cosine distance = 1 - cosine_similarity
        # ChromaDB supports "cosine", "l2", "ip"
        metadata={"hnsw:space": "cosine"},
    )

    # Insert in batches (ChromaDB limits ~5000 items per upsert)
    BATCH = 2000
    n     = len(index_df)
    print(f"[ChromaDB] indexing {n} products in batches of {BATCH}...")

    required_cols = {
        "parent_asin", "product_title", "price_bucket",
        "product_avg_rating", "quality_score", "popularity_score", "emb_row",
    }
    missing = required_cols - set(index_df.columns)
    if missing:
        raise ValueError(f"index_df is missing columns: {missing}")

    for start in range(0, n, BATCH):
        batch_df = index_df.iloc[start : start + BATCH]
        ids      = batch_df["parent_asin"].tolist()
        rows     = batch_df["emb_row"].astype(int).tolist()
        embs     = combined_emb[rows].astype(np.float32).tolist()

        metadatas = [
            {
                "title":            str(row["product_title"])[:500],  # ChromaDB caps string lengths
                "price_bucket":     str(row.get("price_bucket", "unknown")),
                "avg_rating":       float(row.get("product_avg_rating", 0.0) or 0.0),
                "quality_score":    float(row.get("quality_score", 0.0) or 0.0),
                "popularity_score": float(row.get("popularity_score", 0.0) or 0.0),
                "price":            float(row.get("price", 0.0) or 0.0),
                "brand":            str(row.get("brand", ""))[:200],
                "emb_row":          int(row["emb_row"]),
            }
            for _, row in batch_df.iterrows()
        ]

        collection.add(ids=ids, embeddings=embs, metadatas=metadatas)
        print(f"  [{start + len(batch_df)}/{n}] OK")

    print(f"[ChromaDB] indexing complete — {n} products persisted in '{persist_dir}'")
    return collection


# ── Search ────────────────────────────────────────────────────────────────────

def search_chroma(
    query: str,
    model,
    collection: "chromadb.Collection",
    index_df: pd.DataFrame,
    k: int = 5,
    price_buckets: list | None = None,
    min_rating: float = None,
    beta_quality: float = None,
    beta_popularity: float = BETA_POPULARITY,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Semantic search on ChromaDB, mirroring the `search_v3` pipeline:
    synonym expansion → query parsing → encode → ChromaDB query → re-ranking.

    Differences from the FAISS path:
      - Metadata filters (`price_bucket`, `avg_rating`) are pushed *inside*
        the database, not applied in Python after the fetch.
      - ChromaDB returns cosine *distances* (0 = identical, 2 = opposite);
        we convert to similarity = 1 − distance to match the FAISS scale.

    Args:
        query:           raw user text.
        model:           already-loaded SentenceTransformer.
        collection:      ChromaDB collection from `build_chroma_index()`.
        index_df:        original DataFrame (used for columns not stored in metadata).
        k:               final result count.
        price_buckets:   accepted price-bucket values, e.g. ["budget", "low"].
        min_rating:      minimum average rating (None disables the filter).
        beta_quality:    quality-score weight (None = adaptive, like search_v3).
        beta_popularity: popularity-score weight (defaults from config).
        verbose:         debug printing.

    Returns:
        DataFrame with parent_asin, product_title, similarity, score, …
    """
    _check_chroma()

    # 1. synonym expansion + parsing
    expanded = expand_query(query)
    if expanded != query.lower().strip() and verbose:
        print(f"  Expansion: '{query}' → '{expanded}'")

    parsed  = parse_query_v2(expanded)
    buckets = price_buckets if price_buckets is not None else parsed["price_buckets"]
    boost   = parsed["quality_boost"]
    exclude = parsed["exclude_words"]
    dosages = extract_dosages(query)

    if verbose:
        print(f"  clean query     : {parsed['clean']}")
        print(f"  price buckets   : {buckets}")
        print(f"  exclude words   : {exclude}")
        print(f"  detected dosages: {dosages}")

    # 2. encode
    q_vec = model.encode(
        [parsed["clean"]], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32).tolist()

    # 3. build the ChromaDB `where` filter
    # ChromaDB operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
    where_filter: dict | None = None
    conditions = []

    if buckets:
        conditions.append({"price_bucket": {"$in": buckets}})
    if min_rating is not None:
        conditions.append({"avg_rating": {"$gte": float(min_rating)}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    # 4. query ChromaDB — fetch a few extra candidates to allow negation /
    #    dosage filtering in Python without dropping under k results
    n_cand = N_CANDIDATES * 3 if (buckets or exclude or dosages) else N_CANDIDATES * 2

    chroma_results = collection.query(
        query_embeddings=q_vec,
        n_results=min(n_cand, collection.count()),
        where=where_filter,
        include=["metadatas", "distances"],
    )

    if not chroma_results["ids"][0]:
        if verbose:
            print("  no results from ChromaDB")
        return pd.DataFrame()

    # 5. build the candidate DataFrame
    ids        = chroma_results["ids"][0]
    distances  = chroma_results["distances"][0]   # cosine distance in [0, 2]
    metadatas  = chroma_results["metadatas"][0]

    # similarity = 1 − cosine_distance (with normalised embeddings: [0, 1])
    similarities = [1.0 - d for d in distances]

    res = pd.DataFrame(metadatas)
    res["parent_asin"] = ids
    res["similarity"]  = similarities
    res = res.rename(columns={
        "title":      "product_title",
        "avg_rating": "product_avg_rating",
    })

    # 6. post-fetch filters (negation, dosages)
    for word in exclude:
        res = res[~res["product_title"].str.lower().str.contains(word, na=False, regex=False)]
    if dosages:
        res = dosage_filter(res, dosages)

    if len(res) == 0:
        return pd.DataFrame()

    # 7. adaptive beta (same logic as search_v3)
    if beta_quality is None:
        avg_sim      = res["similarity"].mean()
        beta_quality = 0.20 if avg_sim < 0.65 else BETA_QUALITY

    eff_beta = beta_quality * 1.5 if boost else beta_quality

    # 8. final re-ranking
    res["score"] = (
        res["similarity"]
        + eff_beta        * res["quality_score"].fillna(0)
        + beta_popularity * res["popularity_score"].fillna(0)
    )

    return res.sort_values("score", ascending=False).head(k).reset_index(drop=True)


# ── ChromaDB-backed recommendation ────────────────────────────────────────────

def recommend_similar_chroma(
    asin_or_title: str,
    model,
    collection: "chromadb.Collection",
    index_df: pd.DataFrame,
    combined_emb: np.ndarray,
    k: int = 5,
    verbose: bool = False,
):
    """
    Content-based recommendation over ChromaDB. Same logic as
    `recommend_similar()` in search.py, but uses the ChromaDB backend.

    Returns:
        (source_row, recommendations_df)
    """
    _check_chroma()

    # Find the source product
    if asin_or_title in index_df["parent_asin"].values:
        source = index_df[index_df["parent_asin"] == asin_or_title].iloc[0]
    else:
        q_vec    = model.encode(
            [asin_or_title], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        # Use ChromaDB to find the product whose title is closest to the query
        chroma_r = collection.query(
            query_embeddings=q_vec.tolist(),
            n_results=1,
            include=["metadatas"],
        )
        if not chroma_r["ids"][0]:
            raise ValueError(f"Product not found: '{asin_or_title}'")
        source_asin = chroma_r["ids"][0][0]
        source      = index_df[index_df["parent_asin"] == source_asin].iloc[0]

    if verbose:
        print(f"source product: {str(source['product_title'])[:70]}")
        print(f"  ASIN: {source['parent_asin']}")

    # Query using the source product's embedding as input
    emb_row = int(source["emb_row"])
    q_vec   = combined_emb[emb_row : emb_row + 1].astype(np.float32)

    chroma_r = collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=k + 1,
        include=["metadatas", "distances"],
    )

    ids          = chroma_r["ids"][0]
    distances    = chroma_r["distances"][0]
    metadatas    = chroma_r["metadatas"][0]
    similarities = [1.0 - d for d in distances]

    res = pd.DataFrame(metadatas)
    res["parent_asin"] = ids
    res["similarity"]  = similarities
    res = res.rename(columns={"title": "product_title", "avg_rating": "product_avg_rating"})
    res = res[res["parent_asin"] != source["parent_asin"]].head(k).reset_index(drop=True)

    return source, res
