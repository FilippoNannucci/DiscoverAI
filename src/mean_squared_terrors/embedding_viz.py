"""
embedding_viz.py — 2-D visualisation of the embedding space.

Uses UMAP to project the combined embeddings (~7,593 × 768) to 2-D and
Plotly to render an interactive scatter plot where each point is a product.

Functionality:
  - compute_umap()         : reduce embeddings to 2-D (run once)
  - infer_category()       : tag each product with a macro-category from its title
  - plot_embedding_space() : interactive scatter coloured by category / price / rating
  - plot_query_in_space()  : project a query into the same space as a star marker
  - save_html()            : write a standalone HTML file (no Jupyter required)

Install::

    pip install umap-learn plotly

Typical notebook usage::

    from mean_squared_terrors.embedding_viz import (
        compute_umap, infer_category, plot_embedding_space, plot_query_in_space
    )
    coords_2d, umap_model = compute_umap(combined_emb)        # ~1-2 min first run
    index_df["category"]  = index_df["product_title"].apply(infer_category)
    fig = plot_embedding_space(coords_2d, index_df)
    fig.show()

    # With a query overlaid
    fig = plot_query_in_space("moisturizer sensitive skin", model, umap_model,
                               coords_2d, index_df, top_k=5)
    fig.show()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _check_deps() -> None:
    missing = []
    if not UMAP_AVAILABLE:
        missing.append("umap-learn")
    if not PLOTLY_AVAILABLE:
        missing.append("plotly")
    if missing:
        raise ImportError(f"Install: pip install {' '.join(missing)}")


# ── Title-derived macro-category ─────────────────────────────────────────────

# keyword → category. Order matters: first match wins.
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Skincare",     ["moisturizer", "serum", "retinol", "cream", "lotion",
                      "toner", "cleanser", "sunscreen", "spf", "face wash",
                      "hyaluronic", "niacinamide", "aha", "bha", "exfoliant",
                      "eye cream", "lip balm", "facial"]),
    ("Hair Care",    ["shampoo", "conditioner", "hair", "scalp", "dandruff",
                      "biotin", "hair growth", "hair loss", "dry hair"]),
    ("Supplements",  ["vitamin", "supplement", "capsule", "tablet", "capsules",
                      "omega", "probiotic", "collagen", "protein", "zinc",
                      "magnesium", "melatonin", "fish oil", "gummy", "gummies",
                      "immune", "energy", "b12", "iron", "calcium", "d3"]),
    ("Oral Care",    ["toothpaste", "mouthwash", "dental", "whitening", "floss",
                      "tooth", "breath", "gum"]),
    ("Body Care",    ["body wash", "soap", "lotion body", "deodorant", "antiperspirant",
                      "hand cream", "foot", "shaving", "razor", "body oil",
                      "shower gel", "bath"]),
    ("Pain Relief",  ["pain relief", "ibuprofen", "aspirin", "joint", "muscle",
                      "arthritis", "glucosamine", "topical", "ointment",
                      "heating pad", "brace", "support"]),
    ("Baby & Kids",  ["baby", "infant", "kids", "children", "diaper", "newborn",
                      "toddler"]),
    ("Eye & Ear",    ["eye drop", "contact", "vision", "ear", "hearing",
                      "eye care", "lubricant eye"]),
    ("Medical",      ["bandage", "gauze", "first aid", "antiseptic", "wound",
                      "thermometer", "blood pressure", "glucose", "test strip",
                      "medical"]),
]

_CATEGORY_COLORS: dict[str, str] = {
    "Skincare":    "#4C78A8",   # blue
    "Hair Care":   "#F58518",   # orange
    "Supplements": "#54A24B",   # green
    "Oral Care":   "#E45756",   # red
    "Body Care":   "#B279A2",   # purple
    "Pain Relief": "#FF9DA6",   # pink
    "Baby & Kids": "#9D755D",   # brown
    "Eye & Ear":   "#BAB0AC",   # light grey
    "Medical":     "#72B7B2",   # teal
    "Other":       "#D3D3D3",   # grey
}


def infer_category(title: str) -> str:
    """
    Tag a product with a macro-category based on its title.
    First matching rule in `_CATEGORY_RULES` wins. Falls back to 'Other'.
    """
    t = str(title).lower()
    for category, keywords in _CATEGORY_RULES:
        for kw in keywords:
            if kw in t:
                return category
    return "Other"


# ── UMAP ──────────────────────────────────────────────────────────────────────

def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 20,
    min_dist: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, "umap.UMAP"]:
    """
    Reduce the embedding matrix to 2-D with UMAP.

    Parameters chosen for this corpus:
      - n_neighbors=20: balances local and global structure on ~7k points
      - min_dist=0.1:   compact-but-not-collapsed clusters
      - metric='cosine': matches the cosine similarity used by FAISS

    Args:
        embeddings:   (n_products × 768) L2-normalised array.
        n_neighbors:  UMAP neighbour count.
        min_dist:     minimum distance between points in the 2-D embedding.
        random_state: seed for reproducibility.
        verbose:      show UMAP's own progress bar.

    Returns:
        (coords_2d, umap_model)
        - coords_2d : (n_products × 2) array of 2-D coordinates.
        - umap_model: the fitted model — call `.transform(new_emb)` to
                      project query embeddings into the same space.
    """
    _check_deps()

    if verbose:
        print(f"UMAP: {embeddings.shape[0]:,} products × {embeddings.shape[1]} dim → 2D")
        print(f"  n_neighbors={n_neighbors}, min_dist={min_dist}")
        print("  Expected wall time: ~1-2 min on CPU")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        verbose=verbose,
    )
    coords_2d = reducer.fit_transform(embeddings)

    if verbose:
        print(f"  UMAP done. x range: [{coords_2d[:,0].min():.2f}, {coords_2d[:,0].max():.2f}]")

    return coords_2d.astype(np.float32), reducer


# ── Main scatter plot ─────────────────────────────────────────────────────────

def plot_embedding_space(
    coords_2d: np.ndarray,
    index_df: pd.DataFrame,
    color_by: str = "category",
    title: str = "DiscoverAI — Product Embedding Space",
    width: int = 1000,
    height: int = 700,
    point_size: int = 4,
    opacity: float = 0.65,
) -> "go.Figure":
    """
    Interactive Plotly scatter plot of the 2-D embedding space.

    Each point is a product. Hover shows title, brand, category, price and rating.

    Args:
        coords_2d: (n_products × 2) array from `compute_umap()`.
        index_df:  DataFrame with product_title, brand, price,
                   product_avg_rating, price_bucket. Must include a
                   'category' column when `color_by='category'`
                   (set it via
                   `index_df['category'] = index_df['product_title'].apply(infer_category)`).
        color_by:  one of 'category', 'price_bucket', 'product_avg_rating'.
        title:     plot title.
        width:     plot width in pixels.
        height:    plot height in pixels.

    Returns:
        A Plotly `go.Figure` — call `.show()` to render or `.write_html()` to save.
    """
    _check_deps()

    df = index_df.copy()
    df["x"] = coords_2d[:, 0]
    df["y"] = coords_2d[:, 1]

    # Hover tooltip
    df["hover"] = (
        "<b>" + df["product_title"].str[:60].fillna("") + "</b><br>"
        + "Brand: "    + df["brand"].fillna("N/A").str[:30] + "<br>"
        + "Price: $"   + df["price"].fillna(0).round(2).astype(str) + "<br>"
        + "Rating: "   + df["product_avg_rating"].fillna(0).round(1).astype(str) + "★<br>"
        + "Category: " + df.get("category", pd.Series(["?"] * len(df)))
    )

    fig = go.Figure()

    if color_by == "category" and "category" in df.columns:
        # One trace per category → coloured legend with click-to-toggle
        for cat in sorted(df["category"].unique()):
            mask = df["category"] == cat
            sub  = df[mask]
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"],
                mode="markers",
                name=f"{cat} ({mask.sum():,})",
                marker=dict(
                    size=point_size,
                    opacity=opacity,
                    color=_CATEGORY_COLORS.get(cat, "#999999"),
                ),
                text=sub["hover"],
                hovertemplate="%{text}<extra></extra>",
                customdata=sub["parent_asin"],
            ))

    elif color_by == "price_bucket":
        bucket_colors = {
            "budget":  "#2ecc71",
            "low":     "#3498db",
            "mid":     "#f39c12",
            "high":    "#e74c3c",
            "premium": "#9b59b6",
            "unknown": "#bdc3c7",
        }
        for bucket in ["budget", "low", "mid", "high", "premium", "unknown"]:
            mask = df["price_bucket"].fillna("unknown") == bucket
            if not mask.any():
                continue
            sub = df[mask]
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"],
                mode="markers",
                name=f"{bucket.capitalize()} ({mask.sum():,})",
                marker=dict(size=point_size, opacity=opacity,
                            color=bucket_colors.get(bucket, "#999")),
                text=sub["hover"],
                hovertemplate="%{text}<extra></extra>",
            ))

    elif color_by == "product_avg_rating":
        fig.add_trace(go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(
                size=point_size,
                opacity=opacity,
                color=df["product_avg_rating"].fillna(3.0),
                colorscale="RdYlGn",
                cmin=1, cmax=5,
                colorbar=dict(title="Rating ★", thickness=15),
                showscale=True,
            ),
            text=df["hover"],
            hovertemplate="%{text}<extra></extra>",
            name="Products",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="UMAP dim 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP dim 2", showgrid=False, zeroline=False),
        width=width, height=height,
        legend=dict(
            title="Category",
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        margin=dict(l=60, r=60, t=70, b=60),
    )

    return fig


# ── Query-overlay plot ────────────────────────────────────────────────────────

def plot_query_in_space(
    query: str,
    model,
    umap_model: "umap.UMAP",
    coords_2d: np.ndarray,
    index_df: pd.DataFrame,
    faiss_index=None,
    top_k: int = 5,
    color_by: str = "category",
    width: int = 1000,
    height: int = 700,
) -> "go.Figure":
    """
    Project a query into the embedding space and render it as a star marker;
    optionally highlight the top-k closest products with red rings.

    Especially useful in presentations: it visually shows where a query
    lands relative to the product clusters.

    Args:
        query:       query text.
        model:       already-loaded SentenceTransformer.
        umap_model:  the fitted UMAP model from `compute_umap()`.
        coords_2d:   (n_products × 2) array from `compute_umap()`.
        index_df:    product DataFrame (with 'category' if `color_by='category'`).
        faiss_index: optional FAISS index for retrieving the top-k closest.
        top_k:       how many top products to circle.
        color_by:    same parameter as `plot_embedding_space()`.

    Returns:
        A `go.Figure` with the query as a yellow star and top-k circled in red.
    """
    _check_deps()

    # 1. base figure
    fig = plot_embedding_space(
        coords_2d, index_df,
        color_by=color_by,
        title=f'Query: "{query}"',
        width=width, height=height,
    )

    # 2. encode and project the query through UMAP
    q_vec_full = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    q_2d = umap_model.transform(q_vec_full)   # (1, 2)

    # 3. star marker for the query
    fig.add_trace(go.Scatter(
        x=[q_2d[0, 0]], y=[q_2d[0, 1]],
        mode="markers+text",
        marker=dict(
            symbol="star",
            size=18,
            color="#FFD700",        # gold
            line=dict(color="#000000", width=1.5),
        ),
        text=["← Query"],
        textposition="middle right",
        textfont=dict(size=13, color="#000000"),
        name=f'Query: "{query[:30]}"',
        hovertemplate=f"<b>Query</b>: {query}<extra></extra>",
    ))

    # 4. highlight the top-k products (red rings + dotted lines to the query)
    if faiss_index is not None:
        D, I     = faiss_index.search(q_vec_full, top_k)
        top_idx  = I[0]
        top_df   = index_df.iloc[top_idx].copy()
        top_xy   = coords_2d[top_idx]
        top_sims = D[0]

        fig.add_trace(go.Scatter(
            x=top_xy[:, 0], y=top_xy[:, 1],
            mode="markers",
            marker=dict(
                symbol="circle-open",
                size=14,
                color="#FF4136",    # red
                line=dict(width=2.5),
            ),
            name=f"Top-{top_k} results",
            text=[
                f"<b>#{i+1}</b> {str(row['product_title'])[:55]}<br>"
                f"Sim: {top_sims[i]:.3f} | Rating: {row.get('product_avg_rating','?')}★"
                for i, (_, row) in enumerate(top_df.iterrows())
            ],
            hovertemplate="%{text}<extra></extra>",
        ))

        # Dotted lines from the query to each top-k point
        for i in range(len(top_idx)):
            fig.add_shape(type="line",
                x0=q_2d[0,0], y0=q_2d[0,1],
                x1=top_xy[i,0], y1=top_xy[i,1],
                line=dict(color="#FF4136", width=0.8, dash="dot"),
                opacity=0.5,
            )

    return fig


# ── Standalone HTML export ────────────────────────────────────────────────────

def save_html(fig: "go.Figure", path: str) -> None:
    """Save the figure as a standalone HTML file (opens in any browser, no Jupyter required)."""
    _check_deps()
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Plot saved to: {path}")
