import numpy as np
import pandas as pd
import plotly.graph_objects as go

_CATEGORY_COLORS = {
    "Skin Care":      "#E05C8A",
    "Hair Care":      "#F5A623",
    "Supplements":    "#4CAF50",
    "Oral Care":      "#2196F3",
    "Baby Care":      "#9C27B0",
    "Pain Relief":    "#FF5722",
    "Men's Grooming": "#00BCD4",
    "Eye Care":       "#8BC34A",
    "Women's Health": "#E91E63",
    "Other":          "#9E9E9E",
}

_PRICE_BUCKET_COLORS = {
    "Budget":  "#4CAF50",
    "Low":     "#8BC34A",
    "Mid":     "#FFC107",
    "High":    "#FF5722",
    "Premium": "#9C27B0",
}

_CATEGORY_KEYWORDS = {
    # Baby Care must be checked before Skin Care (baby lotion/cream would match Skin Care first)
    "Baby Care": [
        "baby", "infant", "newborn", "diaper", "formula", "baby lotion",
        "baby shampoo", "baby wash", "baby powder", "teething",
    ],
    "Skin Care": [
        "moisturizer", "moisturizing cream", "moisturising", "serum", "retinol",
        "face cream", "sunscreen", "spf", "cleanser", "face wash", "toner",
        "exfoliant", "anti-aging", "wrinkle", "hyaluronic", "niacinamide",
        "acne", "pore", "eye cream", "lip balm", "face mask", "face oil",
        "bb cream", "cc cream", "primer", "foundation", "vitamin c serum",
        "glycolic", "salicylic", "benzoyl", "ceramide", "peptide",
        "collagen cream", "brightening", "dark spot", "skin care", "skincare",
        "facial", "lotion", "body lotion", "body cream", "hand cream",
    ],
    "Hair Care": [
        "shampoo", "conditioner", "hair mask", "hair oil", "hair serum",
        "scalp", "dandruff", "hair loss", "hair growth", "hair color",
        "hair dye", "dry shampoo", "leave-in", "heat protectant", "hair spray",
        "hair gel", "mousse", "volumizing", "keratin", "argan oil for hair",
    ],
    "Supplements": [
        "vitamin", "supplement", "capsule", "softgel", "tablet", "gummy",
        "omega", "probiotic", "protein powder", "collagen peptide", "biotin",
        "zinc", "magnesium", "melatonin", "fish oil", "multivitamin", "iron",
        "calcium", "b12", "d3", "vitamin d", "vitamin c", "folate", "folic",
        "ashwagandha", "turmeric", "elderberry", "echinacea", "coq10",
        "glucosamine", "chondroitin", "creatine", "whey", "plant protein",
        "prebiotic", "digestive enzyme", "nootropic", "adaptogen",
    ],
    "Oral Care": [
        "toothpaste", "toothbrush", "mouthwash", "floss", "whitening",
        "dental", "teeth", "gum", "breath", "tongue scraper", "water flosser",
        "electric toothbrush", "tooth",
    ],
    "Pain Relief": [
        "pain relief", "pain reliever", "arthritis", "joint pain", "muscle",
        "back pain", "ibuprofen", "aspirin", "naproxen", "cbd", "topical pain",
        "heating pad", "ice pack", "tens unit", "compression",
    ],
    "Men's Grooming": [
        "shaving", "razor", "shave cream", "aftershave", "beard", "men's",
        "men's grooming", "cologne", "deodorant for men",
    ],
    "Eye Care": [
        "eye drop", "contact lens", "reading glasses", "eye mask",
        "eye vitamin", "eye supplement", "lutein", "zeaxanthin",
    ],
    "Women's Health": [
        "feminine", "menstrual", "period", "prenatal", "pregnancy", "vaginal",
        "women's health", "breast", "menopause",
    ],
}


def infer_category(title: str) -> str:
    if not isinstance(title, str):
        return "Other"
    t = title.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return category
    return "Other"


def compute_umap(emb: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1):
    import umap as umap_lib
    print(f"UMAP: {emb.shape[0]:,} products × {emb.shape[1]} dim → 2D")
    print(f"  n_neighbors={n_neighbors}, min_dist={min_dist}")
    print("  Estimated time: ~1-2 min on CPU...")
    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
        verbose=True,
        angular_rp_forest=True,
        n_jobs=1,
    )
    coords_2d = reducer.fit_transform(emb)
    x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    print(f"  UMAP complete. x range: [{x_min:.2f}, {x_max:.2f}]")
    return coords_2d, reducer


def plot_embedding_space(
    coords_2d: np.ndarray,
    index_df: pd.DataFrame,
    color_by: str = "category",
    title: str = "Embedding Space",
) -> go.Figure:
    df = index_df.copy().reset_index(drop=True)
    df["_x"] = coords_2d[:, 0]
    df["_y"] = coords_2d[:, 1]

    if color_by == "category":
        color_map = _CATEGORY_COLORS
        groups = df[color_by].fillna("Other").unique()
    elif color_by == "price_bucket":
        color_map = _PRICE_BUCKET_COLORS
        groups = ["Budget", "Low", "Mid", "High", "Premium"]
        groups = [g for g in groups if g in df.get("price_bucket", pd.Series()).values]
    else:
        color_map = {}
        groups = df[color_by].dropna().unique() if color_by in df.columns else ["Other"]

    fig = go.Figure()
    col_series = df[color_by].fillna("Other") if color_by in df.columns else pd.Series(["Other"] * len(df))

    for group in groups:
        mask = col_series == group
        sub = df[mask]
        if sub.empty:
            continue
        color = color_map.get(group, "#888888")

        hover_parts = ["<b>%{customdata[0]}</b>"]
        if "brand" in df.columns:
            hover_parts.append("Brand: %{customdata[1]}")
        if "product_avg_rating" in df.columns:
            hover_parts.append("Rating: %{customdata[2]:.1f}")
        if "price" in df.columns:
            hover_parts.append("Price: $%{customdata[3]:.2f}")
        hover_template = "<br>".join(hover_parts) + "<extra></extra>"

        custom = np.column_stack([
            sub["product_title"].fillna("").str[:80].values,
            sub["brand"].fillna("") .values if "brand" in sub.columns else [""] * len(sub),
            sub["product_avg_rating"].fillna(0).values if "product_avg_rating" in sub.columns else [0] * len(sub),
            sub["price"].fillna(0).values if "price" in sub.columns else [0] * len(sub),
        ])

        fig.add_trace(go.Scatter(
            x=sub["_x"].values,
            y=sub["_y"].values,
            mode="markers",
            name=group,
            marker=dict(size=4, color=color, opacity=0.7,
                        line=dict(width=0.3, color="white")),
            customdata=custom,
            hovertemplate=hover_template,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="UMAP dim 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP dim 2", showgrid=False, zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1100,
        height=700,
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc",
            borderwidth=1,
            itemsizing="constant",
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def save_html(fig: go.Figure, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Chart saved to: {path}")


def plot_query_in_space(
    query: str,
    model,
    umap_model,
    coords_2d: np.ndarray,
    index_df: pd.DataFrame,
    faiss_index,
    top_k: int = 5,
    color_by: str = "category",
) -> go.Figure:
    q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    q_2d = umap_model.transform(q_vec)
    D, I = faiss_index.search(q_vec, top_k)
    nearest_idx = I[0]

    fig = plot_embedding_space(coords_2d, index_df, color_by=color_by,
                               title=f"Query: \"{query}\"")

    # Highlight nearest neighbours with red circles
    nn_df = index_df.iloc[nearest_idx].reset_index(drop=True)
    nn_x = coords_2d[nearest_idx, 0]
    nn_y = coords_2d[nearest_idx, 1]
    nn_labels = nn_df["product_title"].fillna("").str[:50].tolist()

    fig.add_trace(go.Scatter(
        x=nn_x, y=nn_y,
        mode="markers",
        name=f"Top-{top_k} results",
        marker=dict(size=12, color="rgba(0,0,0,0)", opacity=1,
                    line=dict(color="#E53935", width=2.5)),
        text=nn_labels,
        hovertemplate="<b>%{text}</b><extra>Top result</extra>",
    ))

    # Query star
    fig.add_trace(go.Scatter(
        x=[q_2d[0, 0]], y=[q_2d[0, 1]],
        mode="markers+text",
        name="Query",
        marker=dict(symbol="star", size=18, color="#FFD600",
                    line=dict(color="#333", width=1.5)),
        text=[f"  {query[:40]}"],
        textposition="middle right",
        textfont=dict(size=12, color="#333"),
        hovertemplate=f"<b>Query:</b> {query}<extra></extra>",
    ))

    return fig
