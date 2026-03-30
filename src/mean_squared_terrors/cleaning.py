"""Text cleaning and product-text assembly utilities (notebook 01)."""

import re
from html import unescape

import numpy as np
import pandas as pd

from .config import (
    CAP_PRODUCT_TOKENS,
    FORM_KEYS,
    REVIEWS_PER_RATING,
    SEMANTIC_DETAIL_KEYS,
)


def clean_text(text):
    """Strip HTML tags, unescape HTML entities, normalise unicode chars, collapse whitespace."""
    if pd.isna(text) or not text:
        return text
    text = unescape(str(text))
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s.,!?'/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) > 2 else np.nan


def flatten_to_text(val, cap_tokens=None):
    """Convert list or string to clean text, with optional token cap."""
    if isinstance(val, list):
        parts = [clean_text(str(x)) for x in val if x and str(x).strip()]
        parts = [p for p in parts if pd.notna(p) and p]
        text  = ' '.join(parts)
    elif isinstance(val, str):
        text = clean_text(val)
    else:
        return np.nan
    if not text or pd.isna(text):
        return np.nan
    if cap_tokens:
        text = ' '.join(text.split()[:cap_tokens])
    return text if text else np.nan


def extract_details_semantic(details, cap_tokens=None):
    """
    Extract semantic fields from the details dict.
    Returns (form_text, rest_text): FORM fields are separated so they can be
    placed at the top of the product text for category disambiguation.
    """
    if not isinstance(details, dict):
        return np.nan, np.nan
    form_parts, rest_parts = [], []
    for k, v in details.items():
        if k not in SEMANTIC_DETAIL_KEYS:
            continue
        v_str = str(v).strip()
        if not v_str or v_str.lower() in ('none', 'n/a', '-', '—', ''):
            continue
        (form_parts if k in FORM_KEYS else rest_parts).append(f"{k}: {v_str}")
    form_text = '. '.join(form_parts) if form_parts else np.nan
    rest_text = '. '.join(rest_parts) if rest_parts else np.nan
    if cap_tokens and pd.notna(rest_text):
        rest_text = ' '.join(str(rest_text).split()[:cap_tokens])
    return form_text, rest_text


def get_brand(row):
    """Extract brand from multiple fallback sources: brand col, store col, details dict."""
    for field in ('brand', 'store'):
        val = row.get(field)
        if val and str(val).strip().lower() not in ('none', '', 'nan'):
            return clean_text(str(val))
    d = row.get('details')
    if isinstance(d, dict):
        for key in ('Brand', 'Manufacturer'):
            val = d.get(key)
            if val and str(val).strip().lower() not in ('none', '', 'nan'):
                return clean_text(str(val))
    return np.nan


def build_product_text(row):
    """
    Assembles product_text_base in the optimal order for embedding:
    TITLE → BRAND → FORM → FEATURES → DESCRIPTION → SPECS

    FORM is placed right after TITLE so the model immediately knows the product
    category before reading anything else. This disambiguates e.g.
    'Biotin Shampoo' vs 'Biotin Capsule' which otherwise share nearly
    identical titles and descriptions.
    """
    parts = []
    if pd.notna(row.get("product_title")):
        parts.append(f"TITLE: {row['product_title']}")
    if pd.notna(row.get("brand")):
        parts.append(f"BRAND: {row['brand']}")
    if pd.notna(row.get("form_text")):
        parts.append(f"FORM: {row['form_text']}")
    if pd.notna(row.get("features_clean")):
        parts.append(f"FEATURES: {row['features_clean']}")
    if pd.notna(row.get("description_clean")):
        parts.append(f"DESCRIPTION: {row['description_clean']}")
    if pd.notna(row.get("details_rest")):
        parts.append(f"SPECS: {row['details_rest']}")
    full_text = "\n".join(parts)
    words = full_text.split()
    if len(words) > CAP_PRODUCT_TOKENS:
        full_text = " ".join(words[:CAP_PRODUCT_TOKENS])
    return full_text


def select_balanced(grp):
    """Select reviews with balanced rating distribution, most helpful first within each band."""
    selected = []
    for rating, cap in REVIEWS_PER_RATING.items():
        bucket = grp[grp["rating"] == rating].nlargest(cap, "helpful_vote")
        selected.append(bucket)
    return pd.concat(selected) if selected else grp.head(0)
