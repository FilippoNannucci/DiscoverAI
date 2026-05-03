"""
Pseudo-relevance scoring + standard IR metrics (NDCG, MRR, Recall, Precision, MAP).

Scoring strategy (4 livelli):
    rel = 0  → no must_keyword match  OPPURE  excluded_keyword presente
    rel = 1  → must match, bonus_ratio < 0.34
    rel = 2  → must match, bonus_ratio ∈ [0.34, 0.67)
    rel = 3  → must match, bonus_ratio ≥ 0.67  AND  product_avg_rating ≥ min_rating

Quando bonus_keywords è vuoto → solo livelli 0/2.
Quando intent == "off_topic" → tutti i prodotti hanno rel=0 (sistema dovrebbe rifiutare).
"""

import re
import numpy as np


def _contains_kw(text: str, kw: str) -> bool:
    """Match case-insensitive con boundary di parola; gestisce keyword multi-token."""
    if not text or not kw:
        return False
    pat = r"\b" + re.escape(kw.lower()) + r"\b"
    return re.search(pat, text.lower()) is not None


def score_product(product_text: str, product_rating: float, query_def: dict) -> int:
    """Restituisce livello di rilevanza ∈ {0,1,2,3} per un prodotto rispetto a una query."""
    text = (product_text or "").lower()

    # Off-topic: niente è rilevante
    if query_def["intent"] == "off_topic":
        return 0

    # Excluded keywords → squalifica
    for ex in query_def.get("excluded_keywords", []):
        if _contains_kw(text, ex):
            return 0

    # Must keywords: almeno una deve apparire (placeholder __NONE__ per off-topic)
    must = query_def["must_keywords"]
    if must and must != ["__NONE__"]:
        if not any(_contains_kw(text, m) for m in must):
            return 0

    # Conteggio bonus matched (count, non ratio — più robusto a numero di bonus diverso)
    bonus = query_def.get("bonus_keywords", [])
    if not bonus:
        return 2  # match must senza bonus → "rilevante"

    matched = sum(1 for b in bonus if _contains_kw(text, b))

    # rel=1 → must matched ma 0 bonus
    if matched == 0:
        return 1
    # rel=3 → must + ≥3 bonus + rating buono
    if matched >= 3 and product_rating is not None and product_rating >= query_def.get("min_rating", 4.0):
        return 3
    # rel=2 → must + ≥1 bonus
    return 2


def build_qrels(products_df, eval_queries):
    """
    Costruisce la matrice di rilevanza: dict[qid] -> dict[parent_asin -> rel_level].
    products_df deve avere colonne: parent_asin, product_text_base (o equivalente), product_avg_rating
    """
    text_col = "product_text_base" if "product_text_base" in products_df.columns else "product_title"
    rating_col = "product_avg_rating"
    qrels = {}
    for q in eval_queries:
        per_query = {}
        for _, row in products_df.iterrows():
            rel = score_product(row[text_col], row.get(rating_col), q)
            if rel > 0:
                per_query[row["parent_asin"]] = rel
        qrels[q["qid"]] = per_query
    return qrels


# ── Metriche ──────────────────────────────────────────────────────────────────

def dcg(rels):
    """DCG con formulazione (2^rel - 1) / log2(i+2)."""
    rels = np.asarray(rels, dtype=float)
    if rels.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((np.power(2.0, rels) - 1) / discounts))


def ndcg_at_k(retrieved_asins, qrels_for_query, k=5):
    rel_seq = [qrels_for_query.get(a, 0) for a in retrieved_asins[:k]]
    actual = dcg(rel_seq)
    ideal_rels = sorted(qrels_for_query.values(), reverse=True)[:k]
    ideal = dcg(ideal_rels)
    return actual / ideal if ideal > 0 else 0.0


def mrr(retrieved_asins, qrels_for_query, threshold=2):
    """Reciprocal rank della prima asin con rel >= threshold."""
    for i, a in enumerate(retrieved_asins, 1):
        if qrels_for_query.get(a, 0) >= threshold:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved_asins, qrels_for_query, k=5, threshold=2):
    rel_set = {a for a, r in qrels_for_query.items() if r >= threshold}
    if not rel_set:
        return None  # query off-topic / no relevant: skipped
    hits = sum(1 for a in retrieved_asins[:k] if a in rel_set)
    return hits / len(rel_set)


def precision_at_k(retrieved_asins, qrels_for_query, k=5, threshold=2):
    if not retrieved_asins[:k]:
        return 0.0
    hits = sum(1 for a in retrieved_asins[:k] if qrels_for_query.get(a, 0) >= threshold)
    return hits / k


def average_precision(retrieved_asins, qrels_for_query, threshold=2, k=10):
    rel_set = {a for a, r in qrels_for_query.items() if r >= threshold}
    if not rel_set:
        return None
    hits = 0
    sum_prec = 0.0
    for i, a in enumerate(retrieved_asins[:k], 1):
        if a in rel_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(rel_set), k)


def aggregate_metrics(rows, k=5):
    """rows = [(qid, ndcg, mrr, recall, precision, ap), ...]; recall e ap possono essere None."""
    import statistics as st
    def _safe_mean(xs):
        xs = [x for x in xs if x is not None]
        return st.fmean(xs) if xs else float("nan")
    ndcg = _safe_mean([r[1] for r in rows])
    mrr_v = _safe_mean([r[2] for r in rows])
    rec = _safe_mean([r[3] for r in rows])
    prec = _safe_mean([r[4] for r in rows])
    ap = _safe_mean([r[5] for r in rows])
    return {"NDCG@{}".format(k): ndcg, "MRR": mrr_v, "Recall@{}".format(k): rec,
            "Precision@{}".format(k): prec, "MAP": ap}
