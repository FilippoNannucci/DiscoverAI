"""
guardrail.py — Query Intent Validation Layer.

Intercepts off-domain queries BEFORE they reach the search engine.
Specific to the Health & Personal Care catalogue (Amazon Reviews 2023).

Three layers, ordered by computational cost (cheapest first):

  Layer 1 — Structural check  (O(1), no model)
      Rejects queries that are too short, too long, or that match a regex
      blacklist of clearly off-topic terms.

  Layer 2 — Centroid similarity  (O(d), one dot product)
      Cosine similarity between the query embedding and the corpus centroid.
      If the query lies too far from the corpus mean, it is rejected.

  Layer 3 — FAISS top-1 confidence  (O(log n))
      If the closest product in the entire catalogue has similarity below a
      threshold, the query has no meaningful match in the catalogue.

Usage::

    from mean_squared_terrors.guardrail import QueryGuardrail

    guardrail = QueryGuardrail(model, faiss_index, index_df, combined_emb)
    result    = guardrail.validate("who won the Champions League?")
    # GuardrailResult(is_valid=False, reason="off_topic_structural",
    #                 confidence=0.95, message="...")

    result = guardrail.validate("affordable moisturizer sensitive skin")
    # GuardrailResult(is_valid=True, reason="passed", confidence=0.87, message="")
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Blacklist patterns ────────────────────────────────────────────────────────
# Patterns that identify off-domain queries with high confidence.
# Built empirically against Health & Personal Care.

_BLACKLIST_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    # Sports & results
    r"\b(score|goal|match|champion|league|fifa|nba|nfl|serie\s*a|calcio|partita)\b",
    # Weather
    r"\b(weather|forecast|temperatura|pioggia|neve|meteo|clima)\b",
    # Finance
    r"\b(stock|share|bitcoin|crypto|borsa|azioni|invest|trading|forex)\b",
    # Politics & news
    r"\b(president|election|governo|politica|minister|voto|poll)\b",
    # Cooking & recipes (off-domain)
    r"\b(recipe|recipes|ricetta|ingredient[ie]|cook|cooking|cucinare|forno|pasta\s+al|carbonara|amatriciana)\b",
    # Unrelated tech
    r"\b(laptop|computer|phone|smartphone|iphone|android|gaming|gpu|cpu|javascript|python\s+import)\b",
    # Generic-knowledge questions
    r"\b(who\s+is|who\s+was|what\s+is\s+the\s+capital|when\s+did|define\s+)\b",
    r"\b(chi\s+è|cos'è|quando\s+è|dove\s+si\s+trova)\b",
    # Travel
    r"\b(flight|hotel|airbnb|vacanza|viaggio|volo|booking)\b",
    # Personal questions to the chatbot
    r"\b(how\s+are\s+you|what\s+can\s+you\s+do|tell\s+me\s+a\s+joke|sei\s+un\s+robot)\b",
]]

# Positive in-domain signals.
# If the query contains any of these, treat it as in-domain and skip the
# blacklist short-circuit (semantic checks still apply).
_POSITIVE_SIGNALS: frozenset[str] = frozenset([
    # Product categories
    "shampoo", "conditioner", "moisturizer", "serum", "cleanser", "toner",
    "sunscreen", "spf", "supplement", "vitamin", "mineral", "probiotic",
    "protein", "omega", "biotin", "collagen", "melatonin", "zinc", "magnesium",
    "cream", "lotion", "gel", "oil", "spray", "mask", "scrub", "balm",
    "toothpaste", "mouthwash", "deodorant", "razor", "shaving",
    # Common ingredients
    "retinol", "niacinamide", "hyaluronic", "aha", "bha", "glycolic",
    "salicylic", "vitamin c", "spf", "peptide", "ceramide",
    # Targets
    "acne", "dry skin", "oily skin", "sensitive skin", "hair loss",
    "wrinkle", "aging", "dandruff", "eczema", "psoriasis",
    # Health terms
    "pain relief", "sleep", "energy", "immune", "joint", "digestive",
    "weight loss", "muscle", "anxiety", "stress", "allergy",
    # Certifications
    "organic", "vegan", "cruelty-free", "paraben-free", "gluten-free",
    # Generic e-commerce health
    "product", "brand", "affordable", "cheap", "best", "recommended",
    "review", "rating", "buy", "ingredient", "formula",
])


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """Outcome of validating a single query."""
    is_valid:   bool
    reason:     str     # "passed" | "too_short" | "too_long" | "off_topic_structural" | "off_topic_semantic"
    confidence: float   # [0, 1] — how confident the guardrail is in the decision
    message:    str     # human-readable message for the demo UI


# ── Guardrail class ───────────────────────────────────────────────────────────

class QueryGuardrail:
    """
    Validates user queries before they reach the search engine.

    Three usage modes are exposed:
      - validate_structural() — only blacklist + length (no model)
      - validate_semantic()   — centroid similarity + FAISS top-1
      - validate()            — all three layers in sequence (recommended)

    Args:
        model:              already-loaded SentenceTransformer
        faiss_index:        already-built FAISS IndexFlatIP
        index_df:           DataFrame from embedding_index_enriched.csv
        combined_emb:       (n, 768) array from combined_embeddings.npy
        sim_threshold:      FAISS top-1 cosine threshold below which the query
                            is treated as off-topic (default 0.18, calibrated
                            empirically on this corpus).
        centroid_threshold: cosine threshold versus the corpus centroid
                            (default 0.05, very permissive — only catches
                            absurd queries).
    """

    def __init__(
        self,
        model,
        faiss_index,
        index_df: pd.DataFrame,
        combined_emb: np.ndarray,
        sim_threshold: float = 0.18,
        centroid_threshold: float = 0.05,
    ):
        self.model              = model
        self.faiss_index        = faiss_index
        self.index_df           = index_df
        self.combined_emb       = combined_emb
        self.sim_threshold      = sim_threshold
        self.centroid_threshold = centroid_threshold

        # Pre-compute the corpus centroid once
        self._corpus_centroid = self._compute_corpus_centroid()

    def _compute_corpus_centroid(self) -> np.ndarray:
        """L2-normalised mean of all combined embeddings."""
        centroid = self.combined_emb.mean(axis=0)
        norm     = np.linalg.norm(centroid)
        return (centroid / norm).astype(np.float32) if norm > 0 else centroid

    # ── Layer 1: Structural ───────────────────────────────────────────────

    def validate_structural(self, query: str) -> GuardrailResult | None:
        """
        O(1) structural checks: length and blacklist.
        Returns None if the query passes this layer (caller proceeds to L2).
        """
        q = query.strip()

        # Length
        words = q.split()
        if len(words) < 2:
            return GuardrailResult(
                is_valid=False,
                reason="too_short",
                confidence=0.99,
                message="The query is too short. Describe what you are looking for in more detail.",
            )
        if len(words) > 60:
            return GuardrailResult(
                is_valid=False,
                reason="too_long",
                confidence=0.90,
                message="The query is too long. Try to be more concise.",
            )

        q_lower = q.lower()

        # Blacklist patterns are checked BEFORE positive signals.
        # A positive signal (e.g. "best", "organic") cannot rescue a query
        # that clearly contains an off-topic term (e.g. "laptop", "gaming").
        for pattern in _BLACKLIST_PATTERNS:
            if pattern.search(q):
                return GuardrailResult(
                    is_valid=False,
                    reason="off_topic_structural",
                    confidence=0.92,
                    message=(
                        "Your query does not seem to be about health or personal "
                        "care products. Try searching for a specific product, "
                        "ingredient or goal (e.g. 'moisturizer for sensitive skin')."
                    ),
                )

        # Positive in-domain signals — if any are present after the blacklist
        # check, the query is almost certainly in-domain. Skip directly to the
        # semantic layer for the final confidence assessment.
        for signal in _POSITIVE_SIGNALS:
            if signal in q_lower:
                return None  # in-domain confirmed, proceed

        return None  # neither negative nor positive: let the semantic layer decide

    # ── Layers 2 & 3: Semantic ────────────────────────────────────────────

    def validate_semantic(self, query: str) -> GuardrailResult:
        """
        Embedding-based check:
          (L2) similarity versus the corpus centroid
          (L3) similarity of the FAISS top-1 retrieved product
        """
        # Encode
        q_vec = self.model.encode(
            [query.strip()], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        # Layer 2 — centroid similarity
        centroid_sim = float(np.dot(q_vec[0], self._corpus_centroid))
        if centroid_sim < self.centroid_threshold:
            return GuardrailResult(
                is_valid=False,
                reason="off_topic_semantic",
                confidence=round(1.0 - centroid_sim / self.centroid_threshold, 3),
                message=(
                    "Your search does not seem to match the Health & Personal Care "
                    "catalogue. Try a more specific term, e.g. 'vitamin C serum' "
                    "or 'shampoo for dry hair'."
                ),
            )

        # Layer 3 — FAISS top-1 confidence
        D, _ = self.faiss_index.search(q_vec, 1)
        top1_sim = float(D[0][0])

        if top1_sim < self.sim_threshold:
            return GuardrailResult(
                is_valid=False,
                reason="off_topic_semantic",
                confidence=round(1.0 - top1_sim / self.sim_threshold, 3),
                message=(
                    "We did not find any relevant product in the catalogue for "
                    "your query. Try rephrasing it with terms more specific to "
                    "Health & Personal Care."
                ),
            )

        # Passed
        return GuardrailResult(
            is_valid=True,
            reason="passed",
            confidence=round(min(top1_sim, 1.0), 3),
            message="",
        )

    # ── Main entry point ──────────────────────────────────────────────────

    def validate(self, query: str, verbose: bool = False) -> GuardrailResult:
        """
        Run all layers in sequence, short-circuiting on the first failure.
        The semantic layer is only invoked when the structural layer didn't
        already return a verdict — saving an embedding call on obvious cases.

        Args:
            query:   raw user text.
            verbose: if True, prints which layer made the decision.

        Returns:
            A GuardrailResult with is_valid, reason, confidence, message.
        """
        # Layer 1
        structural = self.validate_structural(query)
        if structural is not None:
            if verbose:
                print(f"[Guardrail L1] {structural.reason} (conf={structural.confidence})")
            return structural

        # Layers 2 + 3
        semantic = self.validate_semantic(query)
        if verbose:
            status = "VALID" if semantic.is_valid else "BLOCKED"
            print(f"[Guardrail L2/L3] {status} — {semantic.reason} (conf={semantic.confidence})")
        return semantic


# ── Convenience standalone function ──────────────────────────────────────────

def validate_query(
    query: str,
    model,
    faiss_index,
    index_df: pd.DataFrame,
    combined_emb: np.ndarray,
    sim_threshold: float = 0.18,
    verbose: bool = False,
) -> GuardrailResult:
    """
    Functional wrapper around QueryGuardrail — instantiates a temporary one.
    Handy for one-off checks in a notebook.

    Note: this rebuilds the corpus centroid on every call. For repeated use
    inside a loop, instantiate `QueryGuardrail` directly and reuse it.
    """
    g = QueryGuardrail(
        model=model,
        faiss_index=faiss_index,
        index_df=index_df,
        combined_emb=combined_emb,
        sim_threshold=sim_threshold,
    )
    return g.validate(query, verbose=verbose)
