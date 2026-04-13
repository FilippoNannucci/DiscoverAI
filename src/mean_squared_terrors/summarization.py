"""Summarization and entity extraction utilities (notebook 05)."""

import os
import re
import subprocess
import sys
import time

import pandas as pd
import spacy
import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast


# ── Entity patterns ───────────────────────────────────────────────────────────
INGREDIENT_PATTERN = (
    r"\b(vitamin [a-z]\d{0,2}|hyaluronic acid|retinol|niacinamide|salicylic acid|"
    r"benzoyl peroxide|glycolic acid|collagen|keratin|biotin|caffeine|"
    r"aloe vera|tea tree|coconut oil|argan oil|jojoba|shea butter|"
    r"zinc oxide|titanium dioxide|spf\s*\d+|melatonin|magnesium|"
    r"probiotics?|prebiotics?|omega[- ]?3|glucosamine|turmeric|"
    r"activated charcoal|witch hazel|apple cider vinegar)\b"
)

CERT_PATTERN = (
    r"\b(organic|natural|vegan|cruelty.?free|paraben.?free|sulfate.?free|"
    r"fragrance.?free|hypoallergenic|dermatologist tested|fda|clinically tested|"
    r"non.?gmo|gluten.?free|alcohol.?free|oil.?free|non.?comedogenic)\b"
)

USE_PATTERNS = [
    r"\bfor (dry|oily|sensitive|combination|acne.?prone|mature|aging|normal) skin\b",
    r"\bfor (dry|damaged|fine|thinning|color.?treated|curly|straight) hair\b",
    r"\bfor (back|joint|muscle|knee|neck|shoulder) pain\b",
    r"\bfor (babies?|infants?|kids?|children|toddlers?)\b",
    r"\bfor (men|women|adults?|seniors?)\b",
]

# ── Summarization model ───────────────────────────────────────────────────────
BART_MODEL_NAME = "facebook/bart-large-cnn"


# ── Setup and data loading ────────────────────────────────────────────────────
def setup_notebook_dependencies():
    """Install optional dependencies and mount Google Drive on Colab."""
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        print("Google Drive mounted (Colab).")
    except ModuleNotFoundError:
        print("Running outside Colab: skipping Drive mount.")

    for pkg in ["gradio", "spacy"]:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

    try:
        spacy.load("en_core_web_sm")
        print("spaCy en_core_web_sm: already installed")
    except OSError:
        print("Downloading spaCy en_core_web_sm...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=False,
        )

    print("Dependencies ready.")


def get_device():
    """Return transformers-compatible device id: 0 for CUDA GPU else -1."""
    return 0 if torch.cuda.is_available() else -1


def load_clean_data(data_dir: str):
    """Load products and reviews cleaned datasets."""
    products = pd.read_csv(os.path.join(data_dir, "products_cleaned.csv"))
    reviews = pd.read_csv(os.path.join(data_dir, "reviews_cleaned.csv"))
    return products, reviews


# ── Entity extraction ─────────────────────────────────────────────────────────
def extract_entities(text: str, nlp_model, asin: str = "") -> dict:
    """Extract ingredients, certifications, use cases and brand entities from text."""
    if not text or pd.isna(text):
        return {"ingredients": [], "certifications": [], "use_cases": [], "brands": []}

    text_lower = str(text).lower()
    ing_matches = re.findall(INGREDIENT_PATTERN, text_lower, re.IGNORECASE)
    ingredients = list(set(m.lower().strip() for m in ing_matches))

    cert_matches = re.findall(CERT_PATTERN, text_lower, re.IGNORECASE)
    certs = list(set(m.lower().strip() for m in cert_matches))

    use_cases = []
    for pattern in USE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        use_cases.extend(
            [m.lower().strip() if isinstance(m, str) else " ".join(m).lower().strip() for m in matches]
        )
    use_cases = list(set(use_cases))

    doc = nlp_model(str(text)[:2000])
    brands = sorted({ent.text.strip() for ent in doc.ents if ent.label_ == "ORG" and len(ent.text) > 2})

    return {
        "ingredients": sorted(ingredients),
        "certifications": sorted(certs),
        "use_cases": sorted(use_cases),
        "brands": brands[:8],
    }


def build_entities_dataframe(products: pd.DataFrame, reviews: pd.DataFrame, nlp_model) -> pd.DataFrame:
    """Build per-product entity table from product text plus top review snippets."""
    print(f"Extracting entities from {len(products):,} products...")
    entity_records = []

    for i, (_, row) in enumerate(products.iterrows()):
        ents = extract_entities(row["product_text_base"], nlp_model, row["parent_asin"])
        rev_text = " ".join(
            reviews[reviews["parent_asin"] == row["parent_asin"]]["text"].fillna("").tolist()[:3]
        )
        rev_ents = extract_entities(rev_text[:1000], nlp_model)

        all_ings  = list(set(ents["ingredients"] + rev_ents["ingredients"]))
        all_certs = list(set(ents["certifications"] + rev_ents["certifications"]))
        all_uses  = list(set(ents["use_cases"] + rev_ents["use_cases"]))

        entity_records.append(
            {
                "parent_asin": row["parent_asin"],
                "product_title": row["product_title"],
                "ingredients": " | ".join(all_ings) if all_ings else "",
                "certifications": " | ".join(all_certs) if all_certs else "",
                "use_cases": " | ".join(all_uses) if all_uses else "",
                "brands_mentioned": " | ".join(ents["brands"]) if ents["brands"] else "",
                "n_ingredients": len(all_ings),
                "n_certifications": len(all_certs),
            }
        )

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,}/{len(products):,}")

    return pd.DataFrame(entity_records)


# ── Summarization ─────────────────────────────────────────────────────────────
def build_summarizer(device: int):
    """Load BART model/tokenizer and return a pipeline-compatible callable."""
    print("Loading BART summarization model...")
    print("(first run: downloads ~1.6GB — cached afterwards)")
    t0 = time.time()

    tokenizer = BartTokenizerFast.from_pretrained(BART_MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)
    if device == 0:
        model = model.cuda()

    def summarizer(text, max_length=80, min_length=20, do_sample=False, **kwargs):
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        if device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            num_beams=4,
            early_stopping=True,
        )
        out = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return [{"summary_text": out}]

    print(f"Model loaded in {time.time() - t0:.1f}s")
    return summarizer


def extract_key_sentences(texts, max_sents=5):
    """Extract informative, non-trivial and non-duplicated sentences."""
    all_sents = []
    for text in texts:
        if not text or pd.isna(text):
            continue
        sents = re.split(r"(?<=[.!?])\s+", str(text))
        for sentence in sents:
            sentence = sentence.strip()
            words = sentence.split()
            if len(words) >= 12 and not sentence.lower().startswith("i ") and len(sentence) < 300:
                all_sents.append(sentence)

    unique = []
    for sentence in all_sents:
        if not any(sentence[:40] in prev for prev in unique):
            unique.append(sentence)
    return unique[:max_sents]


def summarize_product(asin, prod_row, rev_df, summarizer, use_model=True):
    """Generate structured summary fields for one product."""
    prod_revs = rev_df[rev_df["parent_asin"] == asin]
    pos_revs = prod_revs[prod_revs["rating"] >= 4]["text"].dropna().tolist()
    neg_revs = prod_revs[prod_revs["rating"] <= 2]["text"].dropna().tolist()
    all_revs = prod_revs["text"].dropna().tolist()

    pos_sents = extract_key_sentences(pos_revs, max_sents=4)
    neg_sents = extract_key_sentences(neg_revs, max_sents=3)
    all_sents = extract_key_sentences(all_revs, max_sents=5)

    input_parts = []
    title = str(prod_row.get("product_title", ""))[:100]
    input_parts.append(f"Product: {title}.")
    if pos_sents:
        input_parts.append("Positive reviews: " + " ".join(pos_sents))
    if neg_sents:
        input_parts.append("Negative reviews: " + " ".join(neg_sents))
    if not pos_sents and not neg_sents and all_sents:
        input_parts.append("Customer feedback: " + " ".join(all_sents))

    model_input = " ".join(input_parts)[:800]
    if len(model_input.strip()) < 40:
        return {
            "parent_asin": asin,
            "summary_full": "",
            "pros": "",
            "cons": "",
            "best_for": "",
            "method": "empty",
        }

    method = "extractive"
    if use_model:
        try:
            summary = summarizer(
                model_input,
                max_length=95,
                min_length=30,
                do_sample=False,
            )[0]["summary_text"]
            method = "model"
        except Exception:
            summary = " ".join(all_sents[:3]) if all_sents else ""
    else:
        summary = " ".join(all_sents[:3]) if all_sents else ""

    pros = "; ".join(pos_sents[:2])
    cons = "; ".join(neg_sents[:2])
    best_for = ""
    if isinstance(prod_row.get("use_cases", ""), str) and prod_row.get("use_cases", "").strip():
        best_for = prod_row["use_cases"].split("|")[0].strip()

    return {
        "parent_asin": asin,
        "summary_full": summary.strip(),
        "pros": pros.strip(),
        "cons": cons.strip(),
        "best_for": best_for.strip(),
        "method": method,
    }


# ── Batch execution and merge ─────────────────────────────────────────────────
def run_batch_summarization(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    summarizer,
    data_dir: str,
    save_every: int = 500,
):
    """
    Batch summarization over all products.

    Uses BART for products with >=5 reviews, extractive fallback otherwise.
    Saves partial checkpoints every `save_every` items.
    """
    print("Starting batch summarization...")
    summaries = []
    t0 = time.time()

    for i, (_, prod_row) in enumerate(products.iterrows()):
        asin = prod_row["parent_asin"]
        n_reviews = len(reviews[reviews["parent_asin"] == asin])
        use_model = n_reviews >= 5
        result = summarize_product(asin, prod_row, reviews, summarizer, use_model=use_model)
        summaries.append(result)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            per_prod = elapsed / (i + 1)
            remaining = per_prod * (len(products) - i - 1)
            print(
                f"  {i + 1:,}/{len(products):,}  |  {elapsed/60:.1f} min elapsed"
                f"  |  ~{remaining/60:.0f} min remaining"
            )

        if (i + 1) % save_every == 0:
            df_partial = pd.DataFrame(summaries)
            df_partial.to_csv(os.path.join(data_dir, "product_summaries_partial.csv"), index=False)
            print(f"  [checkpoint saved: {len(summaries)} products]")

    summaries_df = pd.DataFrame(summaries)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} min")
    return summaries_df


def build_product_profiles(data_dir: str, summaries_df: pd.DataFrame) -> pd.DataFrame:
    """Merge embedding index with summaries and extracted entities."""
    idx = pd.read_csv(os.path.join(data_dir, "embedding_index_enriched.csv"))
    idx = idx.merge(
        summaries_df[["parent_asin", "summary_full", "pros", "cons", "best_for", "method"]],
        on="parent_asin",
        how="left",
    )

    ents = pd.read_csv(os.path.join(data_dir, "product_entities.csv"))
    idx = idx.merge(
        ents[
            [
                "parent_asin",
                "ingredients",
                "certifications",
                "use_cases",
                "n_ingredients",
                "n_certifications",
            ]
        ],
        on="parent_asin",
        how="left",
    )
    return idx
