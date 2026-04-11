# Configuration — change only here, never inside functions

# ── Products ──────────────────────────────────────────────────────────────────
MIN_REVIEWS         = 10   # products with fewer reviews are excluded
CAP_PRODUCT_TOKENS  = 300  # token cap for product text (~390 subword tokens, safe for MPNet 512)
CAP_FEATURES_TOKENS = 150  # max tokens from features
CAP_DESC_TOKENS     = 150  # max tokens from description
CAP_DETAILS_TOKENS  = 80   # max tokens from semantic details

# ── Reviews ───────────────────────────────────────────────────────────────────
MIN_REVIEW_TOKENS = 25  # min tokens per review (removes "Love it!!", "Fast shipping", etc.)

# Balanced rating selection — reduces 5-star bias from ~70% raw → ~48%
REVIEWS_PER_RATING = {
    5: 5,  # max 5 five-star reviews (most helpful first)
    4: 3,  # max 3 four-star
    3: 2,  # max 2 three-star (neutral, low signal)
    2: 3,  # max 3 two-star
    1: 3,  # max 3 one-star (critical, important signal)
}

# ── Embedding ─────────────────────────────────────────────────────────────────
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 64

# Base review weights
W_HELPFUL = 0.7
W_RATING  = 0.3

# Dynamic alpha range [ALPHA_MIN, ALPHA_MAX]
# products with weak review signal  → high alpha (product emb dominates)
# products with strong review signal → low alpha (review emb matters more)
ALPHA_MIN = 0.35
ALPHA_MAX = 0.70

# ── Search & re-ranking ───────────────────────────────────────────────────────
BETA_QUALITY    = 0.12   # quality score weight in final search score
BETA_POPULARITY = 0.05   # popularity score weight in final search score
N_CANDIDATES    = 50     # FAISS candidates retrieved before re-ranking

# ── Details keys ──────────────────────────────────────────────────────────────
# Semantically meaningful keys from 'details' — excludes physical dims, dates, codes
SEMANTIC_DETAIL_KEYS = {
    'Item Form', 'Material', 'Material Feature', 'Special Feature',
    'Flavor', 'Dosage Form', 'Color', 'Size', 'Scent', 'Skin Type',
    'Active Ingredients', 'Specific Uses For Product', 'Product Benefits',
    'Compatible Devices', 'Power Source', 'Age Range (Description)',
    'Style', 'Recommended Uses For Product', 'Department', 'Occasion',
    'Target Audience', 'Hair Type', 'Skin Tone',
}

# Form keys go at the top of product text to immediately disambiguate category
FORM_KEYS = {'Item Form', 'Dosage Form', 'Material'}
