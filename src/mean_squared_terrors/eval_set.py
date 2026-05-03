"""
Evaluation set per DiscoverAI — 38 query di test con pseudo-relevance judgments
basati su keyword matching su title + brand + features.

Schema per ogni query:
    qid             : id breve
    query           : testo esatto inviato al sistema
    intent          : 'semantic' | 'price' | 'negation' | 'dosage' | 'brand' | 'off_topic'
    must_keywords   : almeno UNA deve apparire in product_text per essere candidata
    bonus_keywords  : ognuna che appare aumenta il punteggio (relevance graduata)
    excluded_keywords : se appaiono → relevance = 0 (per query con negation)
    min_rating      : rating minimo per i prodotti "highly relevant" (qualità)

Le keyword sono in lowercase. Match case-insensitive con boundary di parola
(applicato dallo scorer in eval_metrics.py).

Pseudo-relevance scoring (4 livelli):
    0 → no must match  oppure  excluded match
    1 → must match, bonus ratio < 0.34
    2 → must match, bonus ratio ∈ [0.34, 0.67)
    3 → must match, bonus ratio ≥ 0.67  AND  product_avg_rating ≥ min_rating

Quando bonus_keywords è vuoto, si usano solo livelli 0 e 2 (rilevante / no).
"""

EVAL_QUERIES = [
    # ── Query semantiche pure ────────────────────────────────────────────────
    {"qid": "Q01", "query": "moisturizer for dry sensitive skin", "intent": "semantic",
     "must_keywords": ["moisturizer", "moisturiser", "moisturizing", "cream", "lotion", "hydrator"],
     "bonus_keywords": ["dry", "sensitive", "hydrating", "soothing", "fragrance free", "gentle"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q02", "query": "shampoo for color treated hair", "intent": "semantic",
     "must_keywords": ["shampoo"],
     "bonus_keywords": ["color", "colored", "color-treated", "color treated", "dyed", "colour", "sulfate free", "sulfate-free"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q03", "query": "anti aging face serum with retinol", "intent": "semantic",
     "must_keywords": ["serum", "anti-aging", "anti aging"],
     "bonus_keywords": ["retinol", "wrinkle", "fine lines", "aging", "rejuvenating"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q04", "query": "probiotic supplement for digestive health", "intent": "semantic",
     "must_keywords": ["probiotic", "probiotics"],
     "bonus_keywords": ["digestive", "digestion", "gut", "bloat", "ibs", "stomach"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q05", "query": "natural deodorant aluminum free", "intent": "semantic",
     "must_keywords": ["deodorant"],
     "bonus_keywords": ["natural", "aluminum free", "aluminum-free", "aluminium free", "organic"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q06", "query": "sunscreen for face spf 50", "intent": "semantic",
     "must_keywords": ["sunscreen", "sun screen", "sunblock", "spf"],
     "bonus_keywords": ["face", "facial", "spf 50", "spf50", "broad spectrum", "uv"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q07", "query": "biotin supplement for hair growth", "intent": "semantic",
     "must_keywords": ["biotin"],
     "bonus_keywords": ["hair", "growth", "nail", "skin", "vitamin", "supplement"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q08", "query": "electric toothbrush rechargeable", "intent": "semantic",
     "must_keywords": ["toothbrush", "tooth brush"],
     "bonus_keywords": ["electric", "rechargeable", "sonic", "battery", "brushing"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q09", "query": "collagen powder for skin and joints", "intent": "semantic",
     "must_keywords": ["collagen"],
     "bonus_keywords": ["powder", "skin", "joint", "joints", "peptide", "hydrolyzed", "hair"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q10", "query": "essential oil diffuser aromatherapy", "intent": "semantic",
     "must_keywords": ["diffuser", "essential oil", "essential oils", "aromatherapy"],
     "bonus_keywords": ["ultrasonic", "humidifier", "lavender", "peppermint", "mist"],
     "excluded_keywords": [], "min_rating": 4.0},

    # ── Query con intento di prezzo ──────────────────────────────────────────
    {"qid": "Q11", "query": "affordable face cream for daily use", "intent": "price",
     "must_keywords": ["cream", "moisturizer", "lotion"],
     "bonus_keywords": ["face", "facial", "daily", "everyday", "hydrating"],
     "excluded_keywords": [], "min_rating": 4.0,
     "price_constraint": "low_or_budget"},

    {"qid": "Q12", "query": "cheap toothpaste whitening", "intent": "price",
     "must_keywords": ["toothpaste"],
     "bonus_keywords": ["whitening", "white", "fluoride", "mint"],
     "excluded_keywords": [], "min_rating": 4.0,
     "price_constraint": "low_or_budget"},

    {"qid": "Q13", "query": "premium hair growth serum", "intent": "price",
     "must_keywords": ["serum", "treatment", "tonic"],
     "bonus_keywords": ["hair", "growth", "scalp", "regrowth", "thinning"],
     "excluded_keywords": [], "min_rating": 4.0,
     "price_constraint": "high_or_premium"},

    {"qid": "Q14", "query": "budget multivitamin daily", "intent": "price",
     "must_keywords": ["multivitamin", "multi vitamin", "daily vitamin"],
     "bonus_keywords": ["daily", "complete", "men", "women", "adult"],
     "excluded_keywords": [], "min_rating": 4.0,
     "price_constraint": "low_or_budget"},

    # ── Query con negation ───────────────────────────────────────────────────
    {"qid": "Q15", "query": "sleep aid without melatonin", "intent": "negation",
     "must_keywords": ["sleep", "insomnia", "rest"],
     "bonus_keywords": ["valerian", "magnesium", "chamomile", "natural", "calm", "relaxation"],
     "excluded_keywords": ["melatonin"], "min_rating": 4.0},

    {"qid": "Q16", "query": "shampoo without sulfate", "intent": "negation",
     "must_keywords": ["shampoo"],
     "bonus_keywords": ["sulfate free", "sulfate-free", "natural", "gentle", "color safe"],
     "excluded_keywords": ["sulfate", "sls"], "min_rating": 4.0},

    {"qid": "Q17", "query": "moisturizer fragrance free for sensitive skin", "intent": "negation",
     "must_keywords": ["moisturizer", "moisturiser", "cream", "lotion"],
     "bonus_keywords": ["sensitive", "fragrance free", "fragrance-free", "unscented", "hypoallergenic"],
     "excluded_keywords": ["fragrance", "perfumed", "scented"], "min_rating": 4.0},

    {"qid": "Q18", "query": "deodorant aluminum free unscented", "intent": "negation",
     "must_keywords": ["deodorant"],
     "bonus_keywords": ["aluminum free", "aluminum-free", "natural", "unscented", "fragrance free"],
     "excluded_keywords": ["aluminum chloride", "aluminium"], "min_rating": 4.0},

    {"qid": "Q19", "query": "vitamin c serum no parabens", "intent": "negation",
     "must_keywords": ["vitamin c", "vit c", "ascorbic"],
     "bonus_keywords": ["serum", "brightening", "antioxidant", "anti aging", "paraben free"],
     "excluded_keywords": ["paraben", "parabens"], "min_rating": 4.0},

    # ── Query con dosaggio ───────────────────────────────────────────────────
    {"qid": "Q20", "query": "vitamin d3 5000 iu", "intent": "dosage",
     "must_keywords": ["vitamin d", "vitamin d3", "d3", "cholecalciferol"],
     "bonus_keywords": ["5000", "5000iu", "5000 iu", "d3", "supplement"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q21", "query": "magnesium 400mg capsules", "intent": "dosage",
     "must_keywords": ["magnesium"],
     "bonus_keywords": ["400", "400mg", "400 mg", "capsule", "capsules", "citrate", "glycinate"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q22", "query": "fish oil omega 3 1000mg", "intent": "dosage",
     "must_keywords": ["fish oil", "omega 3", "omega-3"],
     "bonus_keywords": ["1000", "1000mg", "epa", "dha", "softgel"],
     "excluded_keywords": [], "min_rating": 4.0},

    # ── Query con problema/sintomo (linguaggio user) ─────────────────────────
    {"qid": "Q23", "query": "help me fall asleep at night", "intent": "semantic",
     "must_keywords": ["sleep", "insomnia", "melatonin", "valerian", "rest"],
     "bonus_keywords": ["aid", "supplement", "natural", "relax", "night"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q24", "query": "stop my hair from falling out", "intent": "semantic",
     "must_keywords": ["hair loss", "hair growth", "minoxidil", "biotin", "scalp"],
     "bonus_keywords": ["regrowth", "thinning", "fall", "thicker", "fuller"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q25", "query": "soothe my dry cracked hands", "intent": "semantic",
     "must_keywords": ["hand cream", "hand lotion", "moisturizer", "balm", "salve", "cream", "ointment"],
     "bonus_keywords": ["hand", "cracked", "dry", "healing", "repair", "intensive"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q26", "query": "relieve joint pain naturally", "intent": "semantic",
     "must_keywords": ["joint", "glucosamine", "turmeric", "msm", "chondroitin"],
     "bonus_keywords": ["pain", "relief", "natural", "supplement", "knee", "arthritis"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q27", "query": "boost my energy in the morning", "intent": "semantic",
     "must_keywords": ["energy", "caffeine", "b12", "guarana", "preworkout", "pre-workout"],
     "bonus_keywords": ["boost", "stamina", "vitality", "supplement", "performance"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q28", "query": "help me lose weight naturally", "intent": "semantic",
     "must_keywords": ["weight loss", "fat burner", "metabolism", "garcinia", "appetite"],
     "bonus_keywords": ["natural", "supplement", "diet", "burn", "slim", "thermogenic"],
     "excluded_keywords": [], "min_rating": 4.0},

    # ── Query brand-specific (brand presenti nel catalogo) ──────────────────
    {"qid": "Q29", "query": "gnc supplement vitamin", "intent": "brand",
     "must_keywords": ["gnc"],
     "bonus_keywords": ["supplement", "vitamin", "capsule", "tablet"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q30", "query": "glade air freshener home", "intent": "brand",
     "must_keywords": ["glade"],
     "bonus_keywords": ["air freshener", "fragrance", "spray", "home", "scent"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q31", "query": "dr scholls foot care insole", "intent": "brand",
     "must_keywords": ["scholl", "dr. scholl", "dr scholl"],
     "bonus_keywords": ["foot", "insole", "comfort", "shoe"],
     "excluded_keywords": [], "min_rating": 4.0},

    # ── Query specifiche / con uso ───────────────────────────────────────────
    {"qid": "Q32", "query": "razor blades for sensitive skin men", "intent": "semantic",
     "must_keywords": ["razor", "blade", "shaver", "shaving"],
     "bonus_keywords": ["sensitive", "men", "smooth", "close shave", "manual"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q33", "query": "best sunscreen for swimming and sports", "intent": "semantic",
     "must_keywords": ["sunscreen", "sunblock", "spf", "sun screen"],
     "bonus_keywords": ["water resistant", "waterproof", "sport", "swim", "active", "broad spectrum"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q34", "query": "scalp treatment for dandruff", "intent": "semantic",
     "must_keywords": ["dandruff", "scalp", "anti-dandruff", "anti dandruff"],
     "bonus_keywords": ["shampoo", "treatment", "ketoconazole", "zinc pyrithione", "tea tree"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q35", "query": "lip balm with spf protection", "intent": "semantic",
     "must_keywords": ["lip balm", "lip"],
     "bonus_keywords": ["spf", "sun", "uv", "protection", "moisturizing"],
     "excluded_keywords": [], "min_rating": 4.0},

    # ── Query ambigue / sfidanti ─────────────────────────────────────────────
    {"qid": "Q36", "query": "best supplement for women over 40", "intent": "semantic",
     "must_keywords": ["women", "multivitamin", "multi vitamin", "vitamin"],
     "bonus_keywords": ["women's", "menopause", "40", "50", "mature", "anti aging"],
     "excluded_keywords": [], "min_rating": 4.0},

    {"qid": "Q37", "query": "kids vitamin gummies daily", "intent": "semantic",
     "must_keywords": ["kids", "children", "child", "gummy", "gummies"],
     "bonus_keywords": ["vitamin", "multivitamin", "daily", "chewable", "kid"],
     "excluded_keywords": [], "min_rating": 4.0},

    # ── Query off-topic (per testare guardrail) ─────────────────────────────
    {"qid": "Q38", "query": "who won the champions league last year", "intent": "off_topic",
     "must_keywords": ["__NONE__"],   # nessun prodotto è rilevante
     "bonus_keywords": [],
     "excluded_keywords": [], "min_rating": 4.0},
]
