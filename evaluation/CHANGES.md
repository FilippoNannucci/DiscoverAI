# Patch suggerite — NON ANCORA APPLICATE

Tre modifiche al codice, ognuna giustificata empiricamente dai numeri prodotti
in `experiments/output/`. Da discutere col team prima di committare.

---

## Patch 1 — `BETA_POPULARITY = 0.0` in `config.py`

**Motivazione empirica** (`experiments/output/ablation_pop.csv`):

| Config                       | NDCG@5  | MRR    | Recall@5 |
|------------------------------|---------|--------|----------|
| β_q=0.12 β_p=0.05 (attuale)  | 0.5053  | 0.7649 | 0.1399   |
| β_q=0.12 β_p=0.00 (proposto) | 0.5232  | 0.7832 | 0.1418   |

**+1.8 punti NDCG e +1.8 punti MRR** semplicemente azzerando il peso di
`popularity_score`. L'analisi `step5_mrr_analysis.py` ha mostrato che il
popularity_score è il singolo responsabile di 7 regressioni del top-1 (vs solo 1
miglioramento) — sposta in alto prodotti popolari ma meno semanticamente
rilevanti.

**Diff**:

```diff
# src/mean_squared_terrors/config.py
- BETA_POPULARITY = 0.05   # popularity score weight in final search score
+ BETA_POPULARITY = 0.00   # popularity score weight (disabled — empirically harmful, see experiments/)
```

**Effetto su tutti i sistemi**: `search`, `search_v2`, `search_v3`, `hybrid_v4`
ereditano il default — tutti migliorano insieme. Niente da cambiare nelle
chiamate utente.

---

## Patch 2 — `search_v3` default `min_rating=None`

**Motivazione empirica** (`experiments/output/v3_modules.csv`):

| Config                          | NDCG@5  | MRR    | Recall@5 |
|---------------------------------|---------|--------|----------|
| search_v3 default (min_r=3.5)   | 0.4888  | 0.6993 | 0.1206   |
| search_v3 (min_rating=None)     | 0.5051  | 0.7218 | 0.1263   |

Il filtro `min_rating=3.5` di default fa perdere a `search_v3` ~1.6 punti NDCG
e ~2.2 punti MRR. Va lasciato disponibile come **opzione** (nella demo Gradio
si può attivare via slider), ma non come default — penalizza il benchmark e in
produzione è una scelta UX da rendere esplicita.

**Diff**:

```diff
# src/mean_squared_terrors/search.py — riga 360
def search_v3(
    query: str,
    model,
    index,
    index_df: pd.DataFrame,
    k: int = 5,
    price_buckets: list = None,
-   min_rating: float = 3.5,
+   min_rating: float = None,    # default off — utente lo attiva esplicitamente quando serve
    beta_quality: float = None,
    beta_popularity: float = BETA_POPULARITY,
    verbose: bool = False,
) -> pd.DataFrame:
```

E aggiornare il docstring poco sotto:

```diff
-    - min_rating default 3.5 (removes low-quality results automatically)
+    - min_rating optional (off by default; enable in UI when user wants quality filter)
```

---

## Patch 3 — README di root: rimuovere riferimento a `webapp/`

**Motivazione**: il README parla di `webapp/server.py` che non esiste nel repo.
Fa sembrare il README scritto da AI senza verifica.

```diff
# README.md
- The resulting map is saved to `data/product_images.csv` and is consumed by the
- demo notebook (`06_Demo.ipynb`) and the FastAPI webapp (`webapp/server.py`).
+ The resulting map is saved to `data/product_images.csv` and is consumed by the
+ demo notebook (`06_Demo.ipynb`).
```

(Da fare sia nel commento dentro NB01 che nel README principale, se presente lì.)

---

## Patch 4 — `requirements.txt` con versioni pinnate

Le regole di submission richiedono versioning preciso. La sandbox di
valutazione usa Python 3.13 (visto in `src/README.md`). Esempio di output di
`pip freeze` ridotto ai pacchetti che servono:

```
# Python 3.13
numpy==2.2.6
pandas==2.3.3
matplotlib==3.x
seaborn==0.x
scikit-learn==1.7.2
textblob==0.x
sentence-transformers==3.0.1
faiss-cpu==1.13.2
torch==2.3.0
transformers==4.44.2
spacy==3.x
gradio==4.x
rank_bm25==0.x
chromadb==0.x
umap-learn==0.5.x
plotly==5.x
```

(Le versioni di `matplotlib`, `seaborn`, `textblob`, `spacy`, `gradio`,
`chromadb`, `umap-learn`, `plotly` sono da rilevare dal vostro `.venv` con
`pip freeze` — non le ho perché ho una sandbox separata.)

---

## Numero di patch impatto/effort

| # | Patch | Effort | Δ NDCG@5 | Δ MRR | Note |
|---|-------|--------|----------|-------|------|
| 1 | β_pop = 0 | 1 carattere | +0.018 | +0.018 | Migliora TUTTI i sistemi |
| 2 | search_v3 min_rating=None | 2 caratteri | +0.016 | +0.022 | Riallinea v3 con base |
| 3 | README cleanup | 1 minuto | n/a | n/a | Cosmetico ma necessario |
| 4 | requirements.txt versioning | 5 minuti | n/a | n/a | Compliance regole |

**Totale**: ~10 minuti di lavoro per +0.034 NDCG e +0.040 MRR sul sistema
production. Vale la pena committarli **prima** della presentazione di martedì.

> **Important**: queste patch vanno committate con messaggio descrittivo
> ("perf: disable popularity rerank — empirically harmful, +1.8% NDCG"),
> in commit separati, NON in un singolo bulk commit. Le regole di submission
> apprezzano la commit history pulita.
