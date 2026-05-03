"""
Single-entry evaluation runner.

Reproduces the entire benchmark used in 08_Evaluation.ipynb. Reads the
artefacts produced by notebooks 01–04 (combined embeddings + FAISS index +
enriched index) and writes all metric tables to ``evaluation/output/``.

Usage (from repo root)::

    # produce the artefacts first by running notebooks 01 → 04, which write
    # to src/io/. Then:
    python evaluation/run_evaluation.py

The script is broken into clear stages so it can be re-executed end-to-end
or single-stage. Each stage prints a header with timings.
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

from mean_squared_terrors import search as team_search           # noqa: E402
from mean_squared_terrors import search_extended as team_extended  # noqa: E402
from eval_set import EVAL_QUERIES                                # noqa: E402
from eval_metrics import (                                       # noqa: E402
    build_qrels, ndcg_at_k, mrr, recall_at_k,
    precision_at_k, average_precision, aggregate_metrics,
)
from rank_bm25 import BM25Okapi                                  # noqa: E402

OUT = ROOT / "evaluation" / "output"
OUT.mkdir(exist_ok=True)
IO  = ROOT / "src" / "io"


def stage(name: str):
    print(f"\n[{name}] ", end="", flush=True)


def banner(text: str) -> None:
    print(f"\n{'─' * 78}\n{text}\n{'─' * 78}")


# ── 1. Load artefacts ─────────────────────────────────────────────────────────
banner("1. Loading artefacts")
t0 = time.time()
faiss_idx    = faiss.read_index(str(IO / "faiss_index.bin"))
combined_emb = np.load(IO / "combined_embeddings.npy")
idx_df       = pd.read_csv(IO / "embedding_index_enriched.csv")
products     = pd.read_csv(IO / "products_cleaned.csv")
catalog      = idx_df.merge(products[["parent_asin", "product_text_base"]],
                              on="parent_asin", how="left")
print(f"loaded in {time.time()-t0:.1f}s · catalog={len(catalog)} products · faiss={faiss_idx.ntotal}")

# ── 2. Build qrels (pseudo-relevance) ─────────────────────────────────────────
banner("2. Building pseudo-relevance qrels")
t0 = time.time()
qrels = build_qrels(catalog, EVAL_QUERIES)
print(f"qrels in {time.time()-t0:.1f}s · queries={len(qrels)}")

# ── 3. Build BM25 corpus ──────────────────────────────────────────────────────
banner("3. Building BM25 corpus")
t0 = time.time()
TOK = re.compile(r"\b[a-z0-9]{2,}\b")
def tok(s): return TOK.findall(str(s).lower())
bm25_corpus = [tok(t) for t in catalog["product_text_base"].fillna("")]
bm25 = BM25Okapi(bm25_corpus)
bm25_team = team_extended.build_bm25_index(catalog)
print(f"BM25 in {time.time()-t0:.1f}s")

# ── 4. Load model and encode queries ─────────────────────────────────────────
banner("4. Loading MPNet + encoding queries")
t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
queries_text = [q["query"] for q in EVAL_QUERIES]
q_embs = model.encode(queries_text, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float32)
print(f"MPNet+encode in {time.time()-t0:.1f}s")

# ── 5. Run all systems on all queries ────────────────────────────────────────
banner("5. Running 5 systems × 37 queries")
queries_to_eval = [(i, q) for i, q in enumerate(EVAL_QUERIES) if q["intent"] != "off_topic"]
K = 5

def fn_bm25(q):
    s = bm25.get_scores(tok(q["query"]))
    top = np.argsort(s)[::-1][:10]
    return catalog.iloc[top]["parent_asin"].tolist()

def fn_semantic(q):
    qv = q_embs[q["__qidx"]:q["__qidx"]+1]
    _, I = faiss_idx.search(qv, 10)
    return catalog.iloc[I[0]]["parent_asin"].tolist()

def fn_base(q):
    return team_search.search(query=q["query"], model=model, index=faiss_idx,
                               index_df=catalog, k=10, verbose=False)["parent_asin"].tolist()

def fn_v3(q):
    return team_search.search_v3(query=q["query"], model=model, index=faiss_idx,
                                  index_df=catalog, k=10, verbose=False)["parent_asin"].tolist()

def fn_v4(q):
    return team_extended.search_v4(query=q["query"], model=model, faiss_index=faiss_idx,
                                    bm25_index=bm25_team, index_df=catalog, k=10, verbose=False
                                    )["parent_asin"].tolist()

SYSTEMS = {"bm25": fn_bm25, "semantic": fn_semantic,
           "search_base": fn_base, "search_v3": fn_v3, "hybrid_v4": fn_v4}

results = {n: [] for n in SYSTEMS}
detailed = []
t0 = time.time()
for qidx, q in queries_to_eval:
    qrels_q = qrels[q["qid"]]
    if not qrels_q or all(v < 2 for v in qrels_q.values()):
        continue
    qx = {**q, "__qidx": qidx}
    for name, fn in SYSTEMS.items():
        try:
            ret = fn(qx)
        except Exception as e:
            print(f"  [!] {name} on {q['qid']}: {type(e).__name__}: {e}")
            ret = []
        n = ndcg_at_k(ret, qrels_q, k=K)
        m = mrr(ret, qrels_q, threshold=2)
        r = recall_at_k(ret, qrels_q, k=K, threshold=2)
        p = precision_at_k(ret, qrels_q, k=K, threshold=2)
        ap = average_precision(ret, qrels_q, threshold=2, k=10)
        results[name].append((q["qid"], n, m, r, p, ap))
        detailed.append({"qid": q["qid"], "intent": q["intent"], "system": name,
                         "query": q["query"], "ndcg5": n, "mrr": m, "recall5": r,
                         "prec5": p, "ap10": ap, "top1": ret[0] if ret else ""})
print(f"evaluation in {time.time()-t0:.1f}s")

# ── 6. Summary ────────────────────────────────────────────────────────────────
banner("6. Headline numbers (37 queries)")
print(f"{'System':<14}{'NDCG@5':>10}{'MRR':>10}{'Recall@5':>12}{'Prec@5':>10}{'MAP@10':>10}")
print("-" * 66)
agg_records = []
for name, rows in results.items():
    if not rows: continue
    a = aggregate_metrics(rows, k=K)
    a.update({"system": name, "n_queries": len(rows)})
    agg_records.append(a)
    print(f"{name:<14}{a['NDCG@5']:>10.4f}{a['MRR']:>10.4f}{a['Recall@5']:>12.4f}{a['Precision@5']:>10.4f}{a['MAP']:>10.4f}")

pd.DataFrame(agg_records)[["system","n_queries","NDCG@5","MRR","Recall@5","Precision@5","MAP"]
                           ].to_csv(OUT / "summary.csv", index=False)
pd.DataFrame(detailed).to_csv(OUT / "per_query.csv", index=False)

# ── 7. Per-intent breakdown ──────────────────────────────────────────────────
banner("7. Per-intent breakdown (NDCG@5)")
per_q = pd.DataFrame(detailed)
piv = per_q.pivot_table(index="intent", columns="system", values="ndcg5").round(3)
piv["best"] = piv.idxmax(axis=1)
print(piv.to_string())
piv.to_csv(OUT / "per_intent_ndcg.csv")

print(f"\n[done] all CSVs written to {OUT}")
