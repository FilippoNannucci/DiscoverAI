# How to run the code

Minimal guide to reproduce the DiscoverAI pipeline end-to-end.

## 1. Environment

- Python **3.13**
- Create and activate a virtual environment, then install dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
python -m spacy download en_core_web_sm
```

## 2. Input/output folder

All inputs are read from and all outputs are written to `src/io/`. This folder is **not tracked on git**: place the raw Amazon files here before running the pipeline.

Required raw inputs:

- `src/io/Health_and_Personal_Care.json` — raw reviews
- `src/io/meta_Health_and_Personal_Care.json` — raw product metadata

All intermediate and final artefacts (CSVs, `.npy` embeddings, FAISS index, summaries) are produced inside the same `src/io/` folder.

## 3. Source modules — `src/mean_squared_terrors/`

Reusable functions imported by the notebooks. Each module is self-contained.

| Module | Role |
|--------|------|
| `config.py` | All tunable parameters (model name, review caps, alpha range, beta weights, paths) |
| `cleaning.py` | Text normalization, balanced rating sampling, product text construction |
| `eda.py` | Plotting helpers used in NB02 |
| `embedding.py` | MPNet encoding, weighted review aggregation, dynamic alpha blending, FAISS index |
| `search.py` | `search`, `search_v2`, `search_v3`, query parsing, similar-products recommendation |
| `summarization.py` | BART-Large-CNN summarization, spaCy entity extraction, profile merge |

## 4. Execution order

Notebooks must be run in numerical order. Each one consumes the artefacts produced by the previous one (all stored in `src/io/`).

| # | Notebook | Reads from `io/` | Writes to `io/` |
|---|----------|------------------|------------------|
| 01 | `01_Data_Ingestion_Cleaning.ipynb` | `Health_and_Personal_Care.json`, `meta_Health_and_Personal_Care.json` | `products_cleaned.csv`, `reviews_cleaned.csv`, `reviews_topN.csv`, `product_catalogue.csv` |
| 02 | `02_EDA.ipynb` | `products_cleaned.csv`, `reviews_cleaned.csv` | (figures only) |
| 03 | `03_Embedding.ipynb` | `product_catalogue.csv`, `reviews_topN.csv` | `product_embeddings.npy`, `review_embeddings.npy`, `combined_embeddings.npy`, `faiss_index.bin`, `embedding_index.csv` |
| 04 | `04_Semantic_Search.ipynb` | `faiss_index.bin`, `embedding_index.csv`, `combined_embeddings.npy` | `embedding_index_enriched.csv` |
| 05 | `05_Summarization_EntityRecognition.ipynb` | `reviews_cleaned.csv`, `product_catalogue.csv`, `embedding_index_enriched.csv` | `product_summaries.csv`, `product_entities.csv`, `product_profiles.csv` |
| 06 | `06_Demo.ipynb` | `product_profiles.csv`, `faiss_index.bin`, `combined_embeddings.npy` | (Gradio app, no files) |

The notebooks live in the project root (one level above `src/`); from there each notebook reads/writes through `src/io/` via the paths defined in `config.py`.

## 5. Reproducibility notes

- Random seeds are fixed in `config.py` (`SEED = 42`); set globally at the top of each notebook.
- The embedding model (`sentence-transformers/all-mpnet-base-v2`) and the summarizer (`facebook/bart-large-cnn`) are downloaded automatically on first use and cached by HuggingFace.
- A CUDA GPU is used automatically if available; otherwise CPU fallback is transparent (NB03 and NB05 are the slow steps on CPU).
- Versions of all libraries used are pinned in `../requirements.txt`.
