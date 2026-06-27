---
title: LS3 Lyrics Similarity Search
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# LS3: Lyrics Similarity Search System

Search songs by meaning, not keywords.    

LS3 can receive a string-query from the user and perform a semantic-search of lyrics, thanks to Large Language Model embeddings and vector databases. Built as a proof-of-concept for intent-based music retrieval, relevant to streaming platforms and recommendation systems.

**Live demo:** [huggingface.co/spaces/joppejoppe/LS3](https://huggingface.co/spaces/joppejoppe/LS3)

Examples of queries and responses from the system:

```
"Happy songs about love"

1. "Is this happiness" - Lana Del Rey  
2. "Happy" - Pharrell Williams  
3. "PRIDE." - Kendrick Lamar 
```

```
"late night driving"

1. "Midnight city" - M83
2. "The 1975" - The 1975
3. "There is a light that never goes out" - The Smiths
```

Instead of matching keywords, LS3 understands the *meaning* of your query and finds songs with semantically similar lyrics.

## Tech stack

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dim vectors, optimized for semantic similarity
- **Vector search:** FAISS (IndexFlatIP with L2 normalization) — 10x faster than brute-force NumPy cosine similarity
- **Backend:** FastAPI + uvicorn
- **Dataset:** Kaggle Genius Song Lyrics (~2.4M songs, filtered to top 10k English tracks by popularity)
- **Deployment:** Docker on Hugging Face Spaces

## Architecture
query string-> QueryEmbedder (MiniLM) -> FAISS index (cosine similarity) -> metadata lookup → JSON response

**Preprocessing pipeline** (run once offline):
1. Language filter — English only
2. Sort by popularity (views) — top N songs
3. Select columns, deduplicate
4. Clean annotations, normalize, truncate lyrics

**Encoding pipeline** (run once offline):
- Chunked embedding with resumable state — safe to interrupt
- Outputs: `.npy` embedding chunks + `metadata.parquet`

**Backend** (always-on, lightweight):
- Loads precomputed embeddings into memory at startup
- Embeds query at runtime, searches FAISS index, returns top-k results
- Fully testable with mock data — no real embeddings needed

## Why FAISS over NumPy?

For 10k songs, FAISS `IndexFlatIP` is ~10x faster than brute-force cosine similarity. See `comparison/` for full benchmark results and latency plots.

## Run locally

```bash
git clone https://github.com/jchergu/LS3.git
cd LS3
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run preprocessing (requires Kaggle dataset)
python -m preprocessing.preprocess

# Run encoding
python -m encoding.run_embedding

# Start backend
uvicorn backend.app:app --reload
```

## Run with Docker

```bash
docker build -t ls3 .
docker run -p 7860:7860 ls3
```
