import time
import numpy as np
import pandas as pd
from pathlib import Path

from backend.loader import load_embeddings, load_metadata
from backend.index import VectorIndex
from backend.embedder import QueryEmbedder
from backend.config import CHUNKS_DIR, METADATA_PATH, EMBEDDING_MODEL_NAME

import faiss
faiss.omp_set_num_threads(1)


print("Loading data...")
embeddings = load_embeddings(CHUNKS_DIR)
metadata = load_metadata(METADATA_PATH)

print("Building indexes...")
index_numpy = VectorIndex(embeddings, metadata, use_faiss=False)
index_faiss = VectorIndex(embeddings, metadata, use_faiss=True)

embedder = QueryEmbedder(EMBEDDING_MODEL_NAME)


queries = Path("comparison/queries.txt").read_text().splitlines()

print("Warming up...")
dummy_q = embedder.embed("warmup query")
index_numpy.search(dummy_q, top_k=5)
index_faiss.search(dummy_q, top_k=5)


results = []
N_RUNS = 5
for q in queries:
    q_emb = embedder.embed(q)

    # NumPy
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        index_numpy.search(q_emb, top_k=5)
    t1 = time.perf_counter()

    # FAISS
    t2 = time.perf_counter()
    for _ in range(N_RUNS):
        index_faiss.search(q_emb, top_k=5)
    t3 = time.perf_counter()


    results.append({
        "query": q,
        "numpy_ms": (t1 - t0) * 1000 / N_RUNS,
        "faiss_ms": (t3 - t2) * 1000 / N_RUNS,
    })


df = pd.DataFrame(results)
Path("comparison").mkdir(exist_ok=True)
df.to_csv("comparison/results.csv", index=False)

print(df.describe())
print("Results saved to comparison/results.csv")


import matplotlib.pyplot as plt

df = pd.read_csv("comparison/results.csv")

plt.figure()
plt.plot(df["numpy_ms"], label="NumPy brute-force")
plt.plot(df["faiss_ms"], label="FAISS")
plt.xlabel("Query index")
plt.ylabel("Latency (ms)")
plt.legend()
plt.title("Query Latency Comparison")

Path("comparison/plots").mkdir(parents=True, exist_ok=True)
plt.savefig("comparison/plots/latency_comparison.png")
plt.close()

plt.figure()
plt.boxplot(
    [df["numpy_ms"], df["faiss_ms"]],
    labels=["NumPy", "FAISS"]
)
plt.ylabel("Latency (ms)")
plt.yscale("log")
plt.title("Latency Distribution")

plt.savefig("comparison/plots/latency_boxplot.png")
plt.close()
