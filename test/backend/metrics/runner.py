from pathlib import Path
import pandas as pd

from backend.service import SearchService
from backend.index import VectorIndex
from backend.loader import load_embeddings
from backend.loader import load_metadata
from test.backend.fake_embedder import FakeQueryEmbedder

from .latency import measure_latency
from .precision import precision_at_k
from .memory import memory_usage_mb
from .plots import plot_latency, plot_precision


BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def _build_service() -> SearchService:
    """Internal helper: build SearchService from mock data."""

    mock_base = BASE_DIR.parent / "mock_data" / "data"

    embeddings = load_embeddings(mock_base / "chunks")
    metadata = load_metadata(mock_base / "metadata.parquet")

    index = VectorIndex(embeddings, metadata)
    embedder = FakeQueryEmbedder()

    return SearchService(index=index, embedder=embedder)


def run(queries: dict, relevance: dict) -> None:
    """Run full backend evaluation and persist results."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    service = _build_service()
    rows = []

    mem_before = memory_usage_mb()

    for name, query in queries.items():
        latency = measure_latency(service.search, query, runs=10)
        results = service.search(query)

        precision = precision_at_k(
            results,
            relevant_ids=relevance[name],
            k=5
        )

        rows.append({
            "query": name,
            "avg_latency_ms": latency["avg_ms"],
            "p95_latency_ms": latency["p95_ms"],
            "precision@5": precision,
        })

    mem_after = memory_usage_mb()

    df = pd.DataFrame(rows)

    # persist
    df.to_csv(TABLES_DIR / "evaluation_results.csv", index=False)
    plot_latency(df, FIGURES_DIR)
    plot_precision(df, FIGURES_DIR)

    # console output (for demos)
    print("=== Backend Evaluation Summary ===")
    print(df.to_string(index=False))
    print(f"\nMemory increase: {mem_after - mem_before:.2f} MB")
    print(f"Tables  -> {TABLES_DIR}")
    print(f"Figures -> {FIGURES_DIR}")
