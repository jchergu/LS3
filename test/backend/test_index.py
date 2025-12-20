from pathlib import Path
import numpy as np

from backend.loader import load_embeddings, load_metadata
from backend.index import VectorIndex
from test.backend.mock_data.generate_mock_chunks import generate_mock_chunks


BASE_DIR = Path("test/backend/mock_data/data")


def main():
    generate_mock_chunks(BASE_DIR)

    emb = load_embeddings(BASE_DIR / "chunks")
    meta = load_metadata(BASE_DIR / "metadata.parquet")

    index = VectorIndex(emb, meta)

    query = emb[3]
    ids, scores = index.search(query, top_k=2)

    assert 3 in ids, f"Expected id 3 in results, got {ids}"
    assert scores[0] == 1.0

    print("✅ test_index PASSED")



if __name__ == "__main__":
    main()
