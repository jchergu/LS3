from pathlib import Path

from backend.loader import load_embeddings, load_metadata
from backend.index import VectorIndex
from backend.service import SearchService
from test.backend.fake_embedder import FakeQueryEmbedder
from test.backend.mock_data.generate_mock_chunks import generate_mock_chunks


BASE_DIR = Path("test/backend/mock_data/data")


def main():
    generate_mock_chunks(BASE_DIR)

    emb = load_embeddings(BASE_DIR / "chunks")
    meta = load_metadata(BASE_DIR / "metadata.parquet")

    index = VectorIndex(emb, meta)
    embedder = FakeQueryEmbedder()
    service = SearchService(index, embedder)

    res = service.search("whatever", top_k=3)

    # basic sanity
    assert len(res) == 3
    assert "id" in res.columns
    assert "score" in res.columns

    # score ordering
    assert res["score"].is_monotonic_decreasing

    # best matches must include at least one known correct id
    valid_ids = {0, 4, 8}  # all vectors identical to fake query
    assert any(i in valid_ids for i in res["id"].values)

    print("✅ test_service PASSED")



if __name__ == "__main__":
    main()
