from pathlib import Path
from backend.loader import load_embeddings, load_metadata
from test.backend.mock_data.generate_mock_chunks import generate_mock_chunks


BASE_DIR = Path("test/backend/mock_data/data")


def main():
    generate_mock_chunks(BASE_DIR)

    emb = load_embeddings(BASE_DIR / "chunks")
    meta = load_metadata(BASE_DIR / "metadata.parquet")

    assert emb.shape == (10, 4)
    assert len(meta) == 10

    print("✅ test_loader PASSED")


if __name__ == "__main__":
    main()
