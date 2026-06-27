from pathlib import Path
import shutil
import numpy as np

from encoding.writer import ChunkedEmbeddingWriter

TEST_DIR = Path("test/_tmp_embeddings")


def main():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    writer = ChunkedEmbeddingWriter(
        base_dir=TEST_DIR,
        chunk_size=4,
        embedding_dim=3,
    )

    emb = np.ones((4, 3), dtype="float32")
    meta = [{"id": i} for i in range(4)]

    writer.write_chunk(0, emb, meta)

    chunk_path = TEST_DIR / "chunks" / "emb_00000.npy"
    assert chunk_path.exists(), "Chunk file not created"

    loaded = np.load(chunk_path)
    assert loaded.shape == (4, 3)

    print("✅ test_writer_chunks PASSED")


if __name__ == "__main__":
    main()
