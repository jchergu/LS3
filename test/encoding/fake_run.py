from pathlib import Path
import shutil

import numpy as np

from encoding.writer import ChunkedEmbeddingWriter
from encoding.state import EmbeddingState
from test.encoding.embedder import FakeEmbedder

BASE_DIR = Path("test/_tmp_pipeline")
STATE_PATH = BASE_DIR / "state.json"


def main():
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)

    BASE_DIR.mkdir(parents=True)

    embedder = FakeEmbedder(dim=5)
    writer = ChunkedEmbeddingWriter(
        base_dir=BASE_DIR,
        chunk_size=10,
        embedding_dim=5,
    )

    state = EmbeddingState(STATE_PATH)

    texts = [f"text_{i}" for i in range(35)]
    metadata = [{"id": i} for i in range(35)]

    buffer_e = []
    buffer_m = []

    last_row, chunk_idx = state.load()

    for i, t in enumerate(texts):
        buffer_e.append(embedder.embed([t]))
        buffer_m.append(metadata[i])

        if len(buffer_m) == 10:
            writer.write_chunk(
                chunk_idx,
                np.vstack(buffer_e),
                buffer_m,
            )
            chunk_idx += 1
            state.save(i + 1, chunk_idx)
            buffer_e.clear()
            buffer_m.clear()

    print("✅ test_fake_run PASSED")


if __name__ == "__main__":
    main()
