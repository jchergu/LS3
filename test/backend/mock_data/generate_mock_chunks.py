from pathlib import Path
import numpy as np
import pandas as pd
import shutil


def generate_mock_chunks(
    base_dir: Path,
    n_chunks: int = 2,
    rows_per_chunk: int = 5,
    dim: int = 4,
):
    if base_dir.exists():
        shutil.rmtree(base_dir)

    chunks_dir = base_dir / "chunks"
    chunks_dir.mkdir(parents=True)

    metadata_rows = []
    global_id = 0

    for chunk_idx in range(n_chunks):
        embeddings = np.zeros((rows_per_chunk, dim), dtype="float32")

        for i in range(rows_per_chunk):
            embeddings[i, i % dim] = 1.0  # simple, deterministic vectors

            metadata_row = {
                "id": global_id,
                "title": f"song_{global_id}",
                "artist": f"artist_{global_id}",
            }
            metadata_rows.append(metadata_row)
            global_id += 1

        np.save(chunks_dir / f"emb_{chunk_idx:05d}.npy", embeddings)

    df = pd.DataFrame(metadata_rows)
    df.to_parquet(base_dir / "metadata.parquet", index=False)
