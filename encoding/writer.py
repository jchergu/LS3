from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


class ChunkedEmbeddingWriter:
    """
    Writes embeddings in immutable .npy chunks.
    Safe to interrupt and resume.
    """

    def __init__(
        self,
        base_dir: Path,
        chunk_size: int,
        embedding_dim: int,
        dtype: str = "float32",
    ):
        self.base_dir = base_dir
        self.chunk_size = chunk_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        self.chunks_dir = self.base_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.base_dir / "metadata.parquet"

    def write_chunk(
        self,
        chunk_index: int,
        embeddings: np.ndarray,
        metadata_rows: List[Dict[str, Any]],
    ) -> None:
        """
        Write one chunk atomically.
        """
        assert embeddings.shape[1] == self.embedding_dim

        embeddings = embeddings.astype(self.dtype)

        chunk_path = self.chunks_dir / f"emb_{chunk_index:05d}.npy"

        # embeddings
        tmp_path = chunk_path.with_suffix(".tmp.npy")
        np.save(tmp_path, embeddings)
        tmp_path.replace(chunk_path)

        # metadata
        df_new = pd.DataFrame(metadata_rows)

        if self.metadata_path.exists():
            df_old = pd.read_parquet(self.metadata_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_parquet(self.metadata_path, index=False)
