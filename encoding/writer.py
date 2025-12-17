from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


class EmbeddingWriter:
    """
    Handles persistent storage of embeddings and metadata.
    Append-only and resume-safe.
    """

    def __init__(
        self,
        embeddings_path: Path,
        metadata_path: Path,
        embedding_dim: int,
        dtype: str = "float32",
    ):
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize files if they don't exist
        if not self.embeddings_path.exists():
            self._init_embeddings_file()

        if not self.metadata_path.exists():
            self._init_metadata_file()

    def _init_embeddings_file(self) -> None:
        """
        Create an empty memmap file for embeddings.
        """
        np.memmap(
            self.embeddings_path,
            dtype=self.dtype,
            mode="w+",
            shape=(0, self.embedding_dim),
        )

    def _init_metadata_file(self) -> None:
        """
        Create an empty metadata parquet file.
        """
        empty_df = pd.DataFrame()
        empty_df.to_parquet(self.metadata_path)

    def append_embeddings(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype(self.dtype)
        n_new, dim = embeddings.shape

        assert dim == self.embedding_dim, "Embedding dimension mismatch"

        dtype_size = np.dtype(self.dtype).itemsize

        if self.embeddings_path.exists():
            file_size = self.embeddings_path.stat().st_size
            current_rows = file_size // (dim * dtype_size)
        else:
            current_rows = 0

        new_total_rows = current_rows + n_new

        mmap = np.memmap(
            self.embeddings_path,
            dtype=self.dtype,
            mode="r+" if self.embeddings_path.exists() else "w+",
            shape=(new_total_rows, dim),
        )

        mmap[current_rows:new_total_rows] = embeddings
        mmap.flush()


    def append_metadata(self, rows: List[Dict[str, Any]]) -> None:
        """
        Append metadata rows to parquet file.
        """
        df_new = pd.DataFrame(rows)

        if self.metadata_path.exists():
            df_existing = pd.read_parquet(self.metadata_path)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_parquet(self.metadata_path, index=False)
