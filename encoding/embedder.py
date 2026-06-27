from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from encoding.config import (
    EMBEDDING_MODEL_NAME,
    BATCH_SIZE,
    EMBEDDING_DTYPE,
)


class Embedder:
    """
    Pure embedding logic
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into a 2D numpy array.

        Parameters
        ----------
        texts : List[str]
            List of input texts (already cleaned and truncated).

        Returns
        -------
        np.ndarray
            Shape: (len(texts), embedding_dim), dtype=float32
        """

        if not texts:
            return np.empty((0, 0), dtype=EMBEDDING_DTYPE)

        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-ready
        )

        return embeddings.astype(EMBEDDING_DTYPE)
