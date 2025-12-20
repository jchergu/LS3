import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        assert len(embeddings) == len(metadata), \
            "Embeddings and metadata must have same length"

        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        # normalize embeddings (safe)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

        self.metadata = metadata.reset_index(drop=True)

    def search(self, query_vector: np.ndarray, top_k: int):
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        # normalize query (safe)
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            raise ValueError("Query vector has zero norm")

        query_vector = query_vector / q_norm

        scores = cosine_similarity(query_vector, self.embeddings)[0]

        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]
