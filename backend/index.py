# backend/index.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import EMBEDDINGS_PATH, METADATA_PATH

class VectorIndex:
    def __init__(self):
        self.embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        self.metadata = pd.read_parquet(METADATA_PATH)

        # Convert to float32 if needed (good practice)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

        # Normalize once
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def search(self, query_vector, top_k):
        query_vector = query_vector / np.linalg.norm(query_vector)

        scores = cosine_similarity(
            query_vector.reshape(1, -1),
            self.embeddings
        )[0]

        top_idx = scores.argsort()[::-1][:top_k]
        return top_idx, scores[top_idx]
