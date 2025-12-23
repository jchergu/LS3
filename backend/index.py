import numpy as np
import pandas as pd
import faiss


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame, use_faiss: bool = True):
        assert len(embeddings) == len(metadata), \
            "Embeddings and metadata must have same length"

        self.metadata = metadata.reset_index(drop=True)
        self.use_faiss = use_faiss

        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        faiss.normalize_L2(self.embeddings)

        if self.use_faiss:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)

    def search(self, query_vector: np.ndarray, top_k: int):
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        faiss.normalize_L2(query_vector)

        if self.use_faiss:
            assert query_vector.shape[1] == self.embeddings.shape[1], \
            "Query embedding dimension mismatch"

            scores, indices = self.index.search(query_vector, top_k)
            return indices[0], scores[0]       
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity(query_vector, self.embeddings)[0]
            top_idx = np.argsort(scores)[::-1][:top_k]
            return top_idx, scores[top_idx]
