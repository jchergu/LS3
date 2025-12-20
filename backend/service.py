from backend.embedder import QueryEmbedder
from backend.index import VectorIndex
from backend.config import DEFAULT_TOP_K


class SearchService:
    def __init__(self, index: VectorIndex, embedder: QueryEmbedder):
        self.index = index
        self.embedder = embedder

    def search(self, query: str, top_k: int = DEFAULT_TOP_K):
        q_emb = self.embedder.embed(query)
        ids, scores = self.index.search(q_emb, top_k=top_k)

        return self.index.metadata.iloc[ids].assign(score=scores)
