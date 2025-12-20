from sentence_transformers import SentenceTransformer

class QueryEmbedder:

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)
