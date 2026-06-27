import numpy as np

class FakeEmbedder:
    def __init__(self, dim=5):
        self.dim = dim

    def embed(self, texts):
        return np.random.rand(len(texts), self.dim).astype("float32")
