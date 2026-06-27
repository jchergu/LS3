import numpy as np

class FakeQueryEmbedder:
    def embed(self, text: str):
        # deterministic: always same vector
        v = np.zeros((1, 4), dtype="float32")
        v[0, 0] = 1.0
        return v
