from backend.index import VectorIndex
import numpy as np

def main():
    idx = VectorIndex()

    dim = idx.embeddings.shape[1]
    fake_query = np.random.rand(dim)

    ids, scores = idx.search(fake_query, top_k=3)

    print("IDS:", ids)
    print("SCORES:", scores)
    print(idx.metadata.iloc[ids][["artist", "title"]])

if __name__ == "__main__":
    main()
