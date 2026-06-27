from backend.loader import load_embeddings, load_metadata
from backend.index import VectorIndex
from backend.embedder import QueryEmbedder
from backend.config import CHUNKS_DIR, METADATA_PATH, EMBEDDING_MODEL_NAME, DEFAULT_TOP_K


def main():
    print("Loading data...")
    embeddings = load_embeddings(CHUNKS_DIR)
    metadata = load_metadata(METADATA_PATH)

    print("Building FAISS index...")
    index = VectorIndex(embeddings, metadata, use_faiss=True)
    embedder = QueryEmbedder(EMBEDDING_MODEL_NAME)

    print("\nSemantic search ready.")
    print("Type a query (or 'exit' to quit)\n")

    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Bye")
            break

        q_emb = embedder.embed(query)
        ids, scores = index.search(q_emb, top_k=DEFAULT_TOP_K)

        results = metadata.iloc[ids].assign(score=scores)

        print("\nResults:")
        for i, row in results.iterrows():
            print(f"- {row['artist'] if 'artist' in row else 'N/A'} | {row['title'] if 'title' in row else 'N/A'} | score={row['score']:.4f}")
        print()


if __name__ == "__main__":
    main()
