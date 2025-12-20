from backend.config import (
    CHUNKS_DIR,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
)

from backend.loader import load_embeddings, load_metadata
from backend.index import VectorIndex
from backend.embedder import QueryEmbedder
from backend.service import SearchService


def create_search_service() -> SearchService:
    embeddings = load_embeddings(CHUNKS_DIR)
    metadata = load_metadata(METADATA_PATH)

    index = VectorIndex(embeddings, metadata)
    embedder = QueryEmbedder(EMBEDDING_MODEL_NAME)

    return SearchService(index, embedder)
