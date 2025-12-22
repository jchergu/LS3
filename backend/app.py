from fastapi import FastAPI, Query

from backend.config import (
    CHUNKS_DIR,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
)

from backend.loader import load_embeddings, load_metadata
from backend.index import VectorIndex
from backend.embedder import QueryEmbedder
from backend.service import SearchService

from fastapi.middleware.cors import CORSMiddleware



def create_search_service() -> SearchService:
    embeddings = load_embeddings(CHUNKS_DIR)
    metadata = load_metadata(METADATA_PATH)

    index = VectorIndex(embeddings, metadata)
    embedder = QueryEmbedder(EMBEDDING_MODEL_NAME)

    return SearchService(index, embedder)


# FASTAPI APP
app = FastAPI(title="Lyrics Semantic Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create service ONCE at startup
search_service = create_search_service()


@app.get("/")
def root():
    return {"hello": "world"}


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = 5
):
    results = search_service.search(q, top_k)
    return results.to_dict(orient="records")
