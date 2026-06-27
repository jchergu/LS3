from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.config import CHUNKS_DIR, METADATA_PATH, EMBEDDING_MODEL_NAME
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

app = FastAPI(title="Lyrics Semantic Search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

search_service = create_search_service()

@app.get("/search")
def search(q: str = Query(...), top_k: int = 5):
    results = search_service.search(q, top_k)
    return results.to_dict(orient="records")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")
