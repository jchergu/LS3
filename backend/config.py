from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings" / "minilm_1500chars"
CHUNKS_DIR = EMBEDDINGS_DIR / "chunks"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.parquet"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Search params
DEFAULT_TOP_K = 5
