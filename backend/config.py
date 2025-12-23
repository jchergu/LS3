from pathlib import Path

from encoding.config import EMBEDDING_MODEL_NAME as encoding_model_name

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings" / "minilm_1500chars" / "lyrics_embeddings.npy"
CHUNKS_DIR = EMBEDDINGS_DIR / "chunks"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.parquet"

# Embedding model
EMBEDDING_MODEL_NAME = encoding_model_name

# Search params
DEFAULT_TOP_K = 5
