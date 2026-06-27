from pathlib import Path
import os
from encoding.config import EMBEDDING_MODEL_NAME as encoding_model_name

# Data paths — overridable via env vars for Docker/AWS
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", "data/embeddings/minilm_1500chars"))
CHUNKS_DIR = EMBEDDINGS_DIR / "chunks"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.parquet"

# Embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", encoding_model_name)

# Search params
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
