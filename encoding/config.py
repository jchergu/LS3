from pathlib import Path

# model

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CHARS = 1500          
BATCH_SIZE = 64           
EMBEDDING_DTYPE = "float32"

# input data

DATA_DIR = Path("data")

PROCESSED_DATA_PATH = DATA_DIR / "processed" / "lyrics_final.csv"
TEXT_COLUMN = "lyrics"
ID_COLUMN = "id"

# output data

EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "minilm_1500chars"

EMBEDDINGS_PATH = EMBEDDINGS_DIR / "lyrics_embeddings.npy"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.parquet"
STATE_PATH = EMBEDDINGS_DIR / "state.json"

#runtime

CSV_CHUNK_SIZE = 10_000   
LOG_EVERY_N_BATCHES = 10 
