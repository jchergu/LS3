from pathlib import Path
import numpy as np
import pandas as pd


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise RuntimeError("Metadata file not found")

    return pd.read_parquet(metadata_path)

def load_embeddings(chunks_dir: Path) -> np.ndarray:
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        raise RuntimeError(f"Embeddings chunks directory not found: {chunks_dir}")
    chunks = sorted(chunks_dir.glob("emb_*.npy"))

    if not chunks:
        raise RuntimeError(f"No embedding chunks found in {chunks_dir}")

    return np.vstack([np.load(c) for c in chunks])

