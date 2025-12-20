from pathlib import Path
import numpy as np
import pandas as pd


def load_embeddings(chunks_dir: Path) -> np.ndarray:
    chunks = sorted(chunks_dir.glob("emb_*.npy"))
    if not chunks:
        raise RuntimeError("No embedding chunks found")

    return np.vstack([np.load(c) for c in chunks])


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise RuntimeError("Metadata file not found")

    return pd.read_parquet(metadata_path)
