from pathlib import Path
import numpy as np
import pandas as pd


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise RuntimeError("Metadata file not found")

    return pd.read_parquet(metadata_path)

def load_embeddings(chunks_dir: Path) -> np.ndarray:
    print("CHUNKS_DIR =", chunks_dir)
    print("EXISTS =", chunks_dir.exists())
    print("IS DIR =", chunks_dir.is_dir())

    files = list(chunks_dir.iterdir()) if chunks_dir.exists() else []
    print("FILES IN DIR =", files)

    chunks = sorted(chunks_dir.glob("emb_*.npy"))
    print("MATCHED CHUNKS =", chunks)

    if not chunks:
        raise RuntimeError(f"No embedding chunks found in {chunks_dir}")

    return np.vstack([np.load(c) for c in chunks])

