from pathlib import Path
import numpy as np

BASE_DIR = Path("test/_tmp_pipeline")
CHUNKS_DIR = BASE_DIR / "chunks"


def main():
    chunks = sorted(CHUNKS_DIR.glob("emb_*.npy"))
    assert chunks, "No chunks found"

    emb = np.vstack([np.load(c) for c in chunks])
    print("Loaded shape:", emb.shape)

    print("✅ test_chunk_loader PASSED")


if __name__ == "__main__":
    main()
