# encoding/run_embedding.py

from encoding.config import (
    STATE_PATH,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    ID_COLUMN,
    TEXT_COLUMN,
    LOG_EVERY_N_BATCHES,
)

from encoding.embedder import Embedder
from encoding.state import EmbeddingState
from encoding.writer import EmbeddingWriter
from encoding.dataset_reader import read_dataset

import numpy as np


def main():
    print("[encoding] Starting embedding job")

    state = EmbeddingState(STATE_PATH)
    start_row = state.load_last_index()
    print(f"[encoding] Resuming from row: {start_row}")

    embedder = Embedder()

    # we need embedding dimension once
    dummy_vec = embedder.embed(["dummy"])
    embedding_dim = dummy_vec.shape[1]

    writer = EmbeddingWriter(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        embedding_dim=embedding_dim,
    )

    total_processed = start_row
    batch_count = 0

    for records in read_dataset(start_row=start_row):
        if not records:
            continue

        texts = [r[TEXT_COLUMN] for r in records]
        metadata = [{k: v for k, v in r.items() if k != TEXT_COLUMN} for r in records]

        # embed
        embeddings = embedder.embed(texts)

        # write
        writer.append_embeddings(embeddings)
        writer.append_metadata(metadata)

        # state
        total_processed += len(records)
        state.save_last_index(total_processed)

        batch_count += 1

        if batch_count % LOG_EVERY_N_BATCHES == 0:
            print(f"[encoding] Processed {total_processed} rows")

    print("[encoding] Embedding job completed successfully")
    print(f"[encoding] Total rows embedded: {total_processed}")


if __name__ == "__main__":
    main()
