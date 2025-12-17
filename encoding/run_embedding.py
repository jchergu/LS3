# encoding/run_embedding.py

from encoding.config import (
    STATE_PATH,
    EMBEDDINGS_PATH,
    ID_COLUMN,
    TEXT_COLUMN,
    LOG_EVERY_N_BATCHES,
)

from encoding.embedder import Embedder
from encoding.state import EmbeddingState
from encoding.writer import ChunkedEmbeddingWriter
from encoding.dataset_reader import read_dataset

import numpy as np


CHUNK_SIZE = 2048


def main():
    print("[encoding] Starting embedding job")

    state = EmbeddingState(STATE_PATH)
    start_row, chunk_index = state.load()
    print(f"[encoding] Resuming from row {start_row}, chunk {chunk_index}")

    embedder = Embedder()

    # determine embedding dimension once
    dummy_vec = embedder.embed(["dummy"])
    embedding_dim = dummy_vec.shape[1]

    writer = ChunkedEmbeddingWriter(
        base_dir=EMBEDDINGS_PATH,
        chunk_size=CHUNK_SIZE,
        embedding_dim=embedding_dim,
    )

    embeddings_buffer = []
    metadata_buffer = []

    total_processed = start_row
    batch_count = 0

    for records in read_dataset(start_row=start_row):
        if not records:
            continue

        texts = [r[TEXT_COLUMN] for r in records]
        metadata = [{k: v for k, v in r.items() if k != TEXT_COLUMN} for r in records]

        embeddings = embedder.embed(texts)

        embeddings_buffer.append(embeddings)
        metadata_buffer.extend(metadata)

        total_processed += len(records)
        batch_count += 1

        # flush chunk
        buffered_rows = sum(e.shape[0] for e in embeddings_buffer)
        if buffered_rows >= CHUNK_SIZE:
            chunk_embeddings = np.vstack(embeddings_buffer)

            writer.write_chunk(
                chunk_index=chunk_index,
                embeddings=chunk_embeddings,
                metadata_rows=metadata_buffer,
            )

            chunk_index += 1
            state.save(total_processed, chunk_index)

            embeddings_buffer.clear()
            metadata_buffer.clear()

        if batch_count % LOG_EVERY_N_BATCHES == 0:
            print(f"[encoding] Processed {total_processed} rows")

    # flush remaining buffers
    if embeddings_buffer:
        chunk_embeddings = np.vstack(embeddings_buffer)

        writer.write_chunk(
            chunk_index=chunk_index,
            embeddings=chunk_embeddings,
            metadata_rows=metadata_buffer,
        )

        state.save(total_processed, chunk_index + 1)

    print("[encoding] Embedding job completed successfully")
    print(f"[encoding] Total rows embedded: {total_processed}")


if __name__ == "__main__":
    main()
