from typing import Iterator, List, Dict, Any
import pandas as pd

from encoding.config import (
    PROCESSED_DATA_PATH,
    TEXT_COLUMN,
    ID_COLUMN,
    CSV_CHUNK_SIZE,
    MAX_CHARS,
)


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text to a maximum number of characters.
    """
    if not isinstance(text, str):
        return ""
    return text[:max_chars]


def read_dataset(
    start_row: int = 0,
) -> Iterator[List[Dict[str, Any]]]:
    """
    Stream the processed dataset in chunks, skipping rows before start_row.

    Yields
    ------
    List[Dict[str, Any]]
        Each item contains:
        - song_id
        - lyrics (truncated)
        - any other metadata columns present in the CSV
    """

    current_row = 0

    for chunk in pd.read_csv(
        PROCESSED_DATA_PATH,
        chunksize=CSV_CHUNK_SIZE,
    ):
        chunk_size = len(chunk)

        # If entire chunk is before start_row, skip it
        if current_row + chunk_size <= start_row:
            current_row += chunk_size
            continue

        # Otherwise, slice chunk to skip already processed rows
        if current_row < start_row:
            chunk = chunk.iloc[start_row - current_row :]

        records: List[Dict[str, Any]] = []

        for _, row in chunk.iterrows():
            lyrics = truncate_text(row.get(TEXT_COLUMN, ""), MAX_CHARS)

            record = row.to_dict()
            record[TEXT_COLUMN] = lyrics

            records.append(record)

        yield records

        current_row += chunk_size
