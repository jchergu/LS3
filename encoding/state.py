import json
from pathlib import Path
from typing import Tuple


class EmbeddingState:
    def __init__(self, state_path: Path):
        self.state_path = state_path

    def load(self) -> Tuple[int, int]:
        """
        Load state from disk.
        Returns (last_row, chunk_index).
        """
        if not self.state_path.exists():
            return 0, 0

        with open(self.state_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return (
            int(data.get("last_row", 0)),
            int(data.get("chunk_index", 0)),
        )

    def save(self, last_row: int, chunk_index: int) -> None:
        """
        Persist state atomically.
        """
        tmp_path = self.state_path.with_suffix(".tmp")

        data = {
            "last_row": int(last_row),
            "chunk_index": int(chunk_index),
        }

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        tmp_path.replace(self.state_path)
