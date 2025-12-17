import json
from pathlib import Path
from typing import Optional


class EmbeddingState:
    """
    Handles progress tracking and resume logic for the embedding job.
    State is stored as a simple JSON file.
    """

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        """Check if a state file already exists."""
        return self.state_path.exists()

    def load_last_index(self) -> int:
        """
        Load the last processed row index.
        Returns 0 if no state exists.
        """
        if not self.exists():
            return 0

        with open(self.state_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return int(data.get("last_processed_row", 0))

    def save_last_index(self, row_index: int) -> None:
        """
        Persist the last successfully processed row index.
        """
        state = {
            "last_processed_row": int(row_index)
        }

        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def reset(self) -> None:
        """
        Delete the state file (use with caution).
        """
        if self.exists():
            self.state_path.unlink()
