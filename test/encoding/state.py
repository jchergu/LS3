from pathlib import Path
import shutil

from encoding.state import EmbeddingState

STATE_PATH = Path("test/_tmp_state.json")


def main():
    if STATE_PATH.exists():
        STATE_PATH.unlink()

    state = EmbeddingState(STATE_PATH)

    r, c = state.load()
    assert r == 0 and c == 0

    state.save(123, 7)

    r, c = state.load()
    assert r == 123
    assert c == 7

    print("✅ test_state_resume PASSED")


if __name__ == "__main__":
    main()
