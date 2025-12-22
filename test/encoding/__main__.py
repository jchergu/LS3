import subprocess
import sys

TESTS = [
    "test.encoding.writer",
    "test.encoding.state",
    "test.encoding.fake_run",
]


def run(test_module: str):
    result = subprocess.run(
        [sys.executable, "-m", test_module],
        check=False,
    )
    if result.returncode != 0:
        print(f"\n❌ FAILED: {test_module}")
        sys.exit(1)


def main():
    print("[encoding tests]\n")
    for t in TESTS:
        run(t)
    print("\n✅ all encoding tests passed.\n")


if __name__ == "__main__":
    main()
