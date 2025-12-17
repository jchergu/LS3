import subprocess
import sys

TESTS = [
    "test.encoding.writer",
    "test.encoding.state",
    "test.encoding.fake_run",
]


def run(test_module: str):
    print(f"\n▶ Running {test_module}")
    result = subprocess.run(
        [sys.executable, "-m", test_module],
        check=False,
    )
    if result.returncode != 0:
        print(f"\n❌ FAILED: {test_module}")
        sys.exit(1)


def main():
    print("Running encoding test suite")
    for t in TESTS:
        run(t)
    print("\n✅ ALL encoding tests PASSED")


if __name__ == "__main__":
    main()
