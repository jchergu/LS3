from test.backend.__main__ import main as test_backend
from test.encoding.__main__ import main as test_encoding


def main():

    print("\nRunning all tests...\n")
    test_encoding()
    test_backend()

    print("All tests passed.\n")


if __name__ == "__main__":
    main()
