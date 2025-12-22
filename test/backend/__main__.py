from test.backend.test_loader import main as test_loader
from test.backend.test_index import main as test_index
from test.backend.test_service import main as test_service


def main():
    print("\n[backend tests]\n")

    test_loader()
    test_index()
    test_service()

    print("\n✅ all backend tests passed.\n")


if __name__ == "__main__":
    main()
