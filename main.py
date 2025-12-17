import sys


def run_preprocessing():
    from preprocessing.preprocess import run_preprocessing
    run_preprocessing()


def run_encoding():
    from encoding.run_embedding import main
    main()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [preprocess | encode | all]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "preprocess":
        run_preprocessing()

    elif command == "encode":
        run_encoding()

    elif command == "all":
        run_preprocessing()
        run_encoding()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
