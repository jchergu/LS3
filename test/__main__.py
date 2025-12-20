from backend.app import create_search_service

def main():
    service = create_search_service()
    res = service.search("sad song about love", top_k=3)
    print(res)

if __name__ == "__main__":
    main()
