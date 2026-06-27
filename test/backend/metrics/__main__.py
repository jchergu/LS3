from .runner import run

QUERIES = {
    "loneliness": "songs about feeling alone and lost",
    "love": "romantic love songs",
    "anger": "angry aggressive lyrics",
}

# Must match metadata row indices
RELEVANCE = {
    "loneliness": [0, 2, 5],
    "love": [1, 3],
    "anger": [4, 6],
}

if __name__ == "__main__":
    run(QUERIES, RELEVANCE)
