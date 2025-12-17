from os import Path

# Pipeline steps with output paths for tracking
STEPS = [
    ("filter_english", "data/processed/lyrics_en.csv"),
    ("select_columns", "data/processed/lyrics_selected.csv"),
    ("remove_duplicates", "data/processed/lyrics_no_dup.csv"),
    ("clean_lyrics", "data/processed/lyrics_clean.csv"),
    ("normalize", "data/processed/lyrics_normalized.csv"),
    ("shorten_lyrics", "data/processed/lyrics_final.csv"),
]

RAW_DATA_PATH = "data/raw/song_lyrics.csv"

MAX_LYRICS_LENGTH = 1500  # characters

COLUMNS = ["id", "title", "artist", "lyrics"]
