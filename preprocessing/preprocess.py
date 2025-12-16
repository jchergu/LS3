import os
from preprocessing.load_data import load_raw_data
from preprocessing.language_filter import filter_english
from preprocessing.drop_columns import select_columns
from preprocessing.remove_duplicates import remove_duplicates
from preprocessing.clean_lyrics import clean_lyrics
from preprocessing.normalize import normalize
from preprocessing.shorten_lyrics import shorten_lyrics

"""
starting columns:

title	
tag	
artist	
year	
views	
features	
lyrics	
id	
language_cld3	
language_ft	
language
"""

# Pipeline steps with output paths for tracking
STEPS = [
    ("filter_english", "data/processed/lyrics_en.csv"),
    ("select_columns", "data/processed/lyrics_selected.csv"),
    ("remove_duplicates", "data/processed/lyrics_no_dup.csv"),
    ("clean_lyrics", "data/processed/lyrics_clean.csv"),
    ("normalize", "data/processed/lyrics_normalized.csv"),
    ("shorten_lyrics", "data/processed/lyrics_final.csv"),
]


def _step_done(output_path: str) -> bool:
    """Check if a step is already complete (output file exists)."""
    return os.path.exists(output_path)


def run_preprocessing():
    print("[preprocessing/preprocess] Starting preprocessing pipeline")
    print()

    # Step 1: Filter English
    if _step_done("data/processed/lyrics_en.csv"):
        print("[preprocessing/preprocess] ✓ filter_english (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → filter_english (RUNNING)")
        df = load_raw_data("data/raw/song_lyrics.csv")
        df = filter_english(df)
        del df

    # Step 2: Select columns
    if _step_done("data/processed/lyrics_selected.csv"):
        print("[preprocessing/preprocess] ✓ select_columns (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → select_columns (RUNNING)")
        select_columns(
            input_path="data/processed/lyrics_en.csv",
            output_path="data/processed/lyrics_selected.csv",
            columns=["id", "title", "artist", "lyrics"]
        )

    # Step 3: Remove duplicates
    if _step_done("data/processed/lyrics_no_dup.csv"):
        print("[preprocessing/preprocess] ✓ remove_duplicates (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → remove_duplicates (RUNNING)")
        remove_duplicates(
            input_path="data/processed/lyrics_selected.csv",
            output_path="data/processed/lyrics_no_dup.csv",
            subset=["lyrics"]
        )

    # Step 4: Clean lyrics
    if _step_done("data/processed/lyrics_clean.csv"):
        print("[preprocessing/preprocess] ✓ clean_lyrics (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → clean_lyrics (RUNNING)")
        clean_lyrics(
            input_path="data/processed/lyrics_no_dup.csv",
            output_path="data/processed/lyrics_clean.csv",
            lyrics_col="lyrics"
        )

    # Step 5: Normalize
    if _step_done("data/processed/lyrics_normalized.csv"):
        print("[preprocessing/preprocess] ✓ normalize (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → normalize (RUNNING)")
        normalize(
            input_path="data/processed/lyrics_clean.csv",
            output_path="data/processed/lyrics_normalized.csv",
            lyrics_col="lyrics"
        )

    # Step 6: Shorten lyrics
    if _step_done("data/processed/lyrics_final.csv"):
        print("[preprocessing/preprocess] ✓ shorten_lyrics (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → shorten_lyrics (RUNNING)")
        shorten_lyrics(
            input_path="data/processed/lyrics_normalized.csv",
            output_path="data/processed/lyrics_final.csv",
            max_chars=5000,
            lyrics_col="lyrics"
        )

    print()
    print("[preprocessing/preprocess] Pipeline complete")




if __name__ == "__main__":
    run_preprocessing()
