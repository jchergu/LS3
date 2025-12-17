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

from preprocessing.config import (
    STEPS,
    RAW_DATA_PATH,
    MAX_LYRICS_LENGTH,
    COLUMNS
    )
steps_dict = dict(STEPS)


def _step_done(output_path: str) -> bool:
    """Check if a step is already complete (output file exists)."""
    return os.path.exists(output_path)


def run_preprocessing():
    print("[preprocessing/preprocess] Starting preprocessing pipeline")
    print()

    # Step 1: Filter English
    if _step_done(steps_dict["filter_english"]):
        print("[preprocessing/preprocess] ✓ filter_english (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → filter_english (RUNNING)")
        df = load_raw_data(RAW_DATA_PATH)
        df = filter_english(df)
        del df

    # Step 2: Select columns
    if _step_done(steps_dict["select_columns"]):
        print("[preprocessing/preprocess] ✓ select_columns (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → select_columns (RUNNING)")
        select_columns(
            input_path=steps_dict["filter_english"],
            output_path=steps_dict["select_columns"],
            columns=COLUMNS
        )

    # Step 3: Remove duplicates
    if _step_done(steps_dict["remove_duplicates"]):
        print("[preprocessing/preprocess] ✓ remove_duplicates (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → remove_duplicates (RUNNING)")
        remove_duplicates(
            input_path=steps_dict["select_columns"],
            output_path=steps_dict["remove_duplicates"],
            subset=["lyrics"]
        )

    # Step 4: Clean lyrics
    if _step_done(steps_dict["clean_lyrics"]):
        print("[preprocessing/preprocess] ✓ clean_lyrics (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → clean_lyrics (RUNNING)")
        clean_lyrics(
            input_path=steps_dict["remove_duplicates"],
            output_path=steps_dict["clean_lyrics"],
            lyrics_col="lyrics"
        )

    # Step 5: Normalize
    if _step_done(steps_dict["normalize"]):
        print("[preprocessing/preprocess] ✓ normalize (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → normalize (RUNNING)")
        normalize(
            input_path=steps_dict["clean_lyrics"],
            output_path=steps_dict["normalize"],
            lyrics_col="lyrics"
        )

    # Step 6: Shorten lyrics
    if _step_done(steps_dict["shorten_lyrics"]):
        print("[preprocessing/preprocess] ✓ shorten_lyrics (SKIPPED — output exists)")
    else:
        print("[preprocessing/preprocess] → shorten_lyrics (RUNNING)")
        shorten_lyrics(
            input_path=steps_dict["normalize"],
            output_path=steps_dict["shorten_lyrics"],
            max_chars=MAX_LYRICS_LENGTH,
            lyrics_col="lyrics"
        )

    print()
    print("[preprocessing/preprocess] Pipeline complete")




if __name__ == "__main__":
    run_preprocessing()
