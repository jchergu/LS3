import pandas as pd

def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    print("[preprocessing/language_filter] Filtering for English language entries")
    return df[
        (df['language'] == 'en') &
        (df['language_cld3'] == 'en') &
        (df['language_ft'] == 'en')
    ].reset_index(drop=True)