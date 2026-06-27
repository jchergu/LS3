import pandas as pd
import re


def clean_lyrics(input_path: str, output_path: str, lyrics_col: str = "lyrics") -> pd.DataFrame:
    """
    Clean lyrics by removing extra whitespace, line breaks, and special characters.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        lyrics_col: Name of the lyrics column (default: "lyrics")
    """
    print(f"[preprocessing/clean_lyrics] Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    def _clean(text):
        if not isinstance(text, str):
            return ""
        # Remove extra whitespace and line breaks
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\'\"\-\.\,\!\?]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    df[lyrics_col] = df[lyrics_col].apply(_clean)
    # Remove empty lyrics
    df = df[df[lyrics_col].str.len() > 0].copy()
    
    df.to_csv(output_path, index=False)
    print(f"[preprocessing/clean_lyrics] Wrote {len(df)} rows to: {output_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean lyrics text")
    parser.add_argument("--input", "-i", default="data/processed/lyrics_no_dup.csv", help="Input CSV path")
    parser.add_argument("--output", "-o", default="data/processed/lyrics_clean.csv", help="Output CSV path")
    parser.add_argument("--lyrics-col", default="lyrics", help="Name of lyrics column")
    args = parser.parse_args()
    clean_lyrics(args.input, args.output, args.lyrics_col)
