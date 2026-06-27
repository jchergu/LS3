import pandas as pd


def shorten_lyrics(input_path: str, output_path: str, max_chars: int = 5000, lyrics_col: str = "lyrics") -> pd.DataFrame:
    """
    Truncate lyrics to max_chars to fit embedding model token limits.
    
    Keeps first N characters to preserve song beginning context.
    Embedding models typically have:
    - Token limits (512–2048 tokens)
    - Performance degradation on very long texts
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        max_chars: Maximum characters per lyric (default: 5000)
        lyrics_col: Name of the lyrics column (default: "lyrics")
    """
    print(f"[preprocessing/shorten_lyrics] Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    initial_stats = df[lyrics_col].str.len().describe()
    
    df[lyrics_col] = df[lyrics_col].str[:max_chars]
    
    final_stats = df[lyrics_col].str.len().describe()
    
    df.to_csv(output_path, index=False)
    
    print(f"[preprocessing/shorten_lyrics] Truncated to max {max_chars} chars")
    print(f"[preprocessing/shorten_lyrics]   Before: mean={initial_stats['mean']:.0f}, max={initial_stats['max']:.0f}")
    print(f"[preprocessing/shorten_lyrics]   After:  mean={final_stats['mean']:.0f}, max={final_stats['max']:.0f}")
    print(f"[preprocessing/shorten_lyrics] Wrote {len(df)} rows to: {output_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shorten lyrics for embedding model token limits")
    parser.add_argument("--input", "-i", default="data/processed/lyrics_normalized.csv", help="Input CSV path")
    parser.add_argument("--output", "-o", default="data/processed/lyrics_final.csv", help="Output CSV path")
    parser.add_argument("--max-chars", type=int, default=5000, help="Max characters per lyric")
    parser.add_argument("--lyrics-col", default="lyrics", help="Name of lyrics column")
    args = parser.parse_args()
    shorten_lyrics(args.input, args.output, args.max_chars, args.lyrics_col)
