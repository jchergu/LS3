import pandas as pd
import re
import unicodedata


def normalize(input_path: str, output_path: str, lyrics_col: str = "lyrics") -> pd.DataFrame:
    """
    Light normalization for lyrics:
    - Lowercase
    - Remove weird encoding artifacts (â€", â€‹, etc.)
    - Normalize unicode (decompose accents)
    - Normalize whitespace
    
    Preserves natural language (no stemming, lemmatization, or stopword removal)
    suitable for embedding models.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        lyrics_col: Name of the lyrics column (default: "lyrics")
    """
    print(f"[preprocessing/normalize] Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    def _normalize(text):
        if not isinstance(text, str):
            return ""
        
        # Remove encoding artifacts (mojibake: â€", â€‹, etc.)
        text = re.sub(r'â€[^\s]*', '', text)
        text = re.sub(r'â[^a-z]*', '', text)
        
        # Normalize unicode (NFD: decompose accents)
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        text = text.lower()
        
        # Normalize whitespace (multiple spaces/newlines -> single space)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    df[lyrics_col] = df[lyrics_col].apply(_normalize)
    
    df.to_csv(output_path, index=False)
    print(f"[preprocessing/normalize] Wrote {len(df)} rows to: {output_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Light normalize lyrics for embeddings")
    parser.add_argument("--input", "-i", default="data/processed/lyrics_clean.csv", help="Input CSV path")
    parser.add_argument("--output", "-o", default="data/processed/lyrics_normalized.csv", help="Output CSV path")
    parser.add_argument("--lyrics-col", default="lyrics", help="Name of lyrics column")
    args = parser.parse_args()
    normalize(args.input, args.output, args.lyrics_col)
