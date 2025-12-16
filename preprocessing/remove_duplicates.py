import pandas as pd


def remove_duplicates(input_path: str, output_path: str, subset=None):
    """
    Remove duplicate rows from CSV file.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        subset: Column(s) to consider for identifying duplicates (default: all columns)
    """
    print(f"[preprocessing/remove_duplicates] Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    initial_count = len(df)
    df_unique = df.drop_duplicates(subset=subset, keep='first')
    final_count = len(df_unique)
    removed_count = initial_count - final_count
    
    df_unique.to_csv(output_path, index=False)
    
    print(f"[preprocessing/remove_duplicates] Removed {removed_count} duplicate rows")
    print(f"[preprocessing/remove_duplicates] Wrote {final_count} rows to: {output_path}")
    return df_unique


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove duplicate rows from dataset")
    parser.add_argument("--input", "-i", default="data/processed/lyrics_en.csv", help="Input CSV path")
    parser.add_argument("--output", "-o", default="data/processed/lyrics_no_dup.csv", help="Output CSV path")
    parser.add_argument("--subset", "-s", nargs="+", help="Columns to consider for duplicates (default: all)")
    args = parser.parse_args()
    remove_duplicates(args.input, args.output, args.subset)
