
import pandas as pd
from typing import List

DEFAULT_COLUMNS = ["id", "title", "artist", "lyrics"]

def select_columns(input_path: str, output_path: str, columns: List[str] = None) -> pd.DataFrame:
	if columns is None:
		columns = DEFAULT_COLUMNS
	print(f"[preprocessing/drop_columns] Loading data from: {input_path}")
	df = pd.read_csv(input_path)
	missing = [c for c in columns if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns in input data: {missing}")
	df_sel = df[columns].copy()
	df_sel.to_csv(output_path, index=False)
	print(f"[preprocessing/drop_columns] Wrote {len(df_sel)} rows to: {output_path}")
	return df_sel


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Select subset of columns from processed dataset")
	parser.add_argument("--input", "-i", default="data/processed/lyrics_en.csv", help="Input CSV path")
	parser.add_argument("--output", "-o", default="data/processed/lyrics_selected.csv", help="Output CSV path")
	parser.add_argument("--cols", "-c", nargs="+", help="Columns to keep (default: id name artist lyrics)")
	args = parser.parse_args()
	select_columns(args.input, args.output, args.cols)

