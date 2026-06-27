import pandas as pd

def sort_by_popularity(input_path: str, output_path: str, top_n: int, views_col: str = "views"):
    df = pd.read_csv(input_path)
    df = df.sort_values(by=views_col, ascending=False).head(top_n)
    df.to_csv(output_path, index=False)
    print(f"[sort_by_popularity] Kept top {top_n} songs by views → {output_path}")
