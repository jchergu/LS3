import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    print("[preprocessing/load_data] Loading raw data from:", path)
    return pd.read_csv(path)
