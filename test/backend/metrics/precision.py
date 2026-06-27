import pandas as pd

def precision_at_k(results, relevant_ids, k):
    if not isinstance(results, pd.DataFrame):
        raise TypeError("precision_at_k expects a pandas DataFrame")

    relevant = set(relevant_ids)

    # Take top-k rows
    retrieved_ids = results.head(k).index.tolist()
    
    # retrieved_ids = results.head(k)["id"].tolist()

    tp = sum(1 for rid in retrieved_ids if rid in relevant)
    return tp / k
