import sys
import pandas as pd

def load_data(file_path: str, source_type: str) -> pd.DataFrame:
    if source_type == "csv":
        return pd.read_csv(file_path)
    elif source_type == "json":
        return pd.read_json(file_path)
    elif source_type == "parquet":
        return pd.read_parquet(file_path)
    elif source_type == "api":
        import requests
        response = requests.get(file_path)
        if response.status_code != 200:
            sys.exit(f"[add_data] API request failed: {response.status_code}")
        return pd.read_json(response.text)
    else:
        sys.exit(f"[add_data] Unsupported source type: {source_type}")