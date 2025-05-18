import random
import pandas as pd


def quick_dataset_summary(df: pd.DataFrame) -> dict:
    """Calculate basic stats for dataset summary and model selection.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with trip data.

    Returns
    -------
    dict
        Dictionary with stats like sample count, duplicate rate,
        and number of low-uniqueness columns.
    """
    n_samples, n_features = df.shape
    duplicate_rate = df.duplicated().mean()
    ratio_unique = df.nunique() / n_samples
    low_unique_cols = (ratio_unique < 0.01).sum()

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "duplicate_rate": duplicate_rate,
        "low_unique_cols": low_unique_cols,
    }


def auto_select_model(df_stats: dict, uniqueness: dict) -> str:
    """Heuristically select a model type based on dataset characteristics.

    Parameters
    ----------
    df_stats : dict
        Output from `quick_dataset_summary`, with dataset statistics.
    uniqueness : dict
        Per-column uniqueness metrics from `DataAnalyzer`.

    Returns
    -------
    str
        Selected model name (e.g., "RF", "Ridge", "Lasso", etc.).
    """
    n = df_stats["n_samples"]
    dup = df_stats["duplicate_rate"]
    low_unique = df_stats["low_unique_cols"]

    candidates: set[str] = set()

    if n < 1_000:
        candidates.update(["LR", "Ridge"])
    elif dup > 0.30 or low_unique > 2:
        candidates.update(["Ridge", "Lasso"])
    else:
        candidates.update(["RF", "Ridge", "DT"])

    mostly_unique_cols = [c for c, s in uniqueness.items() if s.get("value", 1.0) > 0.95]
    very_low_unique = [c for c, s in uniqueness.items() if s.get("value", 0.0) < 0.05]
    has_categorical = any(c in {"vendor_id", "store_and_fwd_flag"} for c in very_low_unique)

    if len(mostly_unique_cols) >= 3:
        candidates -= {"RF", "DT"}
        candidates.update(["Ridge", "Lasso"])

    if has_categorical and len(very_low_unique) > 2:
        candidates.update(["LR", "Ridge"])

    if not candidates:
        candidates.add("Ridge")

    chosen = random.choice(sorted(candidates))
    print(f"[auto_select_model] Candidates: {sorted(candidates)} → Chosen: {chosen}")
    return chosen
