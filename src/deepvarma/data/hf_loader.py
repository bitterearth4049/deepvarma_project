from typing import Tuple, List

import pandas as pd
from datasets import load_dataset


def load_hf_timeseries(
    hf_id: str,
    subset: str,
    split: str,
    time_column: str,
    target_columns: List[str],
    exogenous_columns: List[str],
    is_already_wide: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a Hugging Face dataset and return two dataframes:
    - df_y: time, targets
    - df_x: time, exogenous

    This assumes the dataset can be represented as a single wide time series.
    You may need to customize mapping logic for specific datasets.
    """
    ds = load_dataset(hf_id, subset, split=split)

    df = ds.to_pandas()

    if time_column not in df.columns:
        raise ValueError(
            f"time_column '{time_column}' not found in dataset columns {df.columns.tolist()}"
        )

    missing_targets = [c for c in target_columns if c not in df.columns]
    missing_exog = [c for c in exogenous_columns if c not in df.columns]

    if missing_targets:
        raise ValueError(f"Target columns missing in HF dataset: {missing_targets}")
    if missing_exog:
        # exogenous can be empty; if you really want none, pass empty list
        raise ValueError(f"Exogenous columns missing in HF dataset: {missing_exog}")

    df = df.sort_values(time_column)
    df_y = df[[time_column] + target_columns].copy()
    df_x = df[[time_column] + exogenous_columns].copy()

    return df_y.reset_index(drop=True), df_x.reset_index(drop=True)
