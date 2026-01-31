from typing import Tuple, List, Optional

import pandas as pd
import time
import requests


def load_data_gov_timeseries(
    api_url: str,
    time_column: str,
    target_columns: List[str],
    exogenous_columns: Optional[List[str]] = None,
    date_format: Optional[str] = None,
    aggregate: Optional[str] = "mean",
    timeout_seconds: int = 60,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load a data.gov.in JSON API response into time series dataframes.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(api_url, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_retries:
                raise
            time.sleep(backoff_seconds * attempt)

    records = payload.get("records", [])
    if not records:
        raise ValueError("No records returned from data.gov.in API.")

    df = pd.DataFrame(records)

    if time_column not in df.columns:
        raise ValueError(
            f"time_column '{time_column}' not found in API response columns {df.columns.tolist()}"
        )

    missing_targets = [c for c in target_columns if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns missing in API response: {missing_targets}")

    exogenous_columns = exogenous_columns or []
    missing_exog = [c for c in exogenous_columns if c not in df.columns]
    if missing_exog:
        raise ValueError(f"Exogenous columns missing in API response: {missing_exog}")

    if date_format:
        df[time_column] = pd.to_datetime(
            df[time_column], format=date_format, errors="coerce"
        )
    else:
        df[time_column] = pd.to_datetime(df[time_column], dayfirst=True, errors="coerce")

    for col in target_columns + exogenous_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[time_column] + target_columns)
    df = df.sort_values(time_column)

    if aggregate is not None:
        agg_map = {col: aggregate for col in target_columns + exogenous_columns}
        df = df.groupby(time_column, as_index=False).agg(agg_map)

    df_y = df[[time_column] + target_columns].copy()
    df_x = df[[time_column] + exogenous_columns].copy() if exogenous_columns else None

    df_y = df_y.reset_index(drop=True)
    if df_x is not None:
        df_x = df_x.reset_index(drop=True)
    return df_y, df_x
