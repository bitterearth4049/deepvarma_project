from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _fill_and_align(
    df_y: pd.DataFrame,
    df_x: Optional[pd.DataFrame],
    time_column: str,
    fill_method: str = "ffill",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df_y = df_y.sort_values(time_column).set_index(time_column)

    if df_x is not None:
        df_x = df_x.sort_values(time_column).set_index(time_column)
        # align on intersection of time indexes
        common_idx = df_y.index.intersection(df_x.index)
        df_y = df_y.loc[common_idx]
        df_x = df_x.loc[common_idx]

    # simple filling for gaps
    if fill_method is not None:
        if fill_method == "ffill":
            df_y = df_y.ffill().bfill()
            if df_x is not None:
                df_x = df_x.ffill().bfill()
        elif fill_method == "bfill":
            df_y = df_y.bfill().ffill()
            if df_x is not None:
                df_x = df_x.bfill().ffill()
        else:
            df_y = df_y.fillna(method=fill_method).fillna(method="bfill")
            if df_x is not None:
                df_x = df_x.fillna(method=fill_method).fillna(method="bfill")

    if df_x is not None:
        df_y = df_y.sort_index()
        df_x = df_x.sort_index()
    else:
        df_y = df_y.sort_index()

    if df_x is not None:
        return df_y, df_x
    return df_y, None


def prepare_single_series_tensors(
    df_y: pd.DataFrame,
    df_x: Optional[pd.DataFrame],
    time_column: str,
    fill_method: str = "ffill",
    standardize: bool = True,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    StandardScaler,
    Optional[StandardScaler],
    pd.Index,
]:
    """
    Convert aligned dataframes into numpy arrays of shape (T, dim).
    This treats the series as a single multivariate time series.
    """
    df_y_proc, df_x_proc = _fill_and_align(
        df_y, df_x, time_column=time_column, fill_method=fill_method
    )

    time_index = df_y_proc.index

    y_values = df_y_proc.values.astype("float32")
    x_values = df_x_proc.values.astype("float32") if df_x_proc is not None else None

    scaler_y = StandardScaler()
    scaler_x = StandardScaler() if x_values is not None else None

    if standardize:
        y_values = scaler_y.fit_transform(y_values)
        if x_values is not None:
            x_values = scaler_x.fit_transform(x_values)
    else:
        scaler_y.fit(y_values)
        if x_values is not None:
            scaler_x.fit(x_values)

    return y_values, x_values, scaler_y, scaler_x, time_index
