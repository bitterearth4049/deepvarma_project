import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal
import hydra
from omegaconf import DictConfig, OmegaConf

from deepvarma.data import (
    load_hf_timeseries,
    load_data_gov_timeseries,
    prepare_single_series_tensors,
    TimeSeriesSplitConfig,
    simple_train_val_test_split,
)
from deepvarma.models.deepvarma_model import DeepVARMAHybrid, DeepVARMAConfig
from deepvarma.utils.seed import seed_everything
from deepvarma.utils.logging_utils import get_logger
from deepvarma.utils.metrics import compute_all

logger = get_logger(__name__)


def _build_deepvarma_config(cfg: DictConfig, m: int, n: int) -> DeepVARMAConfig:
    # Clone and fill dims
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    lstm_y_cfg = model_cfg.get("lstm_y")
    lstm_x_cfg = model_cfg.get("lstm_x")

    if lstm_y_cfg is not None:
        lstm_y_cfg["input_dim"] = m
        lstm_y_cfg["output_dim"] = m
    if lstm_x_cfg is not None:
        lstm_x_cfg["input_dim"] = n
        lstm_x_cfg["output_dim"] = n

    varma_cfg = model_cfg["varmax"]
    varma_cfg["use_exogenous"] = bool(varma_cfg.get("use_exogenous", False))
    varma_cfg["lags_exog"] = int(cfg.model.get("lags_exog", 0))

    dv_cfg = DeepVARMAConfig(
        variant=model_cfg["variant"],
        lstm_y_cfg=lstm_y_cfg,
        lstm_x_cfg=lstm_x_cfg,
        varma_cfg=varma_cfg,
        device=cfg.training.device,
        num_epochs_lstm_y=cfg.training.num_epochs_lstm_y,
        num_epochs_lstm_x=cfg.training.num_epochs_lstm_x,
        lr_lstm_y=cfg.training.lr_lstm_y,
        lr_lstm_x=cfg.training.lr_lstm_x,
        batch_size=cfg.training.batch_size,
        clip_grad_norm=cfg.training.clip_grad_norm,
        early_stopping=OmegaConf.to_container(
            cfg.training.early_stopping, resolve=True
        ),
    )
    return dv_cfg


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(cfg.training.seed)

    # 1) Load data
    if cfg.dataset.source == "data_gov":
        df_y, df_x = load_data_gov_timeseries(
            api_url=cfg.dataset.api_url,
            time_column=cfg.dataset.time_column,
            target_columns=list(cfg.dataset.target_columns),
            exogenous_columns=list(cfg.dataset.exogenous_columns),
            date_format=getattr(cfg.dataset, "date_format", None),
            aggregate=getattr(cfg.dataset, "aggregate", "mean"),
        )
    elif cfg.dataset.source == "hf":
        df_y, df_x = load_hf_timeseries(
            hf_id=cfg.dataset.hf_id,
            subset=cfg.dataset.subset,
            split=cfg.dataset.split,
            time_column=cfg.dataset.time_column,
            target_columns=list(cfg.dataset.target_columns),
            exogenous_columns=list(cfg.dataset.exogenous_columns),
            is_already_wide=cfg.dataset.is_already_wide,
        )
    else:
        raise ValueError(f"Unknown dataset source '{cfg.dataset.source}'.")

    # 2) Prepare tensors
    y_all, x_all, scaler_y, scaler_x, time_index = prepare_single_series_tensors(
        df_y=df_y,
        df_x=df_x,
        time_column=cfg.dataset.time_column,
        fill_method=cfg.dataset.fill_method,
        standardize=cfg.dataset.standardize,
    )

    T, m = y_all.shape
    n = x_all.shape[1] if x_all is not None else 0
    logger.info(f"Loaded series with T={T}, m={m}, n={n}")

    split_cfg = TimeSeriesSplitConfig(
        train_fraction=cfg.training.train_fraction,
        val_fraction=cfg.training.val_fraction,
    )
    train_slice, val_slice, test_slice = simple_train_val_test_split(T, split_cfg)

    y_train = y_all[train_slice]
    y_val = y_all[val_slice]
    y_test = y_all[test_slice]

    x_train = x_all[train_slice] if x_all is not None else None
    x_val = x_all[val_slice] if x_all is not None else None
    x_test = x_all[test_slice] if x_all is not None else None

    # 3) Build DeepVARMA config and model
    dv_cfg = _build_deepvarma_config(cfg, m=m, n=n)
    model = DeepVARMAHybrid(dv_cfg)

    # 4) Fit on train
    model.fit(y_train=y_train, x_train=x_train)

    # 4b) In-sample fitted plot (train)
    y_train_hat_in, _ = model.in_sample_predict(y_train, x_train)
    valid_mask = ~np.isnan(y_train_hat_in).any(axis=1)
    if valid_mask.any():
        dates_train = time_index[train_slice][valid_mask]
        y_train_inv = scaler_y.inverse_transform(y_train[valid_mask])
        y_train_hat_inv = scaler_y.inverse_transform(y_train_hat_in[valid_mask])

        os.makedirs("artifacts", exist_ok=True)
        plot_df = pd.DataFrame(
            {
                "Date": dates_train,
                "Actual": y_train_inv[:, 0],
                "In-sample fitted": y_train_hat_inv[:, 0],
            }
        )
        plot_df = plot_df.melt(
            id_vars=["Date"], var_name="Series", value_name="Modal Price"
        )

        p = (
            ggplot(plot_df, aes(x="Date", y="Modal Price", color="Series"))
            + geom_line(size=1.0)
            + labs(title="In-sample fit (train)", x="Date", y="Modal Price")
            + theme_minimal()
        )
        p.save("artifacts/in_sample_fit.png", width=10, height=4, dpi=150)

    # 5) Evaluate on validation and test
    horizon = cfg.model.forecast_horizon

    # For val: we forecast horizon steps after end of train using val exogenous
    y_hist_val = np.concatenate([y_train, y_val], axis=0)
    x_hist_val = (
        np.concatenate([x_train, x_val], axis=0) if x_train is not None else None
    )

    if x_val is not None:
        # For future horizon we use last known exogenous repeated
        x_future_val = np.repeat(x_val[-1:], horizon, axis=0)
    else:
        x_future_val = None

    y_val_future_hat, _ = model.forecast(
        y_hist=y_hist_val,
        x_hist=x_hist_val,
        x_future=x_future_val,
        steps=horizon,
    )

    # Align ground truth for metrics
    y_val_slice = y_all[val_slice]
    val_horizon = min(horizon, y_val_slice.shape[0])
    y_val_future_gt = y_val_slice[:val_horizon]
    y_val_future_hat = y_val_future_hat[:val_horizon]
    val_metrics = compute_all(y_val_future_gt, y_val_future_hat)
    logger.info(f"Validation metrics (first {horizon} steps): {val_metrics}")

    # For test
    y_hist_test = np.concatenate([y_train, y_val, y_test], axis=0)
    x_hist_test = (
        np.concatenate([x_train, x_val, x_test], axis=0)
        if x_train is not None
        else None
    )
    if x_test is not None:
        x_future_test = np.repeat(x_test[-1:], horizon, axis=0)
    else:
        x_future_test = None

    y_test_future_hat, _ = model.forecast(
        y_hist=y_hist_test,
        x_hist=x_hist_test,
        x_future=x_future_test,
        steps=horizon,
    )
    y_test_slice = y_all[test_slice]
    test_horizon = min(horizon, y_test_slice.shape[0])
    y_test_future_gt = y_test_slice[:test_horizon]
    y_test_future_hat = y_test_future_hat[:test_horizon]
    test_metrics = compute_all(y_test_future_gt, y_test_future_hat)
    logger.info(f"Test metrics (first {horizon} steps): {test_metrics}")

    # Optionally save scalers and predictions
    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/y_test_future_hat.npy", y_test_future_hat)
    np.save("artifacts/y_test_future_gt.npy", y_test_future_gt)


if __name__ == "__main__":
    main()
