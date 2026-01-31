from typing import List, Dict, Any

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from deepvarma.data import (
    load_hf_timeseries,
    prepare_single_series_tensors,
    blocked_time_series_folds,
)
from deepvarma.models.deepvarma_model import DeepVARMAHybrid, DeepVARMAConfig
from deepvarma.utils.seed import seed_everything
from deepvarma.utils.logging_utils import get_logger
from deepvarma.utils.metrics import compute_all

logger = get_logger(__name__)


def _build_deepvarma_config(cfg: DictConfig, m: int, n: int) -> DeepVARMAConfig:
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
    if not cfg.training.cv.enabled:
        logger.error(
            "CV is disabled in training.cv.enabled. Enable it or run train.py instead."
        )
        return

    seed_everything(cfg.training.seed)

    df_y, df_x = load_hf_timeseries(
        hf_id=cfg.dataset.hf_id,
        subset=cfg.dataset.subset,
        split=cfg.dataset.split,
        time_column=cfg.dataset.time_column,
        target_columns=list(cfg.dataset.target_columns),
        exogenous_columns=list(cfg.dataset.exogenous_columns),
        is_already_wide=cfg.dataset.is_already_wide,
    )

    y_all, x_all, scaler_y, scaler_x = prepare_single_series_tensors(
        df_y=df_y,
        df_x=df_x,
        time_column=cfg.dataset.time_column,
        standardize=cfg.dataset.standardize,
    )

    T, m = y_all.shape
    n = x_all.shape[1] if x_all is not None else 0
    logger.info(f"Loaded series with T={T}, m={m}, n={n}")

    folds = blocked_time_series_folds(
        T=T,
        n_splits=cfg.training.cv.n_splits,
        min_train_fraction=cfg.training.cv.min_train_fraction,
    )
    logger.info(f"Created {len(folds)} CV folds.")

    all_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_slice, val_slice) in enumerate(folds):
        logger.info(
            f"Fold {fold_idx + 1}/{len(folds)} train={train_slice} val={val_slice}"
        )
        y_train = y_all[train_slice]
        y_val = y_all[val_slice]
        x_train = x_all[train_slice] if x_all is not None else None
        x_val = x_all[val_slice] if x_all is not None else None

        dv_cfg = _build_deepvarma_config(cfg, m=m, n=n)
        model = DeepVARMAHybrid(dv_cfg)
        model.fit(y_train=y_train, x_train=x_train)

        horizon = min(cfg.model.forecast_horizon, y_val.shape[0])
        y_hist = y_train
        x_hist = x_train

        if x_val is not None:
            x_future = np.repeat(x_val[-1:], horizon, axis=0)
        else:
            x_future = None

        y_future_hat, _ = model.forecast(
            y_hist=y_hist,
            x_hist=x_hist,
            x_future=x_future,
            steps=horizon,
        )

        y_future_gt = y_val[:horizon]
        metrics = compute_all(y_future_gt, y_future_hat)
        logger.info(f"Fold {fold_idx + 1} metrics: {metrics}")
        all_metrics.append(metrics)

    # Aggregate metrics
    keys = all_metrics[0].keys()
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    std_metrics = {k: float(np.std([m[k] for m in all_metrics])) for k in keys}
    logger.info(f"CV mean metrics: {avg_metrics}")
    logger.info(f"CV std metrics: {std_metrics}")


if __name__ == "__main__":
    main()
