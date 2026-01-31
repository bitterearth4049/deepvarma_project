from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from .lstm_modules import LSTMPredictor, LSTMEncoder
from .varma_layers import VARMALayer, VARMALayerConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DeepVARMAConfig:
    variant: str  # "re", "en", "full"
    lstm_y_cfg: Optional[Dict[str, Any]]
    lstm_x_cfg: Optional[Dict[str, Any]]
    varma_cfg: Dict[str, Any]
    device: str = "cpu"
    num_epochs_lstm_y: int = 50
    num_epochs_lstm_x: int = 30
    lr_lstm_y: float = 1e-3
    lr_lstm_x: float = 1e-3
    batch_size: int = 1
    clip_grad_norm: float = 1.0
    early_stopping: Dict[str, Any] = None


class DeepVARMAHybrid:
    """
    Implements:
    - DeepVARMA-re
        1. LSTM on y_t to get trend mu_t.
        2. Residual e_t = y_t - mu_t.
        3. VARMA on e_t.
        4. Forecast y_hat_t = mu_t + e_hat_t.

    - DeepVARMA-en
        1. LSTM encoder on exogenous x_t to get H_t.
        2. VARMAX on y_t with exogenous H_t.

    - DeepVARMA (full)
        1. LSTM on y_t to get mu_t, residual e_t.
        2. LSTM encoder on x_t to get H_t.
        3. VARMAX on residual e_t with exogenous H_t.
        4. Forecast y_hat_t = mu_t + e_hat_t.
    """

    def __init__(self, cfg: DeepVARMAConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        variant = cfg.variant.lower()
        if variant not in {"re", "en", "full"}:
            raise ValueError(
                f"Unknown DeepVARMA variant '{cfg.variant}', must be 're', 'en', or 'full'."
            )
        self.variant = variant

        self.lstm_y: Optional[LSTMPredictor] = None
        self.lstm_x: Optional[LSTMEncoder] = None
        self.varma: Optional[VARMALayer] = None

    def _build_lstm_y(self, input_dim: int, output_dim: int) -> None:
        if self.cfg.lstm_y_cfg is None:
            raise ValueError("lstm_y_cfg is None but variant requires LSTM on y.")
        lstm_cfg = self.cfg.lstm_y_cfg.copy()
        lstm_cfg["input_dim"] = input_dim
        lstm_cfg["output_dim"] = output_dim
        self.lstm_y = LSTMPredictor(**lstm_cfg).to(self.device)

    def _build_lstm_x(self, input_dim: int, output_dim: int = None) -> None:
        if self.cfg.lstm_x_cfg is None:
            raise ValueError("lstm_x_cfg is None but variant requires LSTM on x.")
        lstm_cfg = self.cfg.lstm_x_cfg.copy()
        lstm_cfg["input_dim"] = input_dim
        lstm_cfg["output_dim"] = output_dim
        self.lstm_x = LSTMEncoder(**lstm_cfg).to(self.device)

    def _build_varma(self) -> None:
        varma_cfg = VARMALayerConfig(**self.cfg.varma_cfg)
        self.varma = VARMALayer(varma_cfg)

    def _build_trainer(self, num_epochs: int) -> pl.Trainer:
        callbacks = []
        if self.cfg.early_stopping and self.cfg.early_stopping.get("enabled", False):
            callbacks.append(
                EarlyStopping(
                    monitor="train_loss",
                    patience=int(self.cfg.early_stopping.get("patience", 5)),
                    min_delta=float(self.cfg.early_stopping.get("min_delta", 0.0)),
                    mode="min",
                )
            )

        accelerator = "cpu"
        devices = 1
        if self.cfg.device.startswith("cuda"):
            accelerator = "gpu"
            devices = 1

        return pl.Trainer(
            max_epochs=num_epochs,
            accelerator=accelerator,
            devices=devices,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False,
            callbacks=callbacks,
            gradient_clip_val=self.cfg.clip_grad_norm,
            enable_progress_bar=False,
        )

    def _train_lstm_predictor(
        self,
        model: nn.Module,
        y_train: np.ndarray,
        num_epochs: int,
        lr: float,
    ) -> None:
        """
        Train LSTM on sequence y_t to predict next step:
            mu_{t+1} = f(y_{<=t})
        Using teacher forcing with full sequence.
        """

        class _PredictorLightningModule(pl.LightningModule):
            def __init__(self, net: nn.Module, lr_value: float) -> None:
                super().__init__()
                self.net = net
                self.lr_value = lr_value
                self.loss_fn = nn.MSELoss()

            def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
                batch_in, batch_target = batch
                mu = self.net(batch_in)
                loss = self.loss_fn(mu, batch_target)
                self.log("train_loss", loss, on_step=False, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.lr_value)

        # y_train: (T, dim)
        y_seq = torch.from_numpy(y_train).unsqueeze(0).to(self.device)  # (1, T, dim)
        y_in = y_seq[:, :-1, :]
        y_target = y_seq[:, 1:, :]

        dataset = TensorDataset(y_in, y_target)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)

        module = _PredictorLightningModule(model, lr)
        trainer = self._build_trainer(num_epochs)
        trainer.fit(module, loader)

    def _train_lstm_encoder(
        self,
        model: LSTMEncoder,
        x_train: np.ndarray,
        num_epochs: int,
        lr: float,
    ) -> None:
        """
        Train LSTM encoder as a sequence autoencoder:
            x_hat_t = g(H_t)
        Minimizing reconstruction MSE.
        """
        if model.decoder is None:
            logger.warning(
                "LSTMEncoder has no decoder head; skipping encoder training."
            )
            return

        class _EncoderLightningModule(pl.LightningModule):
            def __init__(self, net: LSTMEncoder, lr_value: float) -> None:
                super().__init__()
                self.net = net
                self.lr_value = lr_value
                self.loss_fn = nn.MSELoss()

            def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
                batch_in, batch_target = batch
                x_hat = self.net(batch_in, return_features=False)
                loss = self.loss_fn(x_hat, batch_target)
                self.log("train_loss", loss, on_step=False, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.lr_value)

        x_seq = torch.from_numpy(x_train).unsqueeze(0).to(self.device)
        dataset = TensorDataset(x_seq, x_seq)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)

        module = _EncoderLightningModule(model, lr)
        trainer = self._build_trainer(num_epochs)
        trainer.fit(module, loader)

    def _compute_trend_mu(self, y: np.ndarray) -> np.ndarray:
        """
        Compute mu_t trend for full sequence using trained LSTM predictor.
        We approximate:
            mu_{t+1} = f(y_{<=t})
        and pad mu_0 = y_0 for convenience.
        """
        if self.lstm_y is None:
            raise RuntimeError("LSTM_y not initialized.")

        self.lstm_y.eval()
        with torch.no_grad():
            y_seq = torch.from_numpy(y).unsqueeze(0).to(self.device)
            y_in = y_seq[:, :-1, :]
            mu_next = self.lstm_y(y_in)  # (1, T-1, dim)
            mu_next_np = mu_next.cpu().numpy()[0]
        # pad first element
        mu = np.zeros_like(y, dtype="float32")
        mu[0] = y[0]  # naive trend for first step
        mu[1:] = mu_next_np
        return mu

    def _encode_exogenous(self, x: np.ndarray) -> np.ndarray:
        if self.lstm_x is None:
            raise RuntimeError("LSTM_x not initialized.")

        self.lstm_x.eval()
        with torch.no_grad():
            x_seq = torch.from_numpy(x).unsqueeze(0).to(self.device)
            H_seq = self.lstm_x(x_seq, return_features=True)
            H_np = H_seq.cpu().numpy()[0]
        return H_np.astype("float32")

    def _forecast_trend_mu(self, y_hist: np.ndarray, steps: int) -> np.ndarray:
        if self.lstm_y is None:
            raise RuntimeError("LSTM_y not initialized.")
        if steps <= 0:
            raise ValueError("steps must be > 0")

        self.lstm_y.eval()
        with torch.no_grad():
            y_seq = torch.from_numpy(y_hist).unsqueeze(0).to(self.device)
            mu_future = self.lstm_y.forecast(y_seq, steps=steps)
            mu_future_np = mu_future.cpu().numpy()[0]
        return mu_future_np.astype("float32")

    def fit(
        self,
        y_train: np.ndarray,
        x_train: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit DeepVARMA (re, en, full) on training data.
        y_train: (T, m)
        x_train: (T, n) or None
        """
        T, m = y_train.shape
        n = x_train.shape[1] if x_train is not None else 0

        # Build components
        if self.variant in {"re", "full"}:
            self._build_lstm_y(input_dim=m, output_dim=m)
        if self.variant in {"en", "full"}:
            if x_train is None:
                raise ValueError(
                    f"Variant '{self.variant}' requires exogenous x_train."
                )
            self._build_lstm_x(input_dim=n, output_dim=n)
        self._build_varma()

        # Train LSTMs where needed
        if self.variant in {"re", "full"} and self.lstm_y is not None:
            self._train_lstm_predictor(
                self.lstm_y,
                y_train=y_train,
                num_epochs=self.cfg.num_epochs_lstm_y,
                lr=self.cfg.lr_lstm_y,
            )

        if (
            self.variant in {"en", "full"}
            and self.lstm_x is not None
            and x_train is not None
        ):
            self._train_lstm_encoder(
                self.lstm_x,
                x_train=x_train,
                num_epochs=self.cfg.num_epochs_lstm_x,
                lr=self.cfg.lr_lstm_x,
            )

        # Compute residuals and/or encoded exogenous
        if self.variant == "re":
            mu_train = self._compute_trend_mu(y_train)
            e_train = y_train - mu_train
            H_train = None

        elif self.variant == "en":
            H_train = self._encode_exogenous(x_train)
            e_train = y_train
            mu_train = None

        elif self.variant == "full":
            mu_train = self._compute_trend_mu(y_train)
            e_train = y_train - mu_train
            H_train = self._encode_exogenous(x_train)

        else:
            raise RuntimeError("Unreachable variant dispatch.")

        # Fit VARMA / VARMAX
        self.varma.fit(endog=e_train, exog=H_train)

    def forecast(
        self,
        y_hist: np.ndarray,
        x_hist: Optional[np.ndarray],
        x_future: Optional[np.ndarray],
        steps: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forecast next 'steps' values of y_t.

        y_hist: (T, m)
        x_hist: (T, n) or None
        x_future: (steps, n) or None
        """
        if self.varma is None:
            raise RuntimeError("DeepVARMAHybrid must be fit before forecasting.")

        T, m = y_hist.shape

        # Step 1: compute trend mu for whole history + future conditioning
        mu_hist_full = None
        if self.variant in {"re", "full"}:
            mu_hist_full = self._compute_trend_mu(y_hist)

        # Step 2: build encoded exogenous for residual model
        H_hist = None
        H_future = None

        if self.variant in {"en", "full"}:
            if x_hist is None:
                raise ValueError("x_hist is required for variant 'en' or 'full'.")
            H_hist = self._encode_exogenous(x_hist)
            if x_future is not None:
                H_future = self._encode_exogenous(x_future)
            else:
                # naive: repeat last exogenous encoding
                logger.warning("x_future is None; repeating last H_t for future steps.")
                last_H = H_hist[-1:]
                H_future = np.repeat(last_H, steps, axis=0)

        # Step 3: compute residuals for history
        if self.variant == "re":
            e_hist = y_hist - mu_hist_full
            e_end = e_hist
            exog_future_for_varma = None

        elif self.variant == "en":
            # Here we model y directly with exogenous H
            e_hist = y_hist
            e_end = e_hist
            exog_future_for_varma = H_future

        elif self.variant == "full":
            e_hist = y_hist - mu_hist_full
            e_end = e_hist
            exog_future_for_varma = H_future

        else:
            raise RuntimeError("Unreachable variant dispatch.")

        # Step 4: VARMA forecast residuals or y directly
        # Statsmodels already uses internal state from fit; we only provide exog_future.
        e_future_hat = self.varma.forecast(
            steps=steps,
            exog_future=exog_future_for_varma,
            exog_hist=H_hist,
        )

        if self.variant in {"re", "full"}:
            mu_future = self._forecast_trend_mu(y_hist=y_hist, steps=steps)
            y_future_hat = mu_future + e_future_hat
            mu_future_used = mu_future
        else:
            y_future_hat = e_future_hat
            mu_future_used = np.zeros_like(y_future_hat)

        extras = {
            "e_future_hat": e_future_hat,
            "mu_future": mu_future_used,
        }
        return y_future_hat, extras

    def in_sample_predict(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        In-sample fitted values aligned to y.
        """
        if self.varma is None:
            raise RuntimeError("DeepVARMAHybrid must be fit before in-sample prediction.")

        if self.variant in {"re", "full"}:
            mu = self._compute_trend_mu(y)
        else:
            mu = np.zeros_like(y)

        e_hat = self.varma.in_sample_fitted()

        if self.variant == "en":
            y_hat = e_hat
        else:
            y_hat = np.full_like(mu, np.nan)
            valid_rows = ~np.isnan(e_hat).any(axis=1)
            y_hat[valid_rows] = mu[valid_rows] + e_hat[valid_rows]

        extras = {
            "mu": mu,
            "e_hat": e_hat,
        }
        return y_hat, extras
