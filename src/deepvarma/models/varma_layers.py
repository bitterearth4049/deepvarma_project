from dataclasses import dataclass
from typing import Optional

import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings


@dataclass
class VARMALayerConfig:
    p: int
    q: int
    trend: str = "n"
    error_cov_type: str = "diagonal"
    maxiter: int = 200
    use_exogenous: bool = False
    lags_exog: int = 0  # s in the DeepVARMA paper


class VARMALayer:
    """
    Wraps statsmodels VARMAX or SARIMAX to implement VARMA / VARMAX behavior
    for residuals e_t or endogenous y_t with exogenous H_t.

    For DeepVARMA-re:
        e_t = sum Phi_i e_{t-i} + sum Theta_j eps_{t-j}
    For DeepVARMA-en and DeepVARMA:
        Phi(B) e_t = Theta(B) eps_t + sum_k Gamma_k H_{t-k}

    Implementation detail:
    - If number of endogenous variables m > 1: use VARMAX.
    - If m == 1: use SARIMAX with order=(p, 0, q) to avoid VAR() crashing.
    """

    def __init__(self, cfg: VARMALayerConfig):
        self.cfg = cfg
        self._model = None
        self._result = None
        self._is_univariate: bool = False
        self._k_endog: Optional[int] = None
        self._endog_len: int = 0
        self._cut: int = 0

    def _build_exog_with_lags(self, H: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        H: (T, d) exogenous features
        return X_exog: (T - lags_exog, d * (lags_exog + 1))
        """
        if H is None:
            return None
        s = self.cfg.lags_exog
        if s < 0:
            raise ValueError("lags_exog must be >= 0")

        T, d = H.shape
        if s == 0:
            return H

        # build [H_t, H_{t-1}, ..., H_{t-s}]
        exogs = []
        for k in range(s + 1):
            exogs.append(H[s - k : T - k, :])
        X = np.concatenate(exogs, axis=1)
        return X

    def fit(
        self,
        endog: np.ndarray,
        exog: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit VARMA / VARMAX (multivariate) or ARMA / ARMAX (univariate) on endog and optional exog.
        endog: (T, m)
        exog: (T, d) or (T, d*(s+1)) if lags are already baked in.
        """
        if endog.ndim != 2:
            raise ValueError(f"endog must be 2D (T, m), got shape {endog.shape}")

        T, m = endog.shape
        self._k_endog = m
        self._is_univariate = m == 1
        self._endog_len = T
        self._cut = 0

        if self.cfg.use_exogenous and exog is None:
            raise ValueError("use_exogenous=True but exog is None.")

        # Build lagged exog if needed
        if self.cfg.use_exogenous:
            exog_proc = self._build_exog_with_lags(exog)
            if self.cfg.lags_exog > 0:
                cut = self.cfg.lags_exog
                self._cut = cut
                endog_proc = endog[cut:]
            else:
                endog_proc = endog
        else:
            endog_proc = endog
            exog_proc = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)

            if self._is_univariate:
                # endog_proc: (T, 1) -> flatten to (T,)
                endog_1d = endog_proc[:, 0]

                # ARMA(p, q) with optional exogenous -> SARIMAX
                if self.cfg.use_exogenous:
                    self._model = SARIMAX(
                        endog_1d,
                        exog=exog_proc,
                        order=(self.cfg.p, 0, self.cfg.q),
                        trend=self.cfg.trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                else:
                    self._model = SARIMAX(
                        endog_1d,
                        order=(self.cfg.p, 0, self.cfg.q),
                        trend=self.cfg.trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
            else:
                # Proper multivariate VARMA/VARMAX
                self._model = VARMAX(
                    endog_proc,
                    exog=exog_proc,
                    order=(self.cfg.p, self.cfg.q),
                    trend=self.cfg.trend,
                    error_cov_type=self.cfg.error_cov_type,
                )

            self._result = self._model.fit(maxiter=self.cfg.maxiter, disp=False)

    def forecast(
        self,
        steps: int,
        exog_future: Optional[np.ndarray] = None,
        exog_hist: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forecast next 'steps' residuals or endogenous values.
        exog_future: (steps, d) for future exogenous features (before lag expansion).
        exog_hist: (T, d) historical exogenous features (before lag expansion).
        """
        if self._result is None:
            raise RuntimeError("VARMALayer must be fit before forecasting.")

        if self.cfg.use_exogenous:
            if exog_future is None:
                raise ValueError("Exogenous forecast required but exog_future is None.")
            if self.cfg.lags_exog > 0:
                if exog_hist is None:
                    raise ValueError("exog_hist is required when lags_exog > 0.")
                X_future = self._build_exog_future_with_lags(exog_hist, exog_future)
            else:
                X_future = self._build_exog_with_lags(exog_future)
        else:
            X_future = None

        forecast_res = self._result.get_forecast(steps=steps, exog=X_future)
        pred = forecast_res.predicted_mean

        # For SARIMAX, predicted_mean is 1D array; reshape to (T, 1)
        if self._is_univariate:
            pred = np.asarray(pred, dtype="float32").reshape(-1, 1)
        else:
            pred = np.asarray(pred, dtype="float32")

        return pred

    def _build_exog_future_with_lags(
        self, H_hist: np.ndarray, H_future: np.ndarray
    ) -> np.ndarray:
        """
        Build lagged exogenous design matrix for the future horizon.
        Uses history to provide lags for the first future steps.
        """
        s = self.cfg.lags_exog
        if s <= 0:
            return H_future

        if H_hist.shape[0] < s:
            raise ValueError("Not enough exog history for requested lags_exog.")

        concat = np.concatenate([H_hist, H_future], axis=0)
        T_hist = H_hist.shape[0]
        steps = H_future.shape[0]
        d = H_hist.shape[1]
        X_future = np.zeros((steps, d * (s + 1)), dtype="float32")

        for t in range(steps):
            idx = T_hist + t
            parts = []
            for k in range(s + 1):
                parts.append(concat[idx - k])
            X_future[t] = np.concatenate(parts, axis=0)

        return X_future

    def in_sample_fitted(self) -> np.ndarray:
        """
        Return fitted values aligned to the original endog length.
        For lagged exogenous models, the first lags_exog rows are NaN.
        """
        if self._result is None:
            raise RuntimeError("VARMALayer must be fit before calling in_sample_fitted.")

        fitted = self._result.fittedvalues
        if hasattr(fitted, "to_numpy"):
            fitted = fitted.to_numpy()

        fitted = np.asarray(fitted, dtype="float32")
        if self._is_univariate:
            fitted = fitted.reshape(-1, 1)

        if fitted.shape[0] < self._endog_len:
            pad = self._endog_len - fitted.shape[0]
            padded = np.full((self._endog_len, fitted.shape[1]), np.nan, dtype="float32")
            padded[pad:] = fitted
            return padded

        if self._cut > 0:
            padded = np.full((self._endog_len, fitted.shape[1]), np.nan, dtype="float32")
            padded[self._cut :] = fitted
            return padded

        return fitted

    @property
    def summary(self) -> str:
        if self._result is None:
            return "VARMALayer not fitted yet."
        return str(self._result.summary())
