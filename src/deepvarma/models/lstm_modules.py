from typing import Tuple

import torch
from torch import nn


class LSTMPredictor(nn.Module):
    """
    LSTM prediction layer for endogenous y_t.
    Implements:
        mu_t = LSTM(y_t, h_{t-1})
    in the sense that we feed a sequence of y and map hidden states to trend mu.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.output_layer = nn.Linear(hidden_dim * num_directions, output_dim)

    def forward(self, y_seq: torch.Tensor) -> torch.Tensor:
        """
        y_seq: (batch, seq, input_dim)
        returns mu_seq: (batch, seq, output_dim)
        """
        out, _ = self.lstm(y_seq)
        mu = self.output_layer(out)
        return mu

    def forecast(self, y_hist: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Autoregressive rollout using last hidden state.
        y_hist: (batch, seq, input_dim)
        returns mu_future: (batch, steps, output_dim)
        """
        if steps <= 0:
            raise ValueError("steps must be > 0")

        _, (h, c) = self.lstm(y_hist)
        last_input = y_hist[:, -1:, :]

        preds = []
        for _ in range(steps):
            out, (h, c) = self.lstm(last_input, (h, c))
            mu_step = self.output_layer(out)
            preds.append(mu_step)
            last_input = mu_step

        return torch.cat(preds, dim=1)


class LSTMEncoder(nn.Module):
    """
    LSTM coding layer for exogenous x_t.
    Implements:
        H_t = LSTM(x_t, h_{t-1})
    where H_t are the hidden states. Optionally adds a decoder head to
    reconstruct x for unsupervised training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
        output_dim: int = None,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim * num_directions

        # optional decoder for reconstruction
        self.decoder = None
        if output_dim is not None:
            self.decoder = nn.Linear(self.hidden_dim, output_dim)

    def forward(
        self, x_seq: torch.Tensor, return_features: bool = True
    ) -> torch.Tensor:
        """
        x_seq: (batch, seq, input_dim)
        if return_features: returns H_seq of shape (batch, seq, hidden_dim)
        else: returns decoded x_hat of shape (batch, seq, output_dim)
        """
        out, _ = self.lstm(x_seq)
        if return_features:
            return out
        if self.decoder is None:
            raise RuntimeError("Decoder head not defined for LSTMEncoder.")
        return self.decoder(out)
