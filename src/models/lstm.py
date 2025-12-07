from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from . import register_model
from utils.custom_formatter import setup_logger


@register_model("lstm")
class LSTMRegressor(nn.Module):
    def __init__(
        self,
        hparams: Dict[str, Any],
        input_shape: Tuple[int],
        output_shape: Tuple[int]
    ):
        super().__init__()
        self.console_logger = setup_logger(name="LSTM", level="INFO")

        # ---------- shapes ----------
        self.console_logger.debug(f"input_shape: {input_shape}")
        input_dim  = input_shape[-1]
        output_dim = output_shape if isinstance(output_shape, int) else int(output_shape[0])

        # ---------- hparams ----------
        self.hidden_sizes   = [int(h) for h in hparams.get("lstm_hidden_sizes", [64])]
        self.num_layers     = len(self.hidden_sizes)
        self.bidirectional  = bool(hparams.get("bidirectional", False))
        self.lstm_dropout   = float(hparams.get("lstm_dropout", 0.0))  # inter-layer
        self.readout        = str(hparams.get("readout", "last")).lower()  # "last"|"mean"|"max"
        self.use_ln         = bool(hparams.get("use_ln", False))

        # ---------- build LSTM stack (per-layer sizes) ----------
        lstm_layers = []
        in_dim = input_dim
        for li, h in enumerate(self.hidden_sizes):
            lstm_layers.append(nn.LSTM(
                input_size=in_dim,
                hidden_size=h,
                num_layers=1,             # one layer per module to allow variable widths
                batch_first=True,
                bidirectional=self.bidirectional
            ))
            # next layer input dim = current layer feature dim
            in_dim = h * (2 if self.bidirectional else 1)
        self.lstm_layers = nn.ModuleList(lstm_layers)

        # call this after building self.lstm_layers
        self._initialize_lstm()

        # inter-layer dropout applied between LSTM layers
        self.interlayer_dropout = (
            nn.Dropout(self.lstm_dropout) if self.num_layers > 1 and self.lstm_dropout > 0.0
            else nn.Identity()
        )

        # feature dimension after final LSTM
        feat_dim = in_dim
        self.norm = nn.LayerNorm(feat_dim) if self.use_ln else nn.Identity()

        # ---------- MLP head ----------
        self.mlp_hidden  = hparams.get("mlp_hidden_sizes", [64])
        self.mlp_act     = hparams.get("mlp_activation", ["relu"])
        self.mlp_dropout = float(hparams.get("dropout_rate", 0.0))

        if isinstance(self.mlp_hidden, int):
            self.mlp_hidden = [self.mlp_hidden]
        if isinstance(self.mlp_act, str):
            self.mlp_act = [self.mlp_act] * len(self.mlp_hidden)

        head_layers = []
        in_dim = feat_dim
        for h, act in zip(self.mlp_hidden, self.mlp_act):
            head_layers.append(nn.Linear(in_dim, int(h)))
            head_layers.append(self._get_activation(act))
            if self.mlp_dropout > 0:
                head_layers.append(nn.Dropout(self.mlp_dropout))
            in_dim = int(h)
        head_layers.append(nn.Linear(in_dim, output_dim))
        self.head = nn.Sequential(*head_layers)

    # ---------- utils ----------
    def _initialize_lstm(self):
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.zero_()
                    H = lstm.hidden_size
                    # i, f, g, o chunks
                    param.data[H:2*H].fill_(1.0)  # forget gate bias = +1


    def _get_activation(self, name: str):
        n = name.lower()
        if n == "relu": return nn.ReLU(inplace=False) # doens't save memory but ok for explainer
        if n == "gelu": return nn.GELU()
        if n == "elu": return nn.ELU(inplace=True)
        if n == "silu": return nn.SiLU(inplace=True)
        if n == "tanh": return nn.Tanh()
        if n == "leakyrelu": return nn.LeakyReLU(0.1, inplace=True)
        raise ValueError(f"Unknown activation {name}")

    def _readout(self, seq_out: torch.Tensor) -> torch.Tensor:
        if self.readout == "last":
            return seq_out[:, -1, :]
        elif self.readout == "mean":
            return seq_out.mean(dim=1)
        elif self.readout == "max":
            return seq_out.max(dim=1).values
        else:
            raise ValueError(f"Invalid readout mode: {self.readout}")

    # ---------- forward ----------
    def forward(self, x: torch.Tensor):
        # x: (B,T,D) or (B,T)
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B,T,1)

        seq = x
        for i, lstm in enumerate(self.lstm_layers):
            seq, _ = lstm(seq)  # (B,T,feat_i)
            if i < self.num_layers - 1:
                seq = self.interlayer_dropout(seq)

        z = self._readout(seq)
        z = self.norm(z)
        y = self.head(z)
        return y
