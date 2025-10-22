from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from . import register_model
from utils.custom_formatter import setup_logger

@register_model("simplecnn")
class CNN1D(nn.Module):
    def __init__(self, 
                 hparams: dict,
                 input_shape: Tuple[int]=None, 
                 output_shape:Tuple[int]=None):  # Just for symmetry
        super().__init__()

        # --- get hyperparameters safely ---
        self.conv_channels = hparams.get("conv_channels", [16])
        self.hidden_sizes  = hparams.get("hidden_sizes", [64])
        self.activation    = hparams.get("activation", ["relu"])
        self.use_bn        = hparams.get("use_bn", True)
        self.dropout_rate  = hparams.get("dropout_rate", 0.1)
        self.kernel_size   = hparams.get("kernel_size", 3)
        self.padding       = hparams.get("padding", 0)

        # Pooling choices
        self.pool_type = str(hparams.get("pool", "adaptive_avg")).lower()   # "adaptive_avg" | "adaptive_max"
        self.pool_k    = int(hparams.get("pool_k", 1))                      # >=1

        # Output
        self.output_shape = output_shape

        # "Depth" of the tensor
        self.in_ch = 1
    
        # ensure lists
        if isinstance(self.conv_channels, (int, str)):
            self.conv_channels = [int(self.conv_channels)]
        if isinstance(self.hidden_sizes, int):
            self.hidden_sizes = [self.hidden_sizes]
        if isinstance(self.activation, str):
            # broadcast single activation to all conv blocks
            self.activation = [self.activation] * len(self.conv_channels)

        assert len(self.activation) == len(self.conv_channels), \
            "len(self.activation) and len(self.conv_channels) should be equal, " \
            f"got {len(self.activation)} and {len(self.conv_channels)} instead"
    
        # --- build convolutional block(s) ---
        conv_layers = []
        for out_ch, act_name in zip(self.conv_channels, self.activation):
            conv_layers.append(nn.Conv1d(self.in_ch, out_ch, kernel_size=self.kernel_size, padding=self.padding))
            if self.use_bn:
                conv_layers.append(nn.BatchNorm1d(out_ch))
            conv_layers.append(self._get_activation(act_name))
            self.in_ch = out_ch  # next conv layer input = prev output

        self.conv_blocks = nn.Sequential(*conv_layers)

        # --- pooling ---
        if self.pool_type == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool1d(self.pool_k)
        elif self.pool_type == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool1d(self.pool_k)
        else:
            raise ValueError("pool must be 'adaptive_avg' or 'adaptive_max'")

        # --- build fully connected block(s) ---
        # If we pool to k>1, FC sees channels * k features; if k==1, it’s just channels
        fc_layers = []
        in_dim = self.conv_channels[-1] * self.pool_k
        for h in self.hidden_sizes:
            fc_layers.append(nn.Linear(in_dim, h))
            fc_layers.append(nn.ReLU())  # keep as-is; make configurable later if you want
            if self.dropout_rate > 0:
                fc_layers.append(nn.Dropout(self.dropout_rate))
            in_dim = h

        fc_layers.append(nn.Linear(in_dim, self.output_shape))  # final regression output
        self.fc = nn.Sequential(*fc_layers)

    def _get_activation(self, name: str) -> nn.Module:
        name = (name or "relu").lower()
        if name == "relu":        return nn.ReLU(inplace=True)
        elif name == "gelu":      return nn.GELU()
        elif name == "elu":       return nn.ELU(inplace=True)
        elif name == "silu":      return nn.SiLU(inplace=True)
        elif name == "tanh":      return nn.Tanh()
        elif name == "leakyrelu": return nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)         # (B,1,L)
        x = self.conv_blocks(x)        # (B,C,L')
        x = self.pool(x)               # (B,C,k)
        if self.pool_k == 1:
            x = x.squeeze(-1)          # (B,C)
        else:
            x = x.flatten(1)           # (B,C*k)
        x = self.fc(x)                 # (B,1)
        return x
