import math
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from . import register_model
from utils.custom_formatter import setup_logger

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute pe
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

@register_model("transformer")
class TransformerRegressor(nn.Module):
    def __init__(
        self,
        hparams: Dict[str, Any],
        input_shape: Tuple[int],
        output_shape: Tuple[int]
    ):
        super().__init__()
        # Logger setup (safe in __init__)
        self.console_logger = setup_logger(name="Transformer", level="INFO")

        # ---------- shapes ----------
        input_dim  = input_shape[-1]
        output_dim = output_shape if isinstance(output_shape, int) else int(output_shape[0])

        # ---------- hparams ----------
        self.d_model         = int(hparams.get("d_model", 64))
        self.nhead           = int(hparams.get("nhead", 4))
        self.num_layers      = int(hparams.get("num_layers", 2))
        self.dim_feedforward = int(hparams.get("dim_feedforward", 256))
        self.tf_dropout      = float(hparams.get("transformer_dropout", 0.1))
        self.pe_dropout      = float(hparams.get("pe_dropout", 0.1))
        self.activation      = str(hparams.get("activation", "gelu")).lower()
        self.projection      = str(hparams.get('projection', 'linear')).lower()
        self.readout         = str(hparams.get("readout", "mean")).lower()
        self.use_ln          = bool(hparams.get("use_ln", False))

        # ---------- Input Projection ----------
        if self.projection == 'linear':
            self.input_proj = nn.Linear(input_dim, self.d_model)
        elif self.projection == 'conv':
            # Convolutional Stem
            self.input_proj = nn.Sequential(
                nn.Conv1d(input_dim, self.d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_model),
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
                nn.ReLU()
            )

        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.pe_dropout)

        # ---------- Transformer Encoder ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.tf_dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # ---------- Heads ----------
        self.norm = nn.LayerNorm(self.d_model) if self.use_ln else nn.Identity()
        
        self.mlp_hidden = hparams.get("mlp_hidden_sizes", [64])
        self.mlp_act    = hparams.get("mlp_activation", ["relu"])
        self.mlp_dropout= float(hparams.get("dropout_rate", 0.0))

        head_layers = []
        in_dim = self.d_model
        # Handle int/list inputs for mlp_hidden
        if isinstance(self.mlp_hidden, int): self.mlp_hidden = [self.mlp_hidden]
        if isinstance(self.mlp_act, str): self.mlp_act = [self.mlp_act] * len(self.mlp_hidden)

        for h, act in zip(self.mlp_hidden, self.mlp_act):
            head_layers.append(nn.Linear(in_dim, int(h)))
            head_layers.append(self._get_activation(act))
            if self.mlp_dropout > 0:
                head_layers.append(nn.Dropout(self.mlp_dropout))
            in_dim = int(h)
        head_layers.append(nn.Linear(in_dim, output_dim))
        self.head = nn.Sequential(*head_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_activation(self, name: str):
        n = name.lower()
        if n == "relu": return nn.ReLU(inplace=False)
        if n == "gelu": return nn.GELU()
        if n == "tanh": return nn.Tanh()
        return nn.ReLU(inplace=False)

    def _readout(self, seq_out: torch.Tensor) -> torch.Tensor:
        if self.readout == "last": return seq_out[:, -1, :]
        elif self.readout == "mean": return seq_out.mean(dim=1)
        elif self.readout == "max": return seq_out.max(dim=1).values
        return seq_out.mean(dim=1)

    # ---------- FORWARD (FIXED) ----------
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (B, T, D)
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        # 1. Input Projection
        if self.projection == 'linear':
            x = self.input_proj(x)
            
        elif self.projection == 'conv':
            # FIX: Permute and enforce contiguous memory layout
            # (B, T, D) -> (B, D, T)
            x = x.permute(0, 2, 1).clone()
            
            x = self.input_proj(x)
            
            # FIX: Permute back and enforce contiguous again
            # (B, D, T) -> (B, T, D)
            x = x.permute(0, 2, 1).clone()

        # 2. Positional Encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # 3. Output Head
        z = self._readout(x)
        z = self.norm(z)
        y = self.head(z)
        
        return y