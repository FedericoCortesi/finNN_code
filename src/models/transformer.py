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
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
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
        self.console_logger = setup_logger(name="Transformer", level="INFO")

        # ---------- shapes ----------
        input_dim  = input_shape[-1]
        output_dim = output_shape if isinstance(output_shape, int) else int(output_shape[0])

        # ---------- hparams ----------
        # Transformer specific
        self.d_model         = int(hparams.get("d_model", 64))
        self.nhead           = int(hparams.get("nhead", 4))
        self.num_layers      = int(hparams.get("num_layers", 2))
        self.dim_feedforward = int(hparams.get("dim_feedforward", 256))
        self.tf_dropout      = float(hparams.get("transformer_dropout", 0.1))
        self.pe_dropout      = float(hparams.get("pe_dropout", 0.1))
        self.activation      = str(hparams.get("activation", "gelu")).lower()
        
        # General
        self.readout         = str(hparams.get("readout", "mean")).lower() # Transformers usually prefer mean/cls
        self.use_ln          = bool(hparams.get("use_ln", False))

        # ---------- Input Projection & PE ----------
        # Map raw input features to d_model
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model, 
            dropout=self.pe_dropout
        )

        # ---------- Transformer Encoder ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.tf_dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=True # Generally more stable convergence
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )

        # ---------- Normalization ----------
        # feature dimension is now fixed to d_model
        self.norm = nn.LayerNorm(self.d_model) if self.use_ln else nn.Identity()

        # ---------- MLP head ----------
        self.mlp_hidden   = hparams.get("mlp_hidden_sizes", [64])
        self.mlp_act      = hparams.get("mlp_activation", ["relu"])
        self.mlp_dropout  = float(hparams.get("dropout_rate", 0.0))

        if isinstance(self.mlp_hidden, int):
            self.mlp_hidden = [self.mlp_hidden]
        if isinstance(self.mlp_act, str):
            self.mlp_act = [self.mlp_act] * len(self.mlp_hidden)

        head_layers = []
        in_dim = self.d_model
        for h, act in zip(self.mlp_hidden, self.mlp_act):
            head_layers.append(nn.Linear(in_dim, int(h)))
            head_layers.append(self._get_activation(act))
            if self.mlp_dropout > 0:
                head_layers.append(nn.Dropout(self.mlp_dropout))
            in_dim = int(h)
        head_layers.append(nn.Linear(in_dim, output_dim))
        self.head = nn.Sequential(*head_layers)

        self._initialize_weights()

    # ---------- utils ----------
    def _initialize_weights(self):
        # Xavier initialization often works well for Transformers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_activation(self, name: str):
        n = name.lower()
        if n == "relu": return nn.ReLU(inplace=False)
        if n == "gelu": return nn.GELU()
        if n == "elu": return nn.ELU(inplace=True)
        if n == "silu": return nn.SiLU(inplace=True)
        if n == "tanh": return nn.Tanh()
        if n == "leakyrelu": return nn.LeakyReLU(0.1, inplace=True)
        raise ValueError(f"Unknown activation {name}")

    def _readout(self, seq_out: torch.Tensor) -> torch.Tensor:
        # seq_out: (B, T, d_model)
        if self.readout == "last":
            return seq_out[:, -1, :]
        elif self.readout == "mean":
            return seq_out.mean(dim=1)
        elif self.readout == "max":
            return seq_out.max(dim=1).values
        else:
            raise ValueError(f"Invalid readout mode: {self.readout}")

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (B,T,D) or (B,T)
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        # 1. Project to latent space
        x = self.input_proj(x) # (B, T, d_model)
        
        # 2. Add position info
        x = self.pos_encoder(x)

        # 3. Transformer blocks
        # mask can be passed if you need causal masking or padding masking
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # 4. Readout & Head
        z = self._readout(x)
        z = self.norm(z)
        y = self.head(z)
        return y