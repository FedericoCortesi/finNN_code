from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from . import register_model
@register_model("mlp")
class MLPRegressor(nn.Module):
    def __init__(self, hparams: Dict[str, Any], input_shape: Tuple[int]):
        super().__init__()

        hidden_sizes = [int(h) for h in hparams.get("hidden_sizes", [])]
        dropout = float(hparams.get("dropout", hparams.get("dropout_rate", 0.2)))
        out_act = hparams.get("output_activation", None)

        # default activations: ReLU for all hidden, Linear for last
        fallback_activations = ["relu"] * (len(hidden_sizes) - 1) + ["linear"]
        activations = hparams.get("activation", fallback_activations)

        # create a list of dimensions and act function
        layers_list = []
        in_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape # Ok to pass (20,) or 20

        # go over the number of layers
        for i, width in enumerate(hidden_sizes):
            # append linear for dimension
            layers_list.append(nn.Linear(in_dim, width))

            # Get activation function
            act = activations[i].lower()
            if act == "relu":
                layers_list.append(nn.ReLU())
            elif act == "tanh":
                layers_list.append(nn.Tanh())
            elif act == "sigmoid":
                layers_list.append(nn.Sigmoid())
            elif act == "leakyrelu":
                layers_list.append(nn.LeakyReLU())
            elif act == "linear":
                pass
                # “linear” means no activation
            else:
                raise AssertionError(
                    f"Unsupported activation '{act}' in hidden layer {i}. "
                    f"Allowed: relu, tanh, sigmoid, leakyrelu, linear."
                )

            # dropout possibly
            if dropout > 0:
                layers_list.append(nn.Dropout(dropout))

            in_dim = width

        # output layer
        layers_list.append(nn.Linear(in_dim, 1))
        if out_act:
            act = out_act.lower()
            if act == "relu":
                layers_list.append(nn.ReLU())
            elif act == "tanh":
                layers_list.append(nn.Tanh())
            elif act == "sigmoid":
                layers_list.append(nn.Sigmoid())
            # else linear → skip

        self.net = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
