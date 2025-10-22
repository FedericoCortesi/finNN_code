from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from . import register_model
from utils.custom_formatter import setup_logger
@register_model("mlp")
class MLPRegressor(nn.Module):
    def __init__(self, 
                 hparams: Dict[str, Any], 
                 input_shape: Tuple[int],
                 output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape


        # instantiate logger
        self.console_logger = setup_logger(name="MLP", level="INFO")

        # get values
        hidden_sizes = [int(h) for h in hparams.get("hidden_sizes", [])]
        dropout = float(hparams.get("dropout", hparams.get("dropout_rate", 0.2)))
        out_act = hparams.get("output_activation", None)

        # default activations: ReLU for all hidden, Linear for last
        fallback_activations = ["relu"] * (len(hidden_sizes) - 1) + ["linear"]
        activations = hparams.get("activation", fallback_activations)

        if isinstance(activations, str):
            old_activation = activations # just to print
            activations = [old_activation] * len(hidden_sizes)
            self.console_logger.warning(f"Passed string {old_activation} for activations, using {activations} instead")

        # create a list of dimensions and act function
        layers_list = []
        in_dim = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape # Ok to pass (20,) or 20

        # go over the number of layers
        for i, width in enumerate(hidden_sizes):
            # append linear for dimension
            layers_list.append(nn.Linear(in_dim, width))

            # Get activation function
            act = activations[i].lower()
            if act == "relu":
                layers_list.append(nn.ReLU())
            elif act == "gelu":
                layers_list.append(nn.GELU())
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
        layers_list.append(nn.Linear(in_dim, self.output_shape))
        if out_act:
            act = out_act.lower()
            if act == "relu":
                layers_list.append(nn.ReLU())
                self.console_logger(f"Non linear final activation! Are you sure to continue with {act}?")
            elif act == "tanh":
                layers_list.append(nn.Tanh())
                self.console_logger(f"Non linear final activation! Are you sure to continue with {act}?")
            elif act == "sigmoid":
                layers_list.append(nn.Sigmoid())
                self.console_logger(f"Non linear final activation! Are you sure to continue with {act}?")
            # else linear → skip

        self.net = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
