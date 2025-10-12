from typing import Tuple, Dict, Any
from tensorflow import keras
from tensorflow.keras import layers #type: ignore
from tensorflow.keras.regularizers import l2 as l2_reg_fn #type: ignore

def build_model(hparams: Dict[str, Any], input_shape: Tuple[int]):
    """
    hparams:
      - hidden_sizes: [int, int, ...]                # takes precedence

    Optional keys:
      - dropout (or dropout_rate)
      - activation
      - l2 (or l2_reg)
      - learning_rate
      - loss                   (default: 'mean_squared_error')
      - output_activation      (default: None)
    """
    # ---- read hparams with sensible fallbacks ----
    hidden_sizes = hparams.get("hidden_sizes")
    hidden_sizes = [int(h) for h in hidden_sizes]
    dropout = float(hparams.get("dropout", hparams.get("dropout_rate", 0.2)))
    activation = hparams.get("activation", "relu")
    l2_reg = float(hparams.get("l2_reg",  0.001))
    out_act = hparams.get("output_activation", None)
    

    # ---- build model (Sequential, like your original) ----
    model = keras.Sequential(name="mlp_regressor_yaml")
    model.add(layers.Input(shape=(input_shape,)))

    for width in hidden_sizes:
        model.add(layers.Dense(
            int(width),
            activation=activation,
            kernel_regularizer=l2_reg_fn(l2_reg)
        ))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation=out_act))

    # model gets compiled in the trainer class
    return model
