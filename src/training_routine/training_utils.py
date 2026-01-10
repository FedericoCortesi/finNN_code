import torch
import numpy as np

# necessary helper function to avoid storing states on the GPU
def _state_dict_cpu(model: torch.nn.Module):
    return {k: v.detach().to("cpu") for k, v in model.state_dict().items()}

# necessary helper function to avoid storing states on the GPU
def _optimizer_state_cpu(optim: torch.optim.Optimizer):
    st = optim.state_dict()
    # move all tensors in optimizer state to cpu
    for v in st["state"].values():
        for k, t in list(v.items()):
            if torch.is_tensor(t):
                v[k] = t.detach().to("cpu")
    return st

def early_stopping_step(
    epoch: int,
    val_loss: float,
    best_val: float,
    stalled: int,
    patience: int,
    min_delta: float,
    mode: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    """
    Returns: (new_best_val, new_stalled, improved, should_stop)
    """
    improved = (val_loss < best_val - min_delta) if mode == "min" else (val_loss > best_val + min_delta)


    if improved:
        best_val = val_loss
        stalled = 0
        best_state = { "epoch": epoch, 
                      "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 
                      "optimizer_state": optimizer.state_dict(), }
        should_stop = False
    else:
        stalled += 1
        best_state = None
        should_stop = (patience > 0 and stalled >= patience)

    if val_loss == np.nan:
        should_stop = True

    return best_val, stalled, improved, should_stop
