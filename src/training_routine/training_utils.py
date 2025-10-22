import torch

def _state_dict_cpu(model: torch.nn.Module):
    return {k: v.detach().to("cpu") for k, v in model.state_dict().items()}

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
    optimizer: torch.optim.Optimizer,
    ckpt_path: str,  # NEW: save immediately
):
    """
    Returns: (new_best_val, new_stalled, improved, should_stop)
    """
    improved = (val_loss < best_val - min_delta) if mode == "min" else (val_loss > best_val + min_delta)

    if improved:
        best_val = val_loss
        stalled = 0
        torch.save({
            "epoch": epoch,
            "model_state": _state_dict_cpu(model),
            "optimizer_state": _optimizer_state_cpu(optimizer),
            "monitor": best_val,
        }, ckpt_path)
        should_stop = False
    else:
        stalled += 1
        should_stop = (patience > 0 and stalled >= patience)

    return best_val, stalled, improved, should_stop
