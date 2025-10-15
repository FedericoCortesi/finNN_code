import torch

@torch.no_grad()
def mae(pred: torch.Tensor, targ: torch.Tensor) -> float:
    """Mean Absolute Error (MAE)"""
    return torch.mean(torch.abs(pred - targ)).item()

@torch.no_grad()
def mse(pred: torch.Tensor, targ: torch.Tensor) -> float:
    """Mean Squared Error (MSE)"""
    return torch.mean((pred - targ) ** 2).item()

@torch.no_grad()
def directional_accuracy_pct(pred: torch.Tensor, targ: torch.Tensor) -> float:
    """Directional accuracy in percentage (sign match rate)."""
    ps = torch.sign(pred)
    ts = torch.sign(targ)
    return ps.eq(ts).float().mean().item() * 100.0
