import torch
import torch.nn as nn
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



class QLikeLoss(nn.Module):
    """
    QLIKE loss for variance forecasts:
        L(sigm^2_true, sigm^2_pred) = log(sigm^2_pred) + sigm^2_true / sigm^2_pred
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-12):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
        self.eps = eps

    def forward(self, sigma2_pred: torch.Tensor, sigma2_true: torch.Tensor) -> torch.Tensor:
        # ensure strictly positive sigm^2_pred
        sigma2_pred = torch.clamp(sigma2_pred, min=self.eps)
        loss = torch.log(sigma2_pred) + (sigma2_true / sigma2_pred)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'
