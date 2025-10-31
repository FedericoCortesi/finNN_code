# --- Imports ---
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm

from pipeline.walkforward import WFCVGenerator
from config.config_types import AppConfig
from utils.paths import DATA_DIR, PRICE_EXPERIMENTS_DIR
from models import create_model

# =========================
# Config
# =========================
NAME  = "exp_011_mlp_40"
TRIAL = "trial_20251029_182517"   # adjust as needed
BASE  = Path(PRICE_EXPERIMENTS_DIR) / NAME / TRIAL

# Whether to report normalized MSE = MSE / Var(y) on each split
USE_NMSE = True

# =========================
# Utilities
# =========================
def load_cfg(base: Path) -> AppConfig:
    cfg_json = json.loads((base / "config_snapshot.json").read_text())
    return AppConfig.from_dict(cfg_json["cfg"])

def maybe_load_df_master(cfg: AppConfig):
    if cfg.data["df_master"] is None:
        return None
    return pd.read_parquet(Path(DATA_DIR) / cfg.data["df_master"])

def add_const(x: np.ndarray) -> np.ndarray:
    return sm.add_constant(x, has_constant="add")

def fit_ols_per_target(x_tr: np.ndarray, y_tr: np.ndarray):
    x_tr_c = add_const(x_tr)
    models = []
    for j in range(y_tr.shape[1]):
        models.append(sm.OLS(y_tr[:, j], x_tr_c).fit())
    return models

def pred_ols(models, x: np.ndarray) -> np.ndarray:
    x_c = add_const(x)
    preds = [m.predict(x_c) for m in models]  # list of (N,)
    return np.column_stack(preds)

def load_fold_mlp(base: Path, fold_idx: int, cfg: AppConfig, device: str):
    input_shape  = (cfg.walkforward.lags,)
    output_shape = cfg.walkforward.lookback + 1
    ckpt_path    = base / f"fold_{fold_idx:03d}" / "model_best.pt"

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state"].items()}

    model = create_model(cfg.model, input_shape, output_shape)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

@torch.no_grad()
def pred_mlp(model: torch.nn.Module, x_np: np.ndarray, device: str) -> np.ndarray:
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    return model(x).detach().cpu().numpy()

def dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean() * 100.0)

def fold_metrics(y_tr, y_te, yhat_tr_ols, yhat_te_ols, yhat_tr_mlp, yhat_te_mlp, use_nmse=False) -> Dict[str, float]:
    mse_tr_ols = ((y_tr - yhat_tr_ols) ** 2).mean(axis=0)
    mse_te_ols = ((y_te - yhat_te_ols) ** 2).mean(axis=0)
    mse_tr_mlp = ((y_tr - yhat_tr_mlp) ** 2).mean(axis=0)
    mse_te_mlp = ((y_te - yhat_te_mlp) ** 2).mean(axis=0)

    if use_nmse:
        var_tr = y_tr.var(axis=0); var_tr[var_tr == 0] = 1.0
        var_te = y_te.var(axis=0); var_te[var_te == 0] = 1.0
        mse_tr_ols /= var_tr; mse_tr_mlp /= var_tr
        mse_te_ols /= var_te; mse_te_mlp /= var_te

    return {
        "mse_tr_ols": float(mse_tr_ols.mean()),
        "mse_te_ols": float(mse_te_ols.mean()),
        "mse_tr_mlp": float(mse_tr_mlp.mean()),
        "mse_te_mlp": float(mse_te_mlp.mean()),
        "dir_tr_ols": dir_acc(y_tr, yhat_tr_ols),
        "dir_te_ols": dir_acc(y_te, yhat_te_ols),
        "dir_tr_mlp": dir_acc(y_tr, yhat_tr_mlp),
        "dir_te_mlp": dir_acc(y_te, yhat_te_mlp),
        "normalized_mse": USE_NMSE
    }

def print_table(df: pd.DataFrame, title: str):
    print("\n" + title)
    print(f"{'Model':<6}\t{'Train MSE':>10}\t{'Test MSE':>10}\t{'Train Acc':>9}\t{'Test Acc':>8}")
    for _, r in df.iterrows():
        print(f"{r['Model']:<6}\t{r['Train MSE']:>10.5f}\t{r['Test MSE']:>10.5f}\t{r['Train Acc']:>8.2f}%\t{r['Test Acc']:>7.2f}%")

# =========================
# Main
# =========================
def main():
    cfg = load_cfg(BASE)
    _   = maybe_load_df_master(cfg)  # optional, not used below

    wf = WFCVGenerator(config=cfg.walkforward)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: List[Dict[str, Any]] = []

    for fold_idx, fold in tqdm(enumerate(wf.folds())):
        X_tr, y_tr, X_val, y_val, X_te, y_te = fold

        # OLS (per target)
        ols_models   = fit_ols_per_target(X_tr, y_tr)
        yhat_tr_ols  = pred_ols(ols_models, X_tr)
        yhat_te_ols  = pred_ols(ols_models, X_te)

        # MLP (per fold checkpoint)
        mlp = load_fold_mlp(BASE, fold_idx, cfg, device)
        yhat_tr_mlp = pred_mlp(mlp, X_tr, device)
        yhat_te_mlp = pred_mlp(mlp, X_te, device)

        m = fold_metrics(y_tr, y_te, yhat_tr_ols, yhat_te_ols, yhat_tr_mlp, yhat_te_mlp, use_nmse=USE_NMSE)

        rows += [
            {"fold": fold_idx, "Model": "OLS", "Train MSE": m["mse_tr_ols"], "Test MSE": m["mse_te_ols"],
             "Train Acc": m["dir_tr_ols"], "Test Acc": m["dir_te_ols"], "normalized_mse": USE_NMSE},
            {"fold": fold_idx, "Model": "MLP", "Train MSE": m["mse_tr_mlp"], "Test MSE": m["mse_te_mlp"],
             "Train Acc": m["dir_tr_mlp"], "Test Acc": m["dir_te_mlp"], "normalized_mse": USE_NMSE},
        ]

    results_df = pd.DataFrame(rows)

    # Aggregate across folds
    agg_mean = (results_df
                .groupby("Model", as_index=False)[["Train MSE", "Test MSE", "Train Acc", "Test Acc"]]
                .mean())
    agg_std  = (results_df
                .groupby("Model", as_index=False)[["Train MSE", "Test MSE", "Train Acc", "Test Acc"]]
                .std()
                .rename(columns=lambda c: c if c == "Model" else c + " Std"))

    summary = pd.merge(agg_mean, agg_std, on="Model", how="left")

    # Console output
    print_table(agg_mean, title=f"Average across folds (USE_NMSE={USE_NMSE})")

    # Save
    out_dir = BASE / "analysis"
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / "per_fold_metrics.csv", index=False)
    agg_mean.to_csv(out_dir / "fold_avg_metrics.csv", index=False)
    summary.to_csv(out_dir / "fold_avg_metrics_with_std.csv", index=False)

if __name__ == "__main__":
    main()
