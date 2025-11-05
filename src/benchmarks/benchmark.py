# --- Imports ---
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pipeline.walkforward import WFCVGenerator
from config.config_types import AppConfig
from utils.paths import DATA_DIR, PRICE_EXPERIMENTS_DIR, VOL_EXPERIMENTS_DIR

# =========================
# Config
# =========================
NAME  = "exp_007_mlp_40"
TRIAL = "trial_20251102_114602"
BASE  = Path(VOL_EXPERIMENTS_DIR) / NAME / TRIAL
ACCURACY: int = 8

print(f'Analyze {BASE}')

# Normalize MSE by target variance on each split
USE_NMSE = False


# =========================
# Utilities
# =========================
def load_cfg(base: Path) -> AppConfig:
    cfg_json = json.loads((base / "config_snapshot.json").read_text())
    return AppConfig.from_dict(cfg_json["cfg"])

def add_const(x: np.ndarray) -> np.ndarray:
    return sm.add_constant(x, has_constant="add")

def fit_ols_per_target(x_tr: np.ndarray, y_tr: np.ndarray):
    x_tr_c = add_const(x_tr)
    return [sm.OLS(y_tr[:, j], x_tr_c).fit() for j in range(y_tr.shape[1])]

def pred_ols(models, x: np.ndarray) -> np.ndarray:
    x_c = add_const(x)
    return np.column_stack([m.predict(x_c) for m in models])

def dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional accuracy in percent across all dims & samples."""
    return float((np.sign(y_true) == np.sign(y_pred)).mean() * 100.0)

def print_table(df: pd.DataFrame, title: str):
    cols = [
        ("Model", "str"),
        ("Train MSE", "mse"), ("Val MSE", "mse"), ("Test MSE", "mse"),
        ("Train DirAcc", "pct"), ("Val DirAcc", "pct"), ("Test DirAcc", "pct"),
        ("normalized", "str"),
    ]
    present = [(c, t) for c, t in cols if c in df.columns]

    print("\n" + title)
    header = "\t".join([f"{'Model':<10}" if c == "Model" else f"{c:>14}" for c, _ in present])
    print(header)

    for _, r in df.iterrows():
        row = []
        for c, t in present:
            if c == "Model":
                row.append(f"{str(r[c]):<10}")
            elif t == "mse":
                row.append(f"{float(r[c]):>14.{ACCURACY}f}")
            elif t == "pct":
                # values are already in percent for NN; OLS we compute as percent
                row.append(f"{float(r[c]):>13.{ACCURACY}f}%")
            else:
                row.append(f"{str(r[c]):>14}")
        print("\t".join(row))

# =========================
# Main
# =========================
def main():
    cfg = load_cfg(BASE)

    # 1) Build folds + per-fold split variances (scalar, like your np.var usage)
    wf = WFCVGenerator(config=cfg.walkforward)
    var_rows, fold_cache = [], []
    for fold_idx, fold in tqdm(enumerate(wf.folds()), desc="Preparing folds"):
        X_tr, y_tr, X_val, y_val, X_te, y_te = fold
        fold_cache.append((X_tr, y_tr, X_val, y_val, X_te, y_te))
        var_rows.append({
            "fold": fold_idx,
            "var_train": float(np.var(y_tr)),
            "var_val":   float(np.var(y_val)),
            "var_test":  float(np.var(y_te)),
        })
    var_df = pd.DataFrame(var_rows)

    # 2) Read NN results.csv with your exact schema
    nn_path = BASE / "results.csv"
    use_cols = [
        "trial","fold",
        "tr_loss","val_loss","test_loss",
        "tr_mae","val_mae","test_mae",
        "tr_directional_accuracy_pct","val_directional_accuracy_pct","test_directional_accuracy_pct",
        "seconds","model_path",
    ]
    nn_df = pd.read_csv(nn_path, usecols=use_cols)

    # Merge with variances
    nn = nn_df.merge(var_df, on="fold", how="left")

    # Normalize NN losses if requested
    if USE_NMSE:
        nn["NN_train"] = nn["tr_loss"]   / nn["var_train"]
        nn["NN_val"]   = nn["val_loss"]  / nn["var_val"]
        nn["NN_test"]  = nn["test_loss"] / nn["var_test"]
    else:
        nn["NN_train"] = nn["tr_loss"]
        nn["NN_val"]   = nn["val_loss"]
        nn["NN_test"]  = nn["test_loss"]

    # NN DirAcc already in percent in your CSV
    nn["NN_train_diracc"] = nn["tr_directional_accuracy_pct"]
    nn["NN_val_diracc"]   = nn["val_directional_accuracy_pct"]
    nn["NN_test_diracc"]  = nn["test_directional_accuracy_pct"]

    # 3) OLS per fold (MSE/NMSE + DirAcc)
    ols_rows: List[Dict[str, Any]] = []
    for fold_idx, (X_tr, y_tr, X_val, y_val, X_te, y_te) in tqdm(
        enumerate(fold_cache), total=len(fold_cache), desc="Fitting OLS"
    ):
        models        = fit_ols_per_target(X_tr, y_tr)
        yhat_tr_ols  = pred_ols(models, X_tr)
        yhat_val_ols = pred_ols(models, X_val)
        yhat_te_ols  = pred_ols(models, X_te)

        mse_tr  = ((y_tr  - yhat_tr_ols )**2).mean(axis=0).mean()
        mse_val = ((y_val - yhat_val_ols)**2).mean(axis=0).mean()
        mse_te  = ((y_te  - yhat_te_ols )**2).mean(axis=0).mean()

        da_tr  = dir_acc(y_tr,  yhat_tr_ols)
        da_val = dir_acc(y_val, yhat_val_ols)
        da_te  = dir_acc(y_te,  yhat_te_ols)

        v = var_df.loc[var_df["fold"] == fold_idx].iloc[0]
        if USE_NMSE:
            mse_tr /= (v["var_train"] if v["var_train"] != 0 else 1.0)
            mse_val/= (v["var_val"]   if v["var_val"]   != 0 else 1.0)
            mse_te /= (v["var_test"]  if v["var_test"]  != 0 else 1.0)

        ols_rows.append({
            "fold": fold_idx,
            "OLS_train": mse_tr, "OLS_val": mse_val, "OLS_test": mse_te,
            "OLS_train_diracc": da_tr, "OLS_val_diracc": da_val, "OLS_test_diracc": da_te,
        })
    ols = pd.DataFrame(ols_rows)

    # 4) Per-fold comparison + normalized flag
    per_fold = ols.merge(
        nn[[
            "fold",
            "NN_train","NN_val","NN_test",
            "NN_train_diracc","NN_val_diracc","NN_test_diracc"
        ]],
        on="fold", how="left"
    )
    per_fold["normalized"] = USE_NMSE

    # 5) Aggregations
    avg_df = pd.DataFrame([
        {"Model": "OLS",
         "Train MSE": per_fold["OLS_train"].mean(),
         "Val MSE":   per_fold["OLS_val"].mean(),
         "Test MSE":  per_fold["OLS_test"].mean(),
         "Train DirAcc": per_fold["OLS_train_diracc"].mean(),
         "Val DirAcc":   per_fold["OLS_val_diracc"].mean(),
         "Test DirAcc":  per_fold["OLS_test_diracc"].mean(),
         "normalized": USE_NMSE},
        {"Model": "NN",
         "Train MSE": per_fold["NN_train"].mean(),
         "Val MSE":   per_fold["NN_val"].mean(),
         "Test MSE":  per_fold["NN_test"].mean(),
         "Train DirAcc": per_fold["NN_train_diracc"].mean(),
         "Val DirAcc":   per_fold["NN_val_diracc"].mean(),
         "Test DirAcc":  per_fold["NN_test_diracc"].mean(),
         "normalized": USE_NMSE},
    ])

    std_df = pd.DataFrame([
        {"Model": "OLS",
         "Train MSE": per_fold["OLS_train"].std(),
         "Val MSE":   per_fold["OLS_val"].std(),
         "Test MSE":  per_fold["OLS_test"].std(),
         "Train DirAcc": per_fold["OLS_train_diracc"].std(),
         "Val DirAcc":   per_fold["OLS_val_diracc"].std(),
         "Test DirAcc":  per_fold["OLS_test_diracc"].std(),
         "normalized": USE_NMSE},
        {"Model": "NN",
         "Train MSE": per_fold["NN_train"].std(),
         "Val MSE":   per_fold["NN_val"].std(),
         "Test MSE":  per_fold["NN_test"].std(),
         "Train DirAcc": per_fold["NN_train_diracc"].std(),
         "Val DirAcc":   per_fold["NN_val_diracc"].std(),
         "Test DirAcc":  per_fold["NN_test_diracc"].std(),
         "normalized": USE_NMSE},
    ]).rename(columns=lambda c: c if c in ["Model", "normalized"] else c + " Std")

    summary = avg_df.merge(std_df, on=["Model", "normalized"])

    # 6) Console + Save
    which = "NMSE" if USE_NMSE else "MSE"
    print_table(avg_df, title=f"Average across folds ({which})")

    out_dir = BASE / "analysis"
    out_dir.mkdir(exist_ok=True)
    per_fold.to_csv(out_dir / "per_fold_metrics.csv", index=False)
    avg_df.to_csv(out_dir / "fold_avg_metrics.csv", index=False)
    summary.to_csv(out_dir / "fold_avg_metrics_with_std.csv", index=False)

if __name__ == "__main__":
    main()
