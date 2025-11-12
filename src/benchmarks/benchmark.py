# --- Imports ---
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Any, List
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pipeline.walkforward import WFCVGenerator
from config.config_types import AppConfig
from utils.paths import DATA_DIR, PRICE_EXPERIMENTS_DIR, VOL_EXPERIMENTS_DIR

# =========================
# Config
# =========================

# Normalize MSE by target variance on each split
USE_NMSE = False
ACCURACY: int = 4


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
    print(f'y_tr.shape: {y_tr.shape}')
    if len(y_tr.shape) > 1:
        return [sm.OLS(y_tr[:, j], x_tr_c).fit() for j in range(y_tr.shape[1])]
    else:
        return [sm.OLS(y_tr, x_tr_c).fit()]

def pred_ols(models, x: np.ndarray) -> np.ndarray:
    x_c = add_const(x)
    return np.column_stack([m.predict(x_c) for m in models])

def dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional accuracy in percent across all dims & samples."""
    return float((np.sign(y_true) == np.sign(y_pred)).mean() * 100.0)

def undershooting(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """percentage of predictions lower than true all dims & samples."""
    return float(((y_true > y_pred)*1).mean() * 100.0)

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
    NAME  = "exp_018_lstm_100"
    for file in os.listdir(VOL_EXPERIMENTS_DIR/NAME):
        TRIAL = file
        BASE  = Path(VOL_EXPERIMENTS_DIR) / NAME / TRIAL
        print(f'\n\nAnalyzing {BASE}\n\n')

        cfg = load_cfg(BASE)

        # Read NN results.csv (unchanged)
        nn_path = BASE / "results.csv"
        use_cols = [
            "trial","fold",
            "tr_loss","val_loss","test_loss",
            "tr_mae","val_mae","test_mae",
            "tr_directional_accuracy_pct","val_directional_accuracy_pct","test_directional_accuracy_pct",
            "seconds","model_path",
        ]
        nn_df = pd.read_csv(nn_path, usecols=use_cols)

        # 1) Stream folds once: compute variances + OLS metrics on the fly (no caching)
        wf = WFCVGenerator(config=cfg.walkforward)
        var_rows, ols_rows = [], []

        for fold_idx, (X_tr, y_tr, X_val, y_val, X_te, y_te) in tqdm(
            enumerate(wf.folds()), desc="Streaming folds"
        ):
            # --- per-split variances (scalar) ---
            var_rows.append({
                "fold": fold_idx,
                "var_train": float(np.var(y_tr)),
                "var_val":   float(np.var(y_val)),
                "var_test":  float(np.var(y_te)),
            })

            # --- OLS per fold (keep your statsmodels helpers) ---
            models = fit_ols_per_target(X_tr, y_tr)

            yhat_tr_ols  = pred_ols(models, X_tr)
            yhat_val_ols = pred_ols(models, X_val)
            yhat_te_ols  = pred_ols(models, X_te)

            # ---- shape safety: if target is 1-D, make preds 1-D too ----
            if y_tr.ndim == 1:
                yhat_tr_ols = yhat_tr_ols.ravel()
            if y_val.ndim == 1:
                yhat_val_ols = yhat_val_ols.ravel()
            if y_te.ndim == 1:
                yhat_te_ols = yhat_te_ols.ravel()

            # ---- MSE computation that works for 1-D or multi-D ----
            def mse(a, b):
                diff2 = (a - b) ** 2
                if diff2.ndim == 2:   # average over dims & samples
                    return float(diff2.mean(axis=0).mean())
                return float(diff2.mean())

            mse_tr  = mse(y_tr,  yhat_tr_ols)
            mse_val = mse(y_val, yhat_val_ols)
            mse_te  = mse(y_te,  yhat_te_ols)

            da_tr  = dir_acc(y_tr,  yhat_tr_ols)
            da_val = dir_acc(y_val, yhat_val_ols)
            da_te  = dir_acc(y_te,  yhat_te_ols)

            us_tr  = undershooting(y_tr,  yhat_tr_ols)
            us_val = undershooting(y_val, yhat_val_ols)
            us_te  = undershooting(y_te,  yhat_te_ols)

            ols_rows.append({
                "fold": fold_idx,
                "OLS_train": mse_tr, "OLS_val": mse_val, "OLS_test": mse_te,
                "OLS_train_diracc": da_tr, "OLS_val_diracc": da_val, "OLS_test_diracc": da_te,
                "OLS_train_us": us_tr, "OLS_val_us": us_val, "OLS_test_us": us_te,
            })

            # ---- free big temporaries ASAP ----
            del X_tr, y_tr, X_val, y_val, X_te, y_te
            del yhat_tr_ols, yhat_val_ols, yhat_te_ols, models
            import gc; gc.collect()

        var_df = pd.DataFrame(var_rows)
        ols    = pd.DataFrame(ols_rows)
        ols = ols.merge(var_df, on="fold", how="left")
        
        # Normalize OLS
        if USE_NMSE:
            ols["OLS_train"] = ols["OLS_train"]   / ols["var_train"]
            ols["OLS_val"]   = ols["OLS_val"]  / ols["var_val"]
            ols["OLS_test"]  = ols["OLS_test"] / ols["var_test"]
        else:
            ols["OLS_train"] = ols["OLS_train"]
            ols["OLS_val"]   = ols["OLS_val"] 
            ols["OLS_test"]  = ols["OLS_test"]

        # 2) Merge NN with variances (unchanged logic)
        nn = nn_df.merge(var_df, on="fold", how="left")

        if USE_NMSE:
            nn["NN_train"] = nn["tr_loss"]   / nn["var_train"]
            nn["NN_val"]   = nn["val_loss"]  / nn["var_val"]
            nn["NN_test"]  = nn["test_loss"] / nn["var_test"]
        else:
            nn["NN_train"] = nn["tr_loss"]
            nn["NN_val"]   = nn["val_loss"]
            nn["NN_test"]  = nn["test_loss"]

        nn["NN_train_diracc"] = nn["tr_directional_accuracy_pct"]
        nn["NN_val_diracc"]   = nn["val_directional_accuracy_pct"]
        nn["NN_test_diracc"]  = nn["test_directional_accuracy_pct"]

        # 3) Per-fold comparison + normalized flag (unchanged)
        per_fold = ols.merge(
            nn[[
                "fold",
                "NN_train","NN_val","NN_test",
                "NN_train_diracc","NN_val_diracc","NN_test_diracc"
            ]],
            on="fold", how="left"
        )
        per_fold["normalized"] = USE_NMSE

        # 4) Aggregations (unchanged)
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

        # 5) Console + Save (unchanged)
        which = "NMSE" if USE_NMSE else "MSE"
        print_table(avg_df, title=f"Average across folds ({which})")

        out_dir = BASE / "analysis"
        out_dir.mkdir(exist_ok=True)
        per_fold.to_csv(out_dir / "per_fold_metrics.csv", index=False)
        avg_df.to_csv(out_dir / "fold_avg_metrics.csv", index=False)
        summary.to_csv(out_dir / "fold_avg_metrics_with_std.csv", index=False)

if __name__ == "__main__":
    main()
