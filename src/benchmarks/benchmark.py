# --- Imports ---
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Any, List
import os
import itertools
import hashlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from utils.logging_utils import ExperimentLogger
from pipeline.walkforward import WFCVGenerator
from config.config_types import AppConfig
from utils.paths import DATA_DIR, PRICE_EXPERIMENTS_DIR, VOL_EXPERIMENTS_DIR
from utils.inference_utils import format_legend_name
from models import create_model
from sklearn.linear_model import Lasso

# =========================
# Config
# =========================

# Normalize MSE by target variance on each split
USE_NMSE = True
MERGE_TRAIN_VAL = True
ACCURACY: int = 4
DEVICE = 'cuda'


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
    if len(y_tr.shape) > 1:
        return [sm.OLS(y_tr[:, j], x_tr_c).fit() for j in range(y_tr.shape[1])]
    else:
        return [sm.OLS(y_tr, x_tr_c).fit()]

def pred_ols(models, x: np.ndarray) -> np.ndarray:
    x_c = add_const(x)
    return np.column_stack([m.predict(x_c) for m in models])

def fit_lasso_per_target(x_tr: np.ndarray, y_tr: np.ndarray, alpha: float = 0.025):
    if len(y_tr.shape) > 1:
        return [Lasso(alpha=alpha).fit(x_tr, y_tr[:, j]) for j in range(y_tr.shape[1])]
    else:
        return [Lasso(alpha=alpha).fit(x_tr, y_tr)]

def pred_lasso(models, x: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(x) for m in models])

def dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional accuracy in percent across all dims & samples."""
    return float((np.sign(y_true) == np.sign(y_pred)).mean() * 100.0)

def undershooting(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """percentage of predictions lower than true all dims & samples."""
    return float(((y_true > y_pred)*1).mean() * 100.0)

def _infer_lstm_input_size_from_ckpt(state_dict: dict) -> int | None:
    # look for first LSTM weight_ih
    for k, v in state_dict.items():
        if k.endswith("lstm_layers.0.weight_ih_l0") or ("lstm_layers.0.weight_ih_l0" in k):
            # shape is [4*H, input_size]
            return int(v.shape[1])
    # legacy single-module naming (if any)
    for k, v in state_dict.items():
        if k.endswith("lstm.weight_ih_l0") or ("lstm.weight_ih_l0" in k):
            return int(v.shape[1])
    return None

def _make_input_shape_for_eval(cfg, X_sample: torch.Tensor | np.ndarray, state_dict: dict):
    name = cfg.model.name.lower()
    # infer T and (optional) D from the data
    if isinstance(X_sample, np.ndarray):
        shape = X_sample.shape
    else:
        shape = tuple(X_sample.shape)
    # shape is typically (N, T) or (N, T, D)
    if len(shape) == 2:
        _, T = shape
        D_data = 1
    elif len(shape) == 3:
        _, T, D_data = shape
    else:
        raise ValueError(f"Unexpected batch shape for X: {shape}")

    if name in ["lstm",'transformer']:
        D_ckpt = _infer_lstm_input_size_from_ckpt(state_dict)
        D = D_ckpt if D_ckpt is not None else D_data  # prefer ckpt
        return (T, D)
    elif name == "simplecnn":
        # your CNN expects (C, L) with C=1
        return (1, T)
    elif name == "mlp":
        # your MLP code expects (T,) as before (flattened window)
        return (T,)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

@torch.inference_mode()
def _predict_batched(model, X, device=DEVICE, bs=8192):
    preds = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i+bs], dtype=torch.float32, device=device)
        pb = model(xb).detach().cpu()
        preds.append(pb)
    return torch.cat(preds, dim=0).numpy()

def load_and_predict_nn(
    cfg,
    base_path: Path,
    fold_idx: int,
    X_test: np.ndarray,
    device=DEVICE,
    weights=None,
    normalize_weights: bool = True,
):
    """
    Load and predict with either:
      - a single model (cfg, base_path scalars), or
      - an ensemble (cfg, base_path lists of same length)

    Returns:
      yhat: np.ndarray of shape (N, D) or None
    """

    # ---- detect ensemble vs single ----
    is_ensemble = isinstance(cfg, (list, tuple))

    if not is_ensemble:
        cfgs = [cfg]
        base_paths = [base_path]
    else:
        cfgs = list(cfg)
        base_paths = list(base_path)
        assert len(cfgs) == len(base_paths), "cfg and base_path must have same length"

    K = len(cfgs)

    if weights is None:
        weights = np.ones(K, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        assert len(weights) == K, "weights must match number of models"

    if normalize_weights:
        weights = weights / weights.sum()

    preds = []

    # ---- loop over members ----
    for k, (cfg_k, base_k) in enumerate(zip(cfgs, base_paths)):
        try:
            ckpt_path = base_k / f"fold_{fold_idx:03d}" / "model_best.pt"
            if not ckpt_path.exists():
                continue

            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state"].items()
            }

            input_shape = _make_input_shape_for_eval(cfg_k, X_test, state_dict)
            output_shape = (
                cfg_k.walkforward.lookback + 1
                if cfg_k.walkforward.lookback is not None
                else 1
            )

            model = create_model(cfg_k.model, input_shape, output_shape)
            model.load_state_dict(state_dict, strict=True)
            model.to(device).eval()

            yhat = _predict_batched(model, X_test, device=device, bs=8192)

            # normalize shape: (N,) -> (N,1)
            yhat = np.asarray(yhat)
            if yhat.ndim == 1:
                yhat = yhat[:, None]

            preds.append((weights[k], yhat))

            # cleanup per model
            del model, checkpoint
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[WARN] Error loading/predicting member {k}: {e}")
            continue

    if len(preds) == 0:
        return None

    # ---- sanity check shapes ----
    shape0 = preds[0][1].shape
    for i, (_, p) in enumerate(preds[1:], start=1):
        if p.shape != shape0:
            raise ValueError(f"Member {i} has shape {p.shape}, expected {shape0}")

    # ---- weighted mean ----
    W = np.array([w for w, _ in preds], dtype=np.float64)
    P = np.stack([p for _, p in preds], axis=0)  # (K, N, D)

    if normalize_weights:
        W = W / W.sum()

    yhat = np.tensordot(W, P, axes=(0, 0))  # (N, D)

    return yhat


def print_table(df: pd.DataFrame, title: str):
    cols = [
        ("Model", "str"),
        ("Train MSE", "mse"), ("Val MSE", "mse"), ("Test MSE", "mse"),
        ("Train DirAcc", "pct"), ("Val DirAcc", "pct"), ("Test DirAcc", "pct"),
        ("Train US", "pct"), ("Val US", "pct"), ("Test US", "pct"),
        ("Alpha Train", "coef"), ("Beta Train", "coef"), ("Calib R2 Train", "coef"),
        ("Alpha", "coef"), ("Beta", "coef"), ("Calib R2 Test", "coef"),
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
                row.append(f"{float(r[c]):>13.{ACCURACY}f}%")
            elif t == "coef":
                row.append(f"{float(r[c]):>14.{ACCURACY}f}")
            else:
                row.append(f"{str(r[c]):>14}")
        print("\t".join(row))


def _dedupe_on_fold(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Ensure unique 'fold' keys; drop duplicates if any and warn."""
    if "fold" not in df.columns:
        return df
    if not df["fold"].is_unique:
        dup = df["fold"][df["fold"].duplicated(keep=False)]
        print(f"[WARN] {name}: duplicate folds detected -> {sorted(dup.tolist())}")
        df = (
            df.sort_values(["fold"])
              .drop_duplicates(subset=["fold"], keep="last")
              .reset_index(drop=True)
        )
        print(f"[INFO] {name}: deduped to {len(df)} unique folds")
    return df

def _compute_nn_calibration(cfg, base, fold_idx, Xtr, ytr, Xv, yv, Xte, yte, Xtr_val, ytr_val, Xte_merged, yte_merged):
    """
    Extract NN calibration computation into a helper function for cleaner code.
    """
    if MERGE_TRAIN_VAL:
        X_tr, y_tr = Xtr_val, ytr_val
        X_val, y_val = None, None
        X_te, y_te = Xte_merged, yte_merged
    else:
        X_tr, y_tr = Xtr, ytr
        X_val, y_val = Xv, yv
        X_te, y_te = Xte, yte

    def mse(a, b):
        diff2 = (a - b) ** 2
        if diff2.ndim == 2:
            return float(diff2.mean(axis=0).mean())
        return float(diff2.mean())

    yhat_te_nn = load_and_predict_nn(cfg, base, fold_idx, X_te, device=DEVICE)
    
    if yhat_te_nn is not None:
        if y_te.ndim == 1 and yhat_te_nn.ndim > 1:
            yhat_te_nn = yhat_te_nn.ravel()
        elif y_te.ndim > 1 and yhat_te_nn.ndim == 1:
            yhat_te_nn = yhat_te_nn.reshape(-1, 1)
        
        mse_te_nn = mse(y_te, yhat_te_nn)
        da_te_nn = dir_acc(y_te, yhat_te_nn)
        us_te_nn = undershooting(y_te, yhat_te_nn)
        
        y_te_flat = y_te.reshape(-1) if y_te.ndim > 1 else y_te
        yhat_te_nn_f = yhat_te_nn.reshape(-1) if yhat_te_nn.ndim > 1 else yhat_te_nn
        
        X_cal_nn = add_const(yhat_te_nn_f)
        cal_nn   = sm.OLS(y_te_flat, X_cal_nn).fit()
        alpha_nn_test = float(cal_nn.params[0])
        beta_nn_test  = float(cal_nn.params[1])
        r2_nn_test    = float(cal_nn.rsquared)
        
        if MERGE_TRAIN_VAL:
            yhat_tr_nn = load_and_predict_nn(cfg, base, fold_idx, X_tr, device=DEVICE)
            if yhat_tr_nn is not None:
                if y_tr.ndim == 1 and yhat_tr_nn.ndim > 1:
                    yhat_tr_nn = yhat_tr_nn.ravel()
                elif y_tr.ndim > 1 and yhat_tr_nn.ndim == 1:
                    yhat_tr_nn = yhat_tr_nn.reshape(-1, 1)
                
                mse_tr_nn = mse(y_tr, yhat_tr_nn)
                da_tr_nn = dir_acc(y_tr, yhat_tr_nn)
                us_tr_nn = undershooting(y_tr, yhat_tr_nn)
                mse_val_nn = np.nan
                da_val_nn = np.nan
                us_val_nn = np.nan

                y_tr_flat    = y_tr.reshape(-1) if y_tr.ndim > 1 else y_tr
                yhat_tr_nn_f = yhat_tr_nn.reshape(-1) if yhat_tr_nn.ndim > 1 else yhat_tr_nn
                cal_nn_tr    = sm.OLS(y_tr_flat, add_const(yhat_tr_nn_f)).fit()
                alpha_nn_train = float(cal_nn_tr.params[0])
                beta_nn_train  = float(cal_nn_tr.params[1])
                r2_nn_train    = float(cal_nn_tr.rsquared)

                del yhat_tr_nn
            else:
                mse_tr_nn = da_tr_nn = mse_val_nn = da_val_nn = np.nan
                alpha_nn_train = beta_nn_train = r2_nn_train = np.nan
        else:
            yhat_tr_nn = load_and_predict_nn(cfg, base, fold_idx, X_tr, device=DEVICE)
            yhat_val_nn = load_and_predict_nn(cfg, base, fold_idx, X_val, device=DEVICE)
            
            if yhat_tr_nn is not None:
                if y_tr.ndim == 1 and yhat_tr_nn.ndim > 1:
                    yhat_tr_nn = yhat_tr_nn.ravel()
                elif y_tr.ndim > 1 and yhat_tr_nn.ndim == 1:
                    yhat_tr_nn = yhat_tr_nn.reshape(-1, 1)
                mse_tr_nn = mse(y_tr, yhat_tr_nn)
                da_tr_nn = dir_acc(y_tr, yhat_tr_nn)
                us_tr_nn = undershooting(y_tr, yhat_tr_nn)
                
                y_tr_flat    = y_tr.reshape(-1) if y_tr.ndim > 1 else y_tr
                yhat_tr_nn_f = yhat_tr_nn.reshape(-1) if yhat_tr_nn.ndim > 1 else yhat_tr_nn
                cal_nn_tr    = sm.OLS(y_tr_flat, add_const(yhat_tr_nn_f)).fit()
                alpha_nn_train = float(cal_nn_tr.params[0])
                beta_nn_train  = float(cal_nn_tr.params[1])
                r2_nn_train    = float(cal_nn_tr.rsquared)

                del yhat_tr_nn
            else:
                mse_tr_nn = da_tr_nn = np.nan
                alpha_nn_train = beta_nn_train = r2_nn_train = np.nan
            
            if yhat_val_nn is not None:
                if y_val.ndim == 1 and yhat_val_nn.ndim > 1:
                    yhat_val_nn = yhat_val_nn.ravel()
                elif y_val.ndim > 1 and yhat_val_nn.ndim == 1:
                    yhat_val_nn = yhat_val_nn.reshape(-1, 1)
                mse_val_nn = mse(y_val, yhat_val_nn)
                da_val_nn = dir_acc(y_val, yhat_val_nn)
                us_val_nn = undershooting(y_val, yhat_val_nn)
                
                del yhat_val_nn
            else:
                mse_val_nn = da_val_nn = us_val_nn = np.nan
    else:
        alpha_nn_train = beta_nn_train = r2_nn_train = np.nan
        alpha_nn_test = beta_nn_test = r2_nn_test = np.nan
        mse_tr_nn = mse_val_nn = mse_te_nn = np.nan
        da_tr_nn = da_val_nn = da_te_nn = np.nan
        us_tr_nn = us_val_nn = us_te_nn = np.nan
    
    return {
        "fold": fold_idx,
        "NN_alpha_train": alpha_nn_train,
        "NN_beta_train":  beta_nn_train,
        "NN_r2_train":    r2_nn_train,
        "NN_alpha":       alpha_nn_test,
        "NN_beta":        beta_nn_test,
        "NN_r2_test":     r2_nn_test,
        "NN_train_computed": mse_tr_nn,
        "NN_val_computed":   mse_val_nn,
        "NN_test_computed":  mse_te_nn,
        "NN_train_diracc_computed": da_tr_nn,
        "NN_val_diracc_computed":   da_val_nn,
        "NN_test_diracc_computed":  da_te_nn,
        "NN_train_us": us_tr_nn,
        "NN_val_us":   us_val_nn,
        "NN_test_us":  us_te_nn,
    }


# Fold data cache: maps fold signature -> (ols_row, lasso_row, var_row)
FOLD_DATA_CACHE = {}

def _hash_fold_data(Xtr, ytr, Xv, yv, Xte, yte, Xtr_val, ytr_val, Xte_merged, yte_merged):
    """
    Create a deterministic hash of fold data to detect duplicates.
    Uses shape + first/last few elements to avoid hashing huge arrays.
    """
    def _quick_hash(arr):
        arr = np.asarray(arr)
        sample = np.concatenate([arr.ravel()[:10], arr.ravel()[-10:]])
        return hashlib.md5(sample.astype(np.float32).tobytes()).hexdigest()
    
    sig = (
        _quick_hash(Xtr), _quick_hash(ytr),
        _quick_hash(Xv), _quick_hash(yv),
        _quick_hash(Xte), _quick_hash(yte),
        _quick_hash(Xtr_val), _quick_hash(ytr_val),
        _quick_hash(Xte_merged), _quick_hash(yte_merged),
    )
    return hashlib.md5(str(sig).encode()).hexdigest()


# =========================
# Main
# =========================
TRIAL = 'trial_search_best'
def main():
    names = [
    'exp_200_transformer_100_sgd_v2'
]

    #comb2 = list(itertools.combinations(names, 2))
    #comb3 = list(itertools.combinations(all_names, 3))
    #names = comb2

    for i, ITEM in tqdm(enumerate(names)):

        # Ensemble
        if isinstance(ITEM, (list, tuple)):
            print('Analyzing ensemble')
            # Lists to convert to store variables and pass to predict ensemble function 
            cfg_ensemble_list = []
            base_list = []
            for member in ITEM:
                # define varibales per model
                base_member  = Path(VOL_EXPERIMENTS_DIR) / member / TRIAL
                cfg_member = load_cfg(base_member)
                
                cfg_ensemble_list.append(cfg_member)
                base_list.append(base_member)

            # change names to variables
            # Very ugly but works
            base = base_list
            cfg = cfg_ensemble_list
            print(f'\n\nAnalyzing {ITEM}\n\n')

            # Instantiate with the last one, assuming all have the same structure
            cfg_loading = cfg[-1]

            # change names
            clean_names = [format_legend_name(member) for member in ITEM]
            concatenated = "_".join(clean_names)
            concatenated = concatenated.replace(" ", "").lower()
            cfg_loading.experiment.name = f'ensemble_{concatenated}'
            
            wf = WFCVGenerator(config=cfg_loading.walkforward)
            logger = ExperimentLogger(cfg_loading) 
            out_dir_ensemble = logger.begin_trial()

            
            
        else:
            base  = Path(VOL_EXPERIMENTS_DIR) / ITEM / TRIAL
            if not base.exists():
                parent = base.parent  # Path(VOL_EXPERIMENTS_DIR) / ITEM
                subdirs = sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.name)
                if not subdirs:
                    raise FileNotFoundError(f"No subdirectories found in {parent}")
                base = subdirs[-1]  # last subdir alphabetically
            
            print(f'\n\nAnalyzing {base}\n\n')
            cfg = load_cfg(base)
            wf = WFCVGenerator(config=cfg.walkforward)

        # 1) Stream folds once: compute variances + OLS metrics + NN calibration on the fly
        var_rows, ols_rows, lasso_rows, nn_calib_rows = [], [], [], []

        for fold_idx, (Xtr, ytr, Xv, yv, Xte, yte, Xtr_val, ytr_val, Xte_merged, yte_merged, id_tr, id_v, id_te, windows_tr, windows_te, windows_v) in enumerate(wf.folds()):

            # --- CACHE GUARD: Check if we've seen this fold data before ---
            fold_sig = _hash_fold_data(Xtr, ytr, Xv, yv, Xte, yte, Xtr_val, ytr_val, Xte_merged, yte_merged)
            
            if fold_sig in FOLD_DATA_CACHE:
                print(f"[CACHE HIT] Fold {fold_idx}: Loading OLS/LASSO results from cache")
                cached_var, cached_ols, cached_lasso = FOLD_DATA_CACHE[fold_sig]
                var_rows.append(cached_var)
                ols_rows.append(cached_ols)
                lasso_rows.append(cached_lasso)
                
                # Still need to compute NN calibration (not cached, model-specific)
                nn_calib_row = _compute_nn_calibration(cfg, base, fold_idx, Xtr, ytr, Xv, yv, Xte, yte, Xtr_val, ytr_val, Xte_merged, yte_merged)
                nn_calib_rows.append(nn_calib_row)
                
                # Free memory
                del Xtr, ytr, Xv, yv, Xte, yte, Xtr_val, ytr_val, Xte_merged, yte_merged
                import gc; gc.collect()
                continue
         
            # --- CACHE MISS: Compute OLS, LASSO, and variances ---
            print(f"[CACHE MISS] Fold {fold_idx}: Computing OLS/LASSO")

            if MERGE_TRAIN_VAL:
                # Use pre-scaled merged arrays from new schema
                X_tr, y_tr = Xtr_val, ytr_val
                X_val, y_val = None, None  # Not used in merged mode
                X_te, y_te = Xte_merged, yte_merged
            else:
                # Use original separate arrays
                X_tr, y_tr = Xtr, ytr
                X_val, y_val = Xv, yv
                X_te, y_te = Xte, yte

            # --- per-split variances (scalar) ---
            var_row = {
                "fold": fold_idx,
                "var_train": float(np.var(y_tr)),
                "var_val":   float(np.var(y_val)) if not MERGE_TRAIN_VAL else np.nan,
                "var_test":  float(np.var(y_te)),
            }
            var_rows.append(var_row)

            # --- OLS per fold (keep your statsmodels helpers) ---

            models = fit_ols_per_target(X_tr, y_tr)

            yhat_tr_ols  = pred_ols(models, X_tr)
            if MERGE_TRAIN_VAL:
                yhat_val_ols = None
            else:
                yhat_val_ols = pred_ols(models, X_val)
            yhat_te_ols  = pred_ols(models, X_te)

            # ---- shape safety: if target is 1-D, make preds 1-D too ----
            if y_tr.ndim == 1:
                yhat_tr_ols = yhat_tr_ols.ravel()
            if y_val is not None and not MERGE_TRAIN_VAL:
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
            mse_val = np.nan if MERGE_TRAIN_VAL else mse(y_val, yhat_val_ols)
            mse_te  = mse(y_te,  yhat_te_ols)

            da_tr  = dir_acc(y_tr,  yhat_tr_ols)
            da_val = np.nan if MERGE_TRAIN_VAL else dir_acc(y_val, yhat_val_ols)
            da_te  = dir_acc(y_te,  yhat_te_ols)

            us_tr  = undershooting(y_tr,  yhat_tr_ols)
            us_val = np.nan if MERGE_TRAIN_VAL else undershooting(y_val, yhat_val_ols)
            us_te  = undershooting(y_te,  yhat_te_ols)

            # ---- OLS calibration on TEST: y_true ~ alpha + beta * y_pred_ols ----
            y_tr_flat     = y_tr.reshape(-1) if y_tr.ndim > 1 else y_tr
            yhat_tr_ols_f = yhat_tr_ols.reshape(-1) if yhat_tr_ols.ndim > 1 else yhat_tr_ols

            y_te_flat     = y_te.reshape(-1) if y_te.ndim > 1 else y_te
            yhat_te_ols_f = yhat_te_ols.reshape(-1) if yhat_te_ols.ndim > 1 else yhat_te_ols

            cal_ols_tr = sm.OLS(y_tr_flat, add_const(yhat_tr_ols_f)).fit()
            alpha_ols_train = float(cal_ols_tr.params[0])
            beta_ols_train  = float(cal_ols_tr.params[1])
            r2_ols_train    = float(cal_ols_tr.rsquared)

            cal_ols_te = sm.OLS(y_te_flat, add_const(yhat_te_ols_f)).fit()
            alpha_ols_test = float(cal_ols_te.params[0])
            beta_ols_test  = float(cal_ols_te.params[1])
            r2_ols_test    = float(cal_ols_te.rsquared)

            print('\nOLS analysis complete')
            ols_row = {
                "fold": fold_idx,
                "OLS_train": mse_tr, "OLS_val": mse_val, "OLS_test": mse_te,
                "OLS_train_diracc": da_tr, "OLS_val_diracc": da_val, "OLS_test_diracc": da_te,
                "OLS_train_us": us_tr, "OLS_val_us": us_val, "OLS_test_us": us_te,
                "OLS_alpha_train": alpha_ols_train,
                "OLS_beta_train":  beta_ols_train,
                "OLS_r2_train":    r2_ols_train,
                "OLS_alpha":       alpha_ols_test,
                "OLS_beta":        beta_ols_test,
                "OLS_r2_test":     r2_ols_test,
            }
            ols_rows.append(ols_row)

            # --- LASSO per fold ---
            # Try different alphas and pick best on validation
            if MERGE_TRAIN_VAL:
                # No validation set, just use a default alpha
                lasso_models = fit_lasso_per_target(X_tr, y_tr, alpha=0.05)
            else:
                # Simple grid search over alphas
                best_alpha = 0.05
                best_val_mse = float('inf')
                for alpha in tqdm([0.001, 0.01, 0.025, 0.05, 0.1, 0.25], desc="fitting lasso"):
                    temp_models = fit_lasso_per_target(X_tr, y_tr, alpha=alpha)
                    temp_preds = pred_lasso(temp_models, X_val)
                    if y_val.ndim == 1:
                        temp_preds = temp_preds.ravel()
                    temp_mse = mse(y_val, temp_preds)
                    if temp_mse < best_val_mse:
                        best_val_mse = temp_mse
                        best_alpha = alpha
                lasso_models = fit_lasso_per_target(X_tr, y_tr, alpha=best_alpha)

            yhat_tr_lasso = pred_lasso(lasso_models, X_tr)
            if MERGE_TRAIN_VAL:
                yhat_val_lasso = None
            else:
                yhat_val_lasso = pred_lasso(lasso_models, X_val)
            yhat_te_lasso = pred_lasso(lasso_models, X_te)

            # Shape safety for LASSO
            if y_tr.ndim == 1:
                yhat_tr_lasso = yhat_tr_lasso.ravel()
            if y_val is not None and not MERGE_TRAIN_VAL:
                if y_val.ndim == 1:
                    yhat_val_lasso = yhat_val_lasso.ravel()
            if y_te.ndim == 1:
                yhat_te_lasso = yhat_te_lasso.ravel()

            # LASSO metrics
            mse_tr_lasso  = mse(y_tr, yhat_tr_lasso)
            mse_val_lasso = np.nan if MERGE_TRAIN_VAL else mse(y_val, yhat_val_lasso)
            mse_te_lasso  = mse(y_te, yhat_te_lasso)

            da_tr_lasso  = dir_acc(y_tr, yhat_tr_lasso)
            da_val_lasso = np.nan if MERGE_TRAIN_VAL else dir_acc(y_val, yhat_val_lasso)
            da_te_lasso  = dir_acc(y_te, yhat_te_lasso)

            us_tr_lasso  = undershooting(y_tr, yhat_tr_lasso)
            us_val_lasso = np.nan if MERGE_TRAIN_VAL else undershooting(y_val, yhat_val_lasso)
            us_te_lasso  = undershooting(y_te, yhat_te_lasso)

            # LASSO calibration
            yhat_tr_lasso_f = yhat_tr_lasso.reshape(-1) if yhat_tr_lasso.ndim > 1 else yhat_tr_lasso
            yhat_te_lasso_f = yhat_te_lasso.reshape(-1) if yhat_te_lasso.ndim > 1 else yhat_te_lasso

            cal_lasso_tr = sm.OLS(y_tr_flat, add_const(yhat_tr_lasso_f)).fit()
            alpha_lasso_train = float(cal_lasso_tr.params[0])
            beta_lasso_train  = float(cal_lasso_tr.params[1])
            r2_lasso_train    = float(cal_lasso_tr.rsquared)

            cal_lasso_te = sm.OLS(y_te_flat, add_const(yhat_te_lasso_f)).fit()
            alpha_lasso_test = float(cal_lasso_te.params[0])
            beta_lasso_test  = float(cal_lasso_te.params[1])
            r2_lasso_test    = float(cal_lasso_te.rsquared)

            print('\nLASSO analysis complete')
            lasso_row = {
                "fold": fold_idx,
                "LASSO_train": mse_tr_lasso, "LASSO_val": mse_val_lasso, "LASSO_test": mse_te_lasso,
                "LASSO_train_diracc": da_tr_lasso, "LASSO_val_diracc": da_val_lasso, "LASSO_test_diracc": da_te_lasso,
                "LASSO_train_us": us_tr_lasso, "LASSO_val_us": us_val_lasso, "LASSO_test_us": us_te_lasso,
                "LASSO_alpha_train": alpha_lasso_train,
                "LASSO_beta_train":  beta_lasso_train,
                "LASSO_r2_train":    r2_lasso_train,
                "LASSO_alpha":       alpha_lasso_test,
                "LASSO_beta":        beta_lasso_test,
                "LASSO_r2_test":     r2_lasso_test,
            }
            lasso_rows.append(lasso_row)

            # --- Store in cache for future lookups ---
            FOLD_DATA_CACHE[fold_sig] = (var_row, ols_row, lasso_row)


            # ---- NN calibration and metrics on TEST: load model and compute everything ----
            nn_calib_row = _compute_nn_calibration(cfg, base, fold_idx, Xtr, ytr, Xv, yv, Xte, yte, Xtr_val, ytr_val, Xte_merged, yte_merged)
            nn_calib_rows.append(nn_calib_row)
            
            # ---- free big temporaries ASAP ----
            del X_tr, y_tr, X_te, y_te
            if not MERGE_TRAIN_VAL:
                del X_val, y_val, yhat_val_ols
            del yhat_tr_ols, yhat_te_ols, models
            import gc; gc.collect()

        var_df = pd.DataFrame(var_rows)
        ols    = pd.DataFrame(ols_rows)
        lasso    = pd.DataFrame(lasso_rows)
        nn_calib = pd.DataFrame(nn_calib_rows)

        # --- Debug/guard: ensure unique fold keys before any merge ---
        var_df   = _dedupe_on_fold(var_df,   "var_df")
        lasso      = _dedupe_on_fold(lasso,      "lasso")
        ols      = _dedupe_on_fold(ols,      "ols")
        nn_calib = _dedupe_on_fold(nn_calib, "nn_calib")


        ols = ols.merge(var_df, on="fold", how="left")

        # Normalize OLS
        if USE_NMSE:
            ols["OLS_train"] = ols["OLS_train"] / ols["var_train"]
            if not MERGE_TRAIN_VAL:
                ols["OLS_val"] = ols["OLS_val"] / ols["var_val"]
            else:
                ols["OLS_val"] = np.nan  # No validation in merged mode
            ols["OLS_test"] = ols["OLS_test"] / ols["var_test"]
        else:
            ols["OLS_train"] = ols["OLS_train"]
            if not MERGE_TRAIN_VAL:
                ols["OLS_val"] = ols["OLS_val"]
            else:
                ols["OLS_val"] = np.nan  # No validation in merged mode
            ols["OLS_test"] = ols["OLS_test"]

        lasso = lasso.merge(var_df, on="fold", how="left")

        # Normalize LASSO
        if USE_NMSE:
            lasso["LASSO_train"] = lasso["LASSO_train"] / lasso["var_train"]
            if not MERGE_TRAIN_VAL:
                lasso["LASSO_val"] = lasso["LASSO_val"] / lasso["var_val"]
            else:
                lasso["LASSO_val"] = np.nan  # No validation in merged mode
            lasso["LASSO_test"] = lasso["LASSO_test"] / lasso["var_test"]
        else:
            lasso["LASSO_train"] = lasso["LASSO_train"]
            if not MERGE_TRAIN_VAL:
                lasso["LASSO_val"] = lasso["LASSO_val"]
            else:
                lasso["LASSO_val"] = np.nan  # No validation in merged mode
            lasso["LASSO_test"] = lasso["LASSO_test"]

        # 2) Merge NN with variances and calibration data, use computed metrics
        nn = nn_calib.merge(var_df, on="fold", how="left", validate="one_to_one")

        # Use computed NN metrics instead of results.csv when available
        if USE_NMSE:
            nn["NN_train"] = nn["NN_train_computed"] / nn["var_train"]
            nn["NN_val"]   = np.nan if MERGE_TRAIN_VAL else nn["NN_val_computed"] / nn["var_val"]
            nn["NN_test"]  = nn["NN_test_computed"] / nn["var_test"]
        else:
            nn["NN_train"] = nn["NN_train_computed"]
            nn["NN_val"]   = np.nan if MERGE_TRAIN_VAL else nn["NN_val_computed"]
            nn["NN_test"]  = nn["NN_test_computed"]

        # Use computed directional accuracy
        nn["NN_train_diracc"] = nn["NN_train_diracc_computed"]
        nn["NN_val_diracc"]   = np.nan if MERGE_TRAIN_VAL else nn["NN_val_diracc_computed"]
        nn["NN_test_diracc"]  = nn["NN_test_diracc_computed"]

        # 3) Per-fold comparison + normalized flag (now includes calibration metrics)
        assert ols["fold"].is_unique, "OLS has duplicate folds before merge"
        assert nn["fold"].is_unique,  "NN has duplicate folds before merge"

        per_fold = ols.merge(
            lasso[[
                "fold",
                "LASSO_train","LASSO_val","LASSO_test",
                "LASSO_train_diracc","LASSO_val_diracc","LASSO_test_diracc",
                "LASSO_alpha_train","LASSO_beta_train","LASSO_r2_train",
                "LASSO_alpha","LASSO_beta","LASSO_r2_test",
                "LASSO_train_us","LASSO_val_us","LASSO_test_us"
            ]],
            on="fold", how="left"
        ).merge(
            nn[[
                "fold",
                "NN_train","NN_val","NN_test",
                "NN_train_diracc","NN_val_diracc","NN_test_diracc",
                "NN_alpha_train","NN_beta_train","NN_r2_train",
                "NN_alpha","NN_beta","NN_r2_test",
                "NN_train_us","NN_val_us","NN_test_us"
            ]],
            on="fold", how="left"
        )
        per_fold["normalized"] = USE_NMSE

        # 4) Aggregations (now includes calibration metrics)
        avg_df = pd.DataFrame([
            {"Model": "OLS",
            "Train MSE": per_fold["OLS_train"].mean(),
            "Val MSE":   per_fold["OLS_val"].mean(),
            "Test MSE":  per_fold["OLS_test"].mean(),
            "Train DirAcc": per_fold["OLS_train_diracc"].mean(),
            "Val DirAcc":   per_fold["OLS_val_diracc"].mean(),
            "Test DirAcc":  per_fold["OLS_test_diracc"].mean(),
            "Train US": per_fold["OLS_train_us"].mean(),
            "Val US":   per_fold["OLS_val_us"].mean(),
            "Test US":  per_fold["OLS_test_us"].mean(),
            "Alpha Train": per_fold["OLS_alpha_train"].mean(),
            "Beta Train":  per_fold["OLS_beta_train"].mean(),
            "Calib R2 Train": per_fold["OLS_r2_train"].mean(),
            "Alpha": per_fold["OLS_alpha"].mean(),
            "Beta":  per_fold["OLS_beta"].mean(),
            "Calib R2 Test": per_fold["OLS_r2_test"].mean(),
            "normalized": USE_NMSE},
            {"Model": "LASSO",
            "Train MSE": per_fold["LASSO_train"].mean(),
            "Val MSE":   per_fold["LASSO_val"].mean(),
            "Test MSE":  per_fold["LASSO_test"].mean(),
            "Train DirAcc": per_fold["LASSO_train_diracc"].mean(),
            "Val DirAcc":   per_fold["LASSO_val_diracc"].mean(),
            "Test DirAcc":  per_fold["LASSO_test_diracc"].mean(),
            "Train US": per_fold["LASSO_train_us"].mean(),
            "Val US":   per_fold["LASSO_val_us"].mean(),
            "Test US":  per_fold["LASSO_test_us"].mean(),
            "Alpha Train": per_fold["LASSO_alpha_train"].mean(),
            "Beta Train":  per_fold["LASSO_beta_train"].mean(),
            "Calib R2 Train": per_fold["LASSO_r2_train"].mean(),
            "Alpha": per_fold["LASSO_alpha"].mean(),
            "Beta":  per_fold["LASSO_beta"].mean(),
            "Calib R2 Test": per_fold["LASSO_r2_test"].mean(),
            "normalized": USE_NMSE},
            {"Model": "NN",
            "Train MSE": per_fold["NN_train"].mean(),
            "Val MSE":   per_fold["NN_val"].mean(),
            "Test MSE":  per_fold["NN_test"].mean(),
            "Train DirAcc": per_fold["NN_train_diracc"].mean(),
            "Val DirAcc":   per_fold["NN_val_diracc"].mean(),
            "Test DirAcc":  per_fold["NN_test_diracc"].mean(),
            "Train US": per_fold["NN_train_us"].mean(),
            "Val US":   per_fold["NN_val_us"].mean(),
            "Test US":  per_fold["NN_test_us"].mean(),
            "Alpha Train": per_fold["NN_alpha_train"].mean(),
            "Beta Train":  per_fold["NN_beta_train"].mean(),
            "Calib R2 Train": per_fold["NN_r2_train"].mean(),
            "Alpha": per_fold["NN_alpha"].mean(),
            "Beta":  per_fold["NN_beta"].mean(),
            "Calib R2 Test": per_fold["NN_r2_test"].mean(),
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
            "Train US": per_fold["OLS_train_us"].std(),
            "Val US":   per_fold["OLS_val_us"].std(),
            "Test US":  per_fold["OLS_test_us"].std(),
            "Alpha Train": per_fold["OLS_alpha_train"].std(),
            "Beta Train":  per_fold["OLS_beta_train"].std(),
            "Calib R2 Train": per_fold["OLS_r2_train"].std(),
            "Alpha": per_fold["OLS_alpha"].std(),
            "Beta":  per_fold["OLS_beta"].std(),
            "Calib R2 Test": per_fold["OLS_r2_test"].std(),
            "normalized": USE_NMSE},
            {"Model": "LASSO",
            "Train MSE": per_fold["LASSO_train"].std(),
            "Val MSE":   per_fold["LASSO_val"].std(),
            "Test MSE":  per_fold["LASSO_test"].std(),
            "Train DirAcc": per_fold["LASSO_train_diracc"].std(),
            "Val DirAcc":   per_fold["LASSO_val_diracc"].std(),
            "Test DirAcc":  per_fold["LASSO_test_diracc"].std(),
            "Train US": per_fold["LASSO_train_us"].std(),
            "Val US":   per_fold["LASSO_val_us"].std(),
            "Test US":  per_fold["LASSO_test_us"].std(),
            "Alpha Train": per_fold["LASSO_alpha_train"].std(),
            "Beta Train":  per_fold["LASSO_beta_train"].std(),
            "Calib R2 Train": per_fold["LASSO_r2_train"].std(),
            "Alpha": per_fold["LASSO_alpha"].std(),
            "Beta":  per_fold["LASSO_beta"].std(),
            "Calib R2 Test": per_fold["LASSO_r2_test"].std(),
            "normalized": USE_NMSE},
            {"Model": "NN",
            "Train MSE": per_fold["NN_train"].std(),
            "Val MSE":   per_fold["NN_val"].std(),
            "Test MSE":  per_fold["NN_test"].std(),
            "Train DirAcc": per_fold["NN_train_diracc"].std(),
            "Val DirAcc":   per_fold["NN_val_diracc"].std(),
            "Test DirAcc":  per_fold["NN_test_diracc"].std(),
            "Train US": per_fold["NN_train_us"].std(),
            "Val US":   per_fold["NN_val_us"].std(),
            "Test US":  per_fold["NN_test_us"].std(),
            "Alpha Train": per_fold["NN_alpha_train"].std(),
            "Beta Train":  per_fold["NN_beta_train"].std(),
            "Calib R2 Train": per_fold["NN_r2_train"].std(),
            "Alpha": per_fold["NN_alpha"].std(),
            "Beta":  per_fold["NN_beta"].std(),
            "Calib R2 Test": per_fold["NN_r2_test"].std(),
            "normalized": USE_NMSE},
        ]).rename(columns=lambda c: c if c in ["Model", "normalized"] else c + " Std")
        
        summary = avg_df.merge(std_df, on=["Model", "normalized"])

        # 5) Console + Save (unchanged)
        which = "NMSE" if USE_NMSE else "MSE"
        print_table(avg_df, title=f"Average across folds ({which})")


        if not isinstance(base, (list, tuple)):
            out_dir = base / "analysis" 
        else:
            out_dir = Path(out_dir_ensemble) / 'analysis'

        out_dir.mkdir(exist_ok=True)
        per_fold.to_csv(out_dir / "per_fold_metrics.csv", index=False)
        avg_df.to_csv(out_dir / "fold_avg_metrics.csv", index=False)
        summary.to_csv(out_dir / "fold_avg_metrics_with_std.csv", index=False)

if __name__ == "__main__":
    main()
