import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


def scale_split(
    Xtr, ytr, Xv, yv, Xte, yte,
    scale_type: str = "standard",
    merge: bool = False,
    dtype=np.float64
):
    """
    Scale train/val/test splits with optional merging of train+val.
    
    Args:
        Xtr, ytr: Training data
        Xv, yv: Validation data  
        Xte, yte: Test data
        scale_type: Type of scaling ('standard', 'robust', 'asinh', etc.)
        merge: If True, also return merged train+val with test scaled on merged data
        dtype: Output data type
        
    Returns:
        If merge=False: (Xtr_scaled, ytr_scaled, Xv_scaled, yv_scaled, Xte_scaled, yte_scaled)
        If merge=True: (Xtr_scaled, ytr_scaled, Xv_scaled, yv_scaled, Xte_scaled, yte_scaled,
                        Xtr_val_scaled, ytr_val_scaled, Xte_merged_scaled, yte_merged_scaled)
    """
    
    # --- Sanity checks ---
    for name, arr in [("Xtr", Xtr), ("Xv", Xv), ("Xte", Xte), ("ytr", ytr), ("yv", yv), ("yte", yte)]:
        if arr is None:
            raise ValueError(f"{name} is None — expected ndarray.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or Inf values.")
    
    scale_type = scale_type.lower()
    
    # --- Scale X ---
    X_scaler = None
    if scale_type == "standard":
        X_scaler = StandardScaler(copy=False)
        Xtr_s = X_scaler.fit_transform(Xtr)
        Xv_s = X_scaler.transform(Xv)
        Xte_s = X_scaler.transform(Xte)
    elif scale_type == "robust":
        X_scaler = RobustScaler(quantile_range=(25, 75))
        Xtr_s = X_scaler.fit_transform(Xtr)
        Xv_s = X_scaler.transform(Xv)
        Xte_s = X_scaler.transform(Xte)
    elif scale_type == "asinh":
        Xtr_s = np.arcsinh(Xtr)
        Xv_s = np.arcsinh(Xv)
        Xte_s = np.arcsinh(Xte)
    elif scale_type == "log":
        Xtr_s = np.log(Xtr)
        Xv_s = np.log(Xv)
        Xte_s = np.log(Xte)
    elif scale_type == "log1p":
        Xtr_s = np.log1p(Xtr)
        Xv_s = np.log1p(Xv)
        Xte_s = np.log1p(Xte)
    elif scale_type == "asinhstandard":
        Xtr_s = np.arcsinh(Xtr)
        Xv_s = np.arcsinh(Xv)
        Xte_s = np.arcsinh(Xte)
        X_scaler = StandardScaler(copy=False)
        Xtr_s = X_scaler.fit_transform(Xtr_s)
        Xv_s = X_scaler.transform(Xv_s)
        Xte_s = X_scaler.transform(Xte_s)
    elif scale_type == "log1pstandard":
        Xtr_s = np.log1p(Xtr)
        Xv_s = np.log1p(Xv)
        Xte_s = np.log1p(Xte)
        X_scaler = StandardScaler(copy=False)
        Xtr_s = X_scaler.fit_transform(Xtr_s)
        Xv_s = X_scaler.transform(Xv_s)
        Xte_s = X_scaler.transform(Xte_s)
    elif scale_type == "logstandard":
        Xtr_s = np.log(Xtr)
        Xv_s = np.log(Xv)
        Xte_s = np.log(Xte)
        X_scaler = StandardScaler(copy=False)
        Xtr_s = X_scaler.fit_transform(Xtr_s)
        Xv_s = X_scaler.transform(Xv_s)
        Xte_s = X_scaler.transform(Xte_s)
    elif scale_type == "sqrtstandard":
        Xtr_s = np.sqrt(1+Xtr)
        Xv_s = np.sqrt(1+Xv)
        Xte_s = np.sqrt(1+Xte)
        X_scaler = StandardScaler(copy=False)
        Xtr_s = X_scaler.fit_transform(Xtr_s)
        Xv_s = X_scaler.transform(Xv_s)
        Xte_s = X_scaler.transform(Xte_s)
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")
    
    # Convert dtype
    Xtr_s = Xtr_s.astype(dtype, copy=False)
    Xv_s = Xv_s.astype(dtype, copy=False)
    Xte_s = Xte_s.astype(dtype, copy=False)
    
    # --- Scale y ---
    y_scaler = None
    if scale_type == "standard":
        y_scaler = StandardScaler(copy=False)
        ytr_s = y_scaler.fit_transform(ytr.reshape(-1, 1)).ravel()
        yv_s = y_scaler.transform(yv.reshape(-1, 1)).ravel()
        yte_s = y_scaler.transform(yte.reshape(-1, 1)).ravel()
    elif scale_type == "robust":
        y_scaler = RobustScaler(quantile_range=(25, 75))
        ytr_s = y_scaler.fit_transform(ytr.reshape(-1, 1)).ravel()
        yv_s = y_scaler.transform(yv.reshape(-1, 1)).ravel()
        yte_s = y_scaler.transform(yte.reshape(-1, 1)).ravel()
    elif scale_type == "asinh":
        ytr_s = np.arcsinh(ytr)
        yv_s = np.arcsinh(yv)
        yte_s = np.arcsinh(yte)
    elif scale_type == "log":
        ytr_s = np.log(ytr)
        yv_s = np.log(yv)
        yte_s = np.log(yte)
    elif scale_type == "log1p":
        ytr_s = np.log1p(ytr)
        yv_s = np.log1p(yv)
        yte_s = np.log1p(yte)
    elif scale_type == "asinhstandard":
        ytr_s = np.arcsinh(ytr)
        yv_s = np.arcsinh(yv)
        yte_s = np.arcsinh(yte)
        y_scaler = StandardScaler(copy=False)
        ytr_s = y_scaler.fit_transform(ytr_s.reshape(-1, 1)).ravel()
        yv_s = y_scaler.transform(yv_s.reshape(-1, 1)).ravel()
        yte_s = y_scaler.transform(yte_s.reshape(-1, 1)).ravel()
    elif scale_type == "log1pstandard":
        ytr_s = np.log1p(ytr)
        yv_s = np.log1p(yv)
        yte_s = np.log1p(yte)
        y_scaler = StandardScaler(copy=False)
        ytr_s = y_scaler.fit_transform(ytr_s.reshape(-1, 1)).ravel()
        yv_s = y_scaler.transform(yv_s.reshape(-1, 1)).ravel()
        yte_s = y_scaler.transform(yte_s.reshape(-1, 1)).ravel()
    elif scale_type == "logstandard":
        ytr_s = np.log(ytr)
        yv_s = np.log(yv)
        yte_s = np.log(yte)
        y_scaler = StandardScaler(copy=False)
        ytr_s = y_scaler.fit_transform(ytr_s.reshape(-1, 1)).ravel()
        yv_s = y_scaler.transform(yv_s.reshape(-1, 1)).ravel()
        yte_s = y_scaler.transform(yte_s.reshape(-1, 1)).ravel()
    elif scale_type == "sqrtstandard":
        ytr_s = np.sqrt(1+ytr)
        yv_s = np.sqrt(1+yv)
        yte_s = np.sqrt(1+yte)
        y_scaler = StandardScaler(copy=False)
        ytr_s = y_scaler.fit_transform(ytr_s.reshape(-1, 1)).ravel()
        yv_s = y_scaler.transform(yv_s.reshape(-1, 1)).ravel()
        yte_s = y_scaler.transform(yte_s.reshape(-1, 1)).ravel()
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")
    
    # Convert dtype
    ytr_s = ytr_s.astype(dtype, copy=False)
    yv_s = yv_s.astype(dtype, copy=False)
    yte_s = yte_s.astype(dtype, copy=False)
    
    if not merge:
        return Xtr_s, ytr_s, Xv_s, yv_s, Xte_s, yte_s
    
    # --- Merge train+val and create new test scaled on merged data ---
    Xtr_val = np.concatenate([Xtr, Xv], axis=0)
    ytr_val = np.concatenate([ytr, yv], axis=0)
    
    # Scale merged data
    X_scaler_merged = None
    y_scaler_merged = None
    
    # --- Scale merged X ---
    if scale_type == "standard":
        X_scaler_merged = StandardScaler(copy=False)
        Xtr_val_s = X_scaler_merged.fit_transform(Xtr_val)
        Xte_merged_s = X_scaler_merged.transform(Xte)
    elif scale_type == "robust":
        X_scaler_merged = RobustScaler(quantile_range=(25, 75))
        Xtr_val_s = X_scaler_merged.fit_transform(Xtr_val)
        Xte_merged_s = X_scaler_merged.transform(Xte)
    elif scale_type == "asinh":
        Xtr_val_s = np.arcsinh(Xtr_val)
        Xte_merged_s = np.arcsinh(Xte)
    elif scale_type == "log":
        Xtr_val_s = np.log(Xtr_val)
        Xte_merged_s = np.log(Xte)
    elif scale_type == "log1p":
        Xtr_val_s = np.log1p(Xtr_val)
        Xte_merged_s = np.log1p(Xte)
    elif scale_type == "asinhstandard":
        Xtr_val_s = np.arcsinh(Xtr_val)
        Xte_merged_s = np.arcsinh(Xte)
        X_scaler_merged = StandardScaler(copy=False)
        Xtr_val_s = X_scaler_merged.fit_transform(Xtr_val_s)
        Xte_merged_s = X_scaler_merged.transform(Xte_merged_s)
    elif scale_type == "log1pstandard":
        Xtr_val_s = np.log1p(Xtr_val)
        Xte_merged_s = np.log1p(Xte)
        X_scaler_merged = StandardScaler(copy=False)
        Xtr_val_s = X_scaler_merged.fit_transform(Xtr_val_s)
        Xte_merged_s = X_scaler_merged.transform(Xte_merged_s)
    elif scale_type == "logstandard":
        Xtr_val_s = np.log(Xtr_val)
        Xte_merged_s = np.log(Xte)
        X_scaler_merged = StandardScaler(copy=False)
        Xtr_val_s = X_scaler_merged.fit_transform(Xtr_val_s)
        Xte_merged_s = X_scaler_merged.transform(Xte_merged_s)
    elif scale_type == "sqrtstandard":
        Xtr_val_s = np.sqrt(1+Xtr_val)
        Xte_merged_s = np.sqrt(1+Xte)
        X_scaler_merged = StandardScaler(copy=False)
        Xtr_val_s = X_scaler_merged.fit_transform(Xtr_val_s)
        Xte_merged_s = X_scaler_merged.transform(Xte_merged_s)
    
    # --- Scale merged y ---
    if scale_type == "standard":
        y_scaler_merged = StandardScaler(copy=False)
        ytr_val_s = y_scaler_merged.fit_transform(ytr_val.reshape(-1, 1)).ravel()
        yte_merged_s = y_scaler_merged.transform(yte.reshape(-1, 1)).ravel()
    elif scale_type == "robust":
        y_scaler_merged = RobustScaler(quantile_range=(25, 75))
        ytr_val_s = y_scaler_merged.fit_transform(ytr_val.reshape(-1, 1)).ravel()
        yte_merged_s = y_scaler_merged.transform(yte.reshape(-1, 1)).ravel()
    elif scale_type == "asinh":
        ytr_val_s = np.arcsinh(ytr_val)
        yte_merged_s = np.arcsinh(yte)
    elif scale_type == "log":
        ytr_val_s = np.log(ytr_val)
        yte_merged_s = np.log(yte)
    elif scale_type == "log1p":
        ytr_val_s = np.log1p(ytr_val)
        yte_merged_s = np.log1p(yte)
    elif scale_type == "asinhstandard":
        ytr_val_s = np.arcsinh(ytr_val)
        yte_merged_s = np.arcsinh(yte)
        y_scaler_merged = StandardScaler(copy=False)
        ytr_val_s = y_scaler_merged.fit_transform(ytr_val_s.reshape(-1, 1)).ravel()
        yte_merged_s = y_scaler_merged.transform(yte_merged_s.reshape(-1, 1)).ravel()
    elif scale_type == "log1pstandard":
        ytr_val_s = np.log1p(ytr_val)
        yte_merged_s = np.log1p(yte)
        y_scaler_merged = StandardScaler(copy=False)
        ytr_val_s = y_scaler_merged.fit_transform(ytr_val_s.reshape(-1, 1)).ravel()
        yte_merged_s = y_scaler_merged.transform(yte_merged_s.reshape(-1, 1)).ravel()
    elif scale_type == "logstandard":
        ytr_val_s = np.log(ytr_val)
        yte_merged_s = np.log(yte)
        y_scaler_merged = StandardScaler(copy=False)
        ytr_val_s = y_scaler_merged.fit_transform(ytr_val_s.reshape(-1, 1)).ravel()
        yte_merged_s = y_scaler_merged.transform(yte_merged_s.reshape(-1, 1)).ravel()
    elif scale_type == "sqrtstandard":
        ytr_val_s = np.sqrt(1+ytr_val)
        yte_merged_s = np.sqrt(1+yte)
        y_scaler_merged = StandardScaler(copy=False)
        ytr_val_s = y_scaler_merged.fit_transform(ytr_val_s.reshape(-1, 1)).ravel()
        yte_merged_s = y_scaler_merged.transform(yte_merged_s.reshape(-1, 1)).ravel()
    
    # Convert dtype
    Xtr_val_s = Xtr_val_s.astype(dtype, copy=False)
    Xte_merged_s = Xte_merged_s.astype(dtype, copy=False)
    ytr_val_s = ytr_val_s.astype(dtype, copy=False)
    yte_merged_s = yte_merged_s.astype(dtype, copy=False)
    
    return (
        Xtr_s, ytr_s, Xv_s, yv_s, Xte_s, yte_s,
        Xtr_val_s, ytr_val_s, Xte_merged_s, yte_merged_s
    )
