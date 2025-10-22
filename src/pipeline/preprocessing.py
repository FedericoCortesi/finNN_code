import numpy as np
import pandas as pd
from typing import Iterable
from  utils.paths import SP500_PATH
from utils.custom_formatter import setup_logger
import warnings
warnings.simplefilter(action="ignore")


console_logger = setup_logger("Preprocessing", "DEBUG")

def import_data(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path)

    # Handle dates
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    return df


def adjust_for_splits(
    df: pd.DataFrame,
    price_cols: Iterable[str] = ("close","open","high","low"),
    vol_cols:   Iterable[str] = ("vol",),     # add "shrout" here if desired
    price_factor: str = "cfacpr",
    share_factor: str = "cfacshr",
    group_col: str = "permno",
) -> pd.DataFrame:
    """
    Split/Share adjustments using CRSP cumulative factors.

    Prices:     adjusted = raw / cfacpr
    Volume-like adjusted = raw * cfacshr

    - Returns columns ('ret' or 'ret_*') are NEVER modified.
    - Missing factors default to 1.0 after forward-fill within each permno.
    - Columns not present are silently skipped.

    Parameters
    ----------
    df : DataFrame
    price_cols : columns to treat as prices (divide by cfacpr)
    vol_cols   : columns to treat as volume/shares (multiply by cfacshr)
    price_factor : name of cumulative price factor (cfacpr)
    share_factor : name of cumulative share factor (cfacshr)
    group_col    : security identifier (permno)
    """
    out = df.copy()

    if group_col not in out.columns:
        raise KeyError(f"Missing group column: {group_col}")

    # Helper: filter to existing columns and exclude any 'ret' columns
    def existing_nonret(cols):
        cols = [c for c in cols if c in out.columns]
        return [c for c in cols if not (c == "ret" or str(c).lower().startswith("ret_"))]

    price_cols = tuple(existing_nonret(price_cols))
    vol_cols   = tuple(existing_nonret(vol_cols))

    # --- Price adjustment (cfacpr) ---
    if price_cols and price_factor in out.columns:
        fac_pr = (
            out[price_factor]
              .replace(0, np.nan)                              # guard against zeros
              .groupby(out[group_col], sort=False).ffill()     # causal within permno
              .fillna(1.0)                                     # leading NaNs -> 1
        )
        out.loc[:, price_cols] = out.loc[:, price_cols].div(fac_pr, axis=0)

    # --- Volume/share adjustment (cfacshr) ---
    if vol_cols and share_factor in out.columns:
        fac_sh = (
            out[share_factor]
              .replace(0, np.nan)
              .groupby(out[group_col], sort=False).ffill()
              .fillna(1.0)
        )
        out.loc[:, vol_cols] = out.loc[:, vol_cols].mul(fac_sh, axis=0)

    return out

def handle_nans(df: pd.DataFrame) -> pd.DataFrame:
    # --- Adjust prices first (split-adjusted base) ---
    price_cols = [c for c in ("close","open","high","low") if c in df.columns]
    df = adjust_for_splits(df, price_cols=tuple(price_cols), group_col="permno")

    # --- Sort / group ---
    df = df.sort_values(["permno", "date"]).copy()
    g  = df.groupby("permno", sort=False)

    # --- Clean returns & volume ---
    if "ret" in df.columns:
        pct = df["ret"].isna().mean()
        console_logger.debug(f"percentage of nan returns {pct:.4%}")
        df["ret"] = df["ret"].fillna(0.0)
    if "vol" in df.columns:
        df["vol"] = df["vol"].fillna(0.0)

    # --- Forward-fill prices (causal) ---
    for c in ("close","low","high"):
        if c in df.columns:
            df[c] = g[c].ffill()

    # --- Open: prev close, then ffill within stock; final same-day fallbacks ---
    if {"open","close"}.issubset(df.columns):
        prev_close = g["close"].shift(1)
        df["open"] = df["open"].fillna(prev_close)
        df["open"] = g["open"].ffill()

        # if still NaN (typically first obs per permno)
        mask = df["open"].isna()
        if mask.any():
            # prefer same-day close if present (non-leaky; same timestamp)
            have_close = mask & df["close"].notna()
            df.loc[have_close, "open"] = df.loc[have_close, "close"]
            # then mid if high/low present
            if {"high","low"}.issubset(df.columns):
                have_mid = df["open"].isna() & df["high"].notna() & df["low"].notna()
                df.loc[have_mid, "open"] = 0.5*(df.loc[have_mid,"high"] + df.loc[have_mid,"low"])

    # --- Optional: shares & market cap ---
    if "shrout" in df.columns:
        df["shrout"] = g["shrout"].ffill()
    if {"close","shrout"}.issubset(df.columns):
        # CRSP PRC can be negative; use abs when forming cap
        df["cap"] = df["close"].abs() * df["shrout"]

    return df

def create_ohlc_returns(df:pd.DataFrame,
                       cols:list=["open", "high", "low", "close"]
                       )->pd.DataFrame:
    """
    Transform prices to returns.
    """
    df = df.sort_values(['permno', 'date']).copy()  # always sort before groupby

    # Iterate over each column
    for c in cols:
        df[f'ret_{c}'] = (
            df.groupby('permno')[c]
            .apply(lambda x: x / x.shift(1)-1).values
        )

    # Drop Nans (first day returns)
    df = df.dropna()

    return df

def create_time_index(df:pd.DataFrame):
    # Transform to contiguos index
    unique_dates = np.sort(df["date"].unique())
    date_to_int = {d: i for i, d in enumerate(unique_dates)} # 
    df["t"] = df["date"].map(date_to_int)

    return df

def compute_variance(df:pd.DataFrame):
    df = df.copy()
    
    # Pre-calculate the constant value
    C2 = 2 * np.log(2) - 1  # Approx. 0.3863

    # Extract NumPy arrays for faster computation
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()

    # Calculate volatility using NumPy arrays
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    
    df["var"] = 0.5 * (log_hl ** 2) - C2 * (log_co ** 2)

    console_logger.debug(f"number of nans in var: {df["var"].isna().sum()}")
    df = df.dropna()

    return df

def preprocess(path:str=SP500_PATH, 
               nan_imputation:bool=True,
               ohlc_rets:bool=False,
               variance:bool=True):
    df = import_data(path)
    # we dont need to refactor since we only look
    # at ratios for volatility.
    #df = adjust_for_splits(df) #  Necessary only for OHLC
    if nan_imputation:
        df = handle_nans(df)
    
    if ohlc_rets:
        df = create_ohlc_returns(df)
    
    if variance:
        df = compute_variance(df)

    df = create_time_index(df)

    return df



