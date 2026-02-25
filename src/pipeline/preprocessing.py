import numpy as np
import pandas as pd
from typing import Iterable
from  utils.paths import SP500_PATH
from utils.custom_formatter import setup_logger
import warnings
warnings.simplefilter(action="ignore")


console_logger = setup_logger("Preprocessing", "INFO")

def import_data(path) -> pd.DataFrame:
    console_logger.debug('In import data')
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
    console_logger.debug('In adjust for splits')
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
    console_logger.debug('In handle nans')
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
    console_logger.debug('create returns')
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
    console_logger.debug('create time index')
    # Transform to contiguos index
    unique_dates = np.sort(df["date"].unique())
    date_to_int = {d: i for i, d in enumerate(unique_dates)} # 
    df["t"] = df["date"].map(date_to_int)

    return df

def compute_variance(df:pd.DataFrame, annualize_var:bool=False):
    console_logger.debug('compute var')
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

    if annualize_var: 
        df["var"] = 252 * df["var"]

    console_logger.debug(f"number of nans in var: {df["var"].isna().sum()}")
    df = df.dropna()

    return df

def gk_decile_portfolios(
    df: pd.DataFrame,
    window: int = 20,
    num: int = 10
) -> pd.DataFrame:
    """
    Build daily decile "representative portfolios" sorted by rolling variance of returns (per stock),
    then compute the Garman-Klass daily variance estimate at the *portfolio* level.

    Portfolio GK is computed by:
      1) per-stock log ratios: log_hl = log(H/L), log_co = log(C/O)
      2) aggregate within (date, decile) using mean (equal-weight) or weighted mean
      3) plug aggregated log ratios into GK formula:
           var = 0.5 * log_hl^2 - (2*ln(2)-1) * log_co^2

    Parameters
    ----------
    df : DataFrame
        Must contain columns: permno, date, ret, open, high, low, close
        If weight_col is not None, must also contain that column.
    window : int
        Rolling window for sorting signal (variance of ret).

    Returns
    -------
    port_df : DataFrame (long format)
    """
    x = df.copy()
    min_periods = window

    # --- hygiene ---
    x["date"] = pd.to_datetime(x["date"]).dt.normalize()
    x = x.sort_values(["permno", "date"])

    # --- 1) sorting signal: rolling variance of ret per stock ---
    x["rolling_var"] = (
        x.groupby("permno", sort=False)["ret"]
          .rolling(window=window, min_periods=min_periods)
          .var()
          .reset_index(level=0, drop=True)
    )

    # --- 2) daily deciles by rolling_var ---
    mask = x["rolling_var"].notna()
    r = x.loc[mask].groupby("date")["rolling_var"].rank(method="first")
    n_day = x.loc[mask].groupby("date")["rolling_var"].transform("count")
    x.loc[mask, "decile"] = (np.floor((r - 1) * num / n_day) + 1).astype("int64").clip(1, num)

    # --- 3) compute per-stock GK inputs (log ratios) ---
    # Require positive OHLC
    valid = (
        x["decile"].notna()
        & (x["open"] > 0) & (x["high"] > 0) & (x["low"] > 0) & (x["close"] > 0)
    )
    x = x.loc[valid].copy()

    x["log_hl"] = np.log(x["high"] / x["low"])
    x["log_co"] = np.log(x["close"] / x["open"])

    # --- 4) aggregate to portfolio level (date, decile) ---
    port = (
        x.groupby(["date", "decile"], as_index=False)
            .agg(
                log_hl=("log_hl", "mean"),
                log_co=("log_co", "mean"),
            )
    )

    # --- 5) portfolio GK variance ---
    C2 = 2 * np.log(2) - 1
    port["var"] = 0.5 * (port["log_hl"] ** 2) - C2 * (port["log_co"] ** 2)


    # --- labels / long format ---
    port["permno"] = port["decile"].astype(int)
    port = port.sort_values(["date", "decile"]).reset_index(drop=True)


    return port[['date', 'var', 'permno']]

def preprocess(path:str=SP500_PATH, 
               nan_imputation:bool=True,
               ohlc_rets:bool=False,
               annualize_var:bool=False,
               portfolios:bool=False
               ):
    console_logger.debug('in preprocess')
    
    df = import_data(path)
    # we dont need to refactor since we only look
    # at ratios for volatility.
    #df = adjust_for_splits(df) #  Necessary only for OHLC
    if nan_imputation:
        df = handle_nans(df)
    
    if ohlc_rets:
        df = create_ohlc_returns(df)
    
    if portfolios > 0 :
        console_logger.warning('Portfolios done')
        df = gk_decile_portfolios(df, num=portfolios)
    else:
        df = compute_variance(df, annualize_var)

    df = create_time_index(df)

    return df



