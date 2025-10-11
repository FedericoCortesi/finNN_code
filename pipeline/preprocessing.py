import pandas as pd
import numpy as np

from  paths import SP500_PATH

def import_data(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path)

    # Handle dates
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    return df

def adjust_for_splits(df: pd.DataFrame,
                      base_cols=("close","open","high","low"),
                      factor_col="cfacpr",
                      group_col="permno") -> pd.DataFrame:
    """
    Split-adjust price columns using CRSP cumulative price factor `cfacpr`.
    Convention: adjusted_col = raw_col / cfacpr.
    """
    out = df.copy()
    # ensure factor exists and is 1 when missing, per stock
    if factor_col not in out.columns:
        raise KeyError(f"Missing factor column: {factor_col}")
    
    # if factor is occasionally 0 (data glitch), set to 1 to avoid div/0
    fac = out[factor_col].replace(0, np.nan)
    
    # forward-fill within permno, then fill remaining NaNs with 1
    fac = out.groupby(group_col, sort=False)[factor_col].ffill().fillna(1.0)
    out.loc[:, base_cols] = out.loc[:, base_cols].div(fac, axis=0)
    
    return out

def handle_nans(df: pd.DataFrame) -> pd.DataFrame:
    # --- Adjust prices first (split-adjusted base) ---
    price_cols = [c for c in ("close","open","high","low") if c in df.columns]
    df = adjust_for_splits(df, base_cols=tuple(price_cols), factor_col="cfacpr", group_col="permno")

    # --- Sort / group ---
    df = df.sort_values(["permno", "date"]).copy()
    g  = df.groupby("permno", sort=False)

    # --- Clean returns & volume ---
    if "ret" in df.columns:
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

def create_returns(df:pd.DataFrame,
                       cols:list=["open", "high", "low", "close"]
                       )->pd.DataFrame:
    """
    Transform prices to log returns.
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


def preprocess(returns:bool=False):
    df = import_data(SP500_PATH)
    df = adjust_for_splits(df)
    df = handle_nans(df)
    if returns:
        df = create_returns(df)

    df = create_time_index(df)

    return df



