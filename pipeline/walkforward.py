from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


from typing import Callable, Dict, Tuple, List

Array2D = np.ndarray  # shape (T, F)
Array3D = np.ndarray  # shape (T, L, F)

@dataclass
class WFConfig:
    lags: int = 20 # to validate
    T_train: int = 252*3
    T_val: int = 252
    T_test: int = 252
    step: int = 252 
    batch: int = 100 # For SGD 
    epochs: int = 50
    patience: int = 8
    monitor: str = "val_rmse"

class WFCVTrainer:
    def __init__(
        self,
        cfg: WFConfig,
        df_long: pd.DataFrame,                  # preprocessed long df
        id_col: str, 
        date_col: str = "date",
        value_cols: List[str] = ["logret_open", 
                                 "logret_high", 
                                 "logret_low", 
                                 "logret_close", 
                                 "volume"],     # e.g. ["close","open","high","low"]
        target_col: str = "ret",                        # e.g. "ret" or "close_lead1"
        scaler_factory: Callable = StandardScaler.fit_transform,
    ):
        self.df = df_long.copy()
        self.id_col, self.date_col = id_col, date_col
        self.value_cols, self.target_col = value_cols, target_col
        self.cfg = cfg
        self.scaler_factory = scaler_factory

        # build wide dict once (cheap, readable)
        # self.wide = self._long_to_wide(self.df, value_cols + [target_col])

    def _choose_symbols(self, pd.DataFrame):

        return


    

    def _long_to_wide(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.DataFrame]:
        df = df.sort_values([self.date_col, self.id_col])
        # collapse duplicates deterministically if any
        df = (df.groupby([self.date_col, self.id_col], as_index=False).last())
        wide = {}
        for c in cols:
            w = df.pivot(index=self.date_col, columns=self.id_col, values=c).sort_index()
            wide[c] = w
        # align columns across variables
        common_cols = None
        for w in wide.values():
            common_cols = w.columns if common_cols is None else common_cols.intersection(w.columns)
        for k in wide:
            wide[k] = wide[k].reindex(columns=common_cols)
        return wide

    # ---- utilities
    def _walk_forward(self, T: int):
        s = 0
        while True:
            a, b = s, s + self.cfg.T_train
            c, d = b, b + self.cfg.T_val
            e = d + self.cfg.T_test
            if e > T: break
            yield slice(a,b), slice(b,c), slice(c,d)
            s += self.cfg.step

    def _make_lagged(self, X: Array2D, L: int) -> Array3D:
        # X: (T,F) -> (T-L,L,F)
        T, F = X.shape
        return np.stack([X[i:T-L+i] for i in range(L)], axis=1)

    # ---- fold assembly
    def _design_for_fold(
        self, tr: slice, va: slice, te: slice
    ) -> Tuple[Array3D, np.ndarray, Array3D, np.ndarray, Array3D, np.ndarray, List[int]]:
        # choose symbols using **train slice only**
        train_frame = self.wide[self.value_cols[0]].iloc[tr]  # any base var for coverage
        syms = self._choose_symbols(train_frame)               # returns list of permno

        # stack features in fixed order across variables & symbols
        X_blocks = [self.wide[v][syms] for v in self.value_cols]   # each (T, |S|)
        X_df = pd.concat(X_blocks, axis=1)                         # (T, V*|S|)
        y_df = self.wide[self.target_col][syms]                    # (T, |S|) or pick one

        # optional scaler: fit on train, apply everywhere (no leakage)
        if self.scaler_factory is not None:
            scaler = self.scaler_factory(X_df.iloc[tr])
            X_df = scaler(X_df)

        # collapse y for a single target per timepoint (example: cross-section mean)
        y_series = y_df.mean(axis=1)  # replace with your target rule

        # build arrays per slice
        L = self.cfg.lags
        def to_arrays(slc):
            X_2d = X_df.iloc[slc].to_numpy()
            X_3d = self._make_lagged(X_2d, L)              # (Tslc-L,L,F)
            y_1d = y_series.iloc[slc][L:].to_numpy()       # align lead
            return X_3d, y_1d

        Xtr, ytr = to_arrays(tr)
        Xva, yva = to_arrays(va)
        Xte, yte = to_arrays(te)
        return Xtr, ytr, Xva, yva, Xte, yte, syms
