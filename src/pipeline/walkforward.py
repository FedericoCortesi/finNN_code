import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


import ast
from pathlib import Path
from collections.abc import Sequence
from typing import Callable, Dict, Tuple, List, Optional

from pipeline.preprocessing import preprocess
from config.config_types import WFConfig
from pipeline.pipeline_utils import scale_split

from utils.custom_formatter import setup_logger
from  utils.paths import SP500_PATH, DATA_DIR

class WFCVGenerator:
    def __init__(
        self,
        config: WFConfig,
        id_col: str = "permno",
        df_long = None,   # None => call preprocess(), can be str for path or df
        time_col: str = "t",
    ):
        self.console_logger = setup_logger("WFCVGenerator", "INFO")
        self.config = config
        self.console_logger.debug(self.config.summary())


        self.scale = self.config.scale 
        self.scale_type = self.config.scale_type 
        if self.scale is None:
            self.scale = False
        
        if self.scale_type is None and self.scale:
            self.scale_type = "standard"
        elif self.scale_type is not None:
            self.scale_type = self.config.scale_type.lower() 

        self.console_logger.debug(f'self.scale_type: {self.scale_type}')
        self.console_logger.debug(f'self.scale: {self.scale}')

        self.id_col, self.time_col = id_col, time_col
        self.target_col = self.config.target_col
        
        self.df = self._load_df(df_long)  # validated & trimmed

        # now safe to proceed
        self.T = self.df[self.time_col].nunique()
        self.stamps_and_windows_array = self._make_windows()
        self.df_master = self._build_master_df()

    def _load_df(self, df_long) -> pd.DataFrame:
        if df_long is None:
            df = preprocess(annualize_var=self.config.annualize)[[self.time_col, self.target_col, self.id_col]].copy()
        elif isinstance(df_long, str):
            df = preprocess(path=f'{DATA_DIR}/df_long', annualize_var=self.config.annualize)[[self.time_col, self.target_col, self.id_col]].copy()
        elif isinstance(df_long, pd.DataFrame):
            df = df_long
            
        # validate required columns
        required = {self.time_col, self.target_col, self.id_col}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # select & enforce dtypes
        df = df[[self.time_col, self.target_col, self.id_col]].copy()
        df[self.time_col] = pd.to_numeric(df[self.time_col], errors="raise", downcast="integer")
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="raise")
        # permno often fits in int32; keep as object-safe if needed
        if not pd.api.types.is_integer_dtype(df[self.id_col]):
            df[self.id_col] = pd.to_numeric(df[self.id_col], errors="raise", downcast="integer")


        self.console_logger.debug(f'Preprocessed df: {df}')
        return df




    def _walk_forward(self):
        # self.config.lagsast train index
        t_0 = 0

        result = []

        folds_count = 0  

        while True:
            self.console_logger.debug(f"In walk forward true")
            a, b = t_0 , t_0 + self.config.T_train # train
            c = b + self.config.T_val # validation
            d = c + self.config.T_test # test
            if d > self.T: 
                break
        
            result.append([slice(a,b), slice(b,c), slice(c,d)])
            t_0 += self.config.step
            folds_count += 1
        
        self.folds_count = folds_count
        msg = f"self.folds_count is zero! Check walkforward paramters, make sure train end does not exceeed {self.T}." 
        msg = msg + f"Right now train, val, and test sizes are: {self.config.T_train}, {self.config.T_val}, {self.config.T_test}." 
        assert self.folds_count != 0, msg 
        return result




    def _make_windows(self):
        """
        Returns:
        train_stamps, train_windows, val_stamps, val_windows, test_stamps, test_windows
        """
        # Declare vars to store
        train_stamps = []
        val_stamps = []
        test_stamps = []

        train_windows = []
        val_windows = []
        test_windows = []

        # Iterate over folds
        for train, val, test in self._walk_forward():
            # Append to lists
            train_stamps.append(train)
            val_stamps.append(val)
            test_stamps.append(test)

    
            new_train = [(train.start+i, train.start+i+self.config.lags) 
                         for i in range(self.config.T_train-self.config.lags)]
            new_val = [(val.start+i, val.start+i+self.config.lags) 
                       for i in range(self.config.T_val-self.config.lags)]
            new_test = [(test.start+i, test.start+i+self.config.lags) 
                        for i in range(self.config.T_test-self.config.lags)]


            assert all(train.start <= a and b <= train.stop
                    for a,b in new_train), "Train windows out of bounds!"
            assert all(val.start <= a and b <= val.stop
                    for a,b in new_val), "Val windows out of bounds!"
            assert all(test.start <= a and b <= test.stop
                    for a,b in new_test), "Test windows out of bounds!"

            # Make inner lists with all possible windows
            train_windows.append(new_train) 
            val_windows.append(new_val) 
            test_windows.append(new_test) 

        # define all windows possible
        all_windows = set([(i, i+self.config.lags)for i in range(self.T-self.config.lags)])
        all_windows = sorted(all_windows)


        return train_stamps, train_windows, val_stamps, val_windows, test_stamps, test_windows, all_windows
    


    def _build_master_df(
            self, 
            ) -> pd.DataFrame:
        
        # Obtain windows 
        windows = self.stamps_and_windows_array[6].copy()

        assert not self.df.duplicated([self.id_col, self.time_col]).any(), \
       "Duplicate (permno, t) rows before pivot."


        # 1) Wide once: rows=permno, cols=t (sorted) 
        # Values has to be ret!!!!!!!!!!!!!
        # W is a wide matrix (permno x time)
        W = (self.df.pivot(index=self.id_col, columns=self.time_col, values=self.target_col)
                    .sort_index(axis=1))
        col_index = W.columns  # Int64Index of t's
        W = W.astype("float64")  # otherwise numpy doesnt coerce pd nans
        V = W.values           # ndarray (n_permno x n_time), avoids per-iteration DataFrame ops

        # Precompute final column names for each block
        out_cols = [f"feature_{i}" for i in range(self.config.lags)] + ["y"]

        out_frames = []  # collect blocks here

        for a, b in windows:

            # 2) Get column *positions* for [a..b] (inclusive), skip if any missing
            # add one for y
            wanted = np.arange(a, b + 1)
            pos = col_index.get_indexer(wanted)   # -1 where missing
            if (pos < 0).any():
                # some dates missing in wide; skip this window entirely
                continue

            # 3) Slice ndarray by columns; shape (n_permno × (L+1))
            block = V[:, pos]

            # 4) Drop any permno with NaN in the window
            ok_rows = ~np.isnan(block).any(axis=1)
            if not ok_rows.any():
                continue
            #self.console_logger.debug(f"block: {block}")
            block = block[ok_rows]

            # 5) Build a small DataFrame for this window; rename cols
            #    First L columns → features, last col → y
            df_block = pd.DataFrame(block, columns=out_cols)

            # add window debugrmatio
            df_block["window"] = [(a,b)]*len(df_block)


            # (optional) If you want to keep which permno each row is:
            # keep_ids = W.index.to_numpy()[ok_rows]
            # df_block.insert(0, self.id_col, keep_ids)
            # df_block.insert(1, "t_end", b)

            out_frames.append(df_block)

        # 6) Single concat at the end
        if out_frames:
            df_base = pd.concat(out_frames, ignore_index=True)
        else:
            df_base = pd.DataFrame(columns=out_cols)

        if self.config.lookback > 0:
            # create lookback columns
            lookback_cols = [f"lookback_{i}" for i in range(self.config.lookback)]
            cols_to_take = [f"feature_{i}" for i in range(self.config.lags-self.config.lookback, self.config.lags)]
            df_base[lookback_cols] = df_base[cols_to_take]
            
            # define the intended order
            feature_cols  = [f"feature_{i}" for i in range(self.config.lags)]
            lookback_cols = [f"lookback_{i}" for i in range(self.config.lookback)]
            final_cols    = feature_cols + lookback_cols + ["y", "window"]

            # reorder DataFrame
            df_base = df_base[final_cols]

        return df_base


    def obtain_datasets_fold(self, fold:int, df_master: pd.DataFrame | None=None):
        """
        Given a fold index, return (df_train, df_val, df_test).
        Accepts an optional external df_master with columns: feature_*, y, window.
        """
        if not (0 <= fold < getattr(self, "folds_count", 0)):
            raise IndexError(f"Fold {fold} out of range [0, {self.folds_count-1}].")

        # Back out windows
        train_stamps, train_windows, val_stamps, val_windows, test_stamps, test_windows, all_windows = self.stamps_and_windows_array

        # choose master df (internal or provided)
        base = (df_master.copy() if df_master is not None else self.df_master.copy())

        # sanity: required columns
        required = {"y", "window"}
        missing = required - set(base.columns)
        if missing:
            raise KeyError(f"Provided DataFrame missing required columns: {missing}")
        
        base = self._normalize_window_col(base)

        # slice by windows for this fold
        tw, vw, tew = train_windows[fold], val_windows[fold], test_windows[fold]
        df_train = base[base["window"].isin(tw)].drop(columns="window")
        df_val   = base[base["window"].isin(vw)].drop(columns="window")
        df_test  = base[base["window"].isin(tew)].drop(columns="window")

        # optional: warn on empties
        if df_train.empty or df_val.empty or df_test.empty:
            self.console_logger.warning(
                f"Fold {fold}: empty split(s). "
                f"train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
            )

        return df_train, df_val, df_test

    def _normalize_window_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure df['window'] contains 2-tuples of ints: (a, b).
        Accepts tuples, lists, NumPy arrays (shape (2,) or (1,2)), Pandas arrays/Series,
        and strings like '(0, 20)' or '[0, 20]'. Raises on anything else.
        """
        if "window" not in df.columns:
            raise KeyError("Expected column 'window' in the provided DataFrame.")

        def to_tuple(x):
            # Treat missing as error (you can decide to drop/forward-fill instead)
            if x is None or (isinstance(x, float) and np.isnan(x)):
                raise ValueError("Found missing value in 'window' column.")

            # Fast path: already a 2-tuple
            if isinstance(x, tuple) and len(x) == 2:
                return (int(x[0]), int(x[1]))

            # Generic 2-length sequence (but not str)
            if isinstance(x, np.ndarray) and not isinstance(x, (str, bytes)):
                # Convert any sequence-like (list, pd.Series, pd.Array, etc.) to 1D array
                arr = np.asarray(x).ravel()
                if arr.shape == (2,):
                    return (int(arr[0]), int(arr[1]))

            # Strings: try literal_eval into list/tuple, then recurse to the sequence branch
            if isinstance(x, str):
                try:
                    y = ast.literal_eval(x)
                    arr = np.asarray(y).ravel()
                    if arr.shape == (2,):
                        return (int(arr[0]), int(arr[1]))
                except Exception:
                    pass  # fall through to error


            raise ValueError(
                f"Unrecognized window value {x!r}. Expected a 2-length sequence "
                f"(tuple/list/array/Series) or a string like '(a, b)'."
            )

        out = df.copy()
        # Use list comprehension for clearer exceptions and speed
        out["window"] = [to_tuple(v) for v in out["window"].to_numpy()]
        return out

        
    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        """
        Return feature columns ordered by numeric suffix: feature_0, feature_1, ...
        """
        cols = [c for c in df.columns if c.startswith("feature_")]
        if not cols:
            raise KeyError("No 'feature_*' columns found.")
        try:
            cols_sorted = sorted(cols, key=lambda c: int(c.split("_")[1]))
        except Exception:
            cols_sorted = sorted(cols)  # fallback alphabetical
        return cols_sorted

    def _lookback_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Return lookback columns ordered by numeric suffix: feature_0, feature_1, ...
        """
        cols = [c for c in df.columns if c.startswith("lookback_")]
        if not cols:
            raise KeyError("No 'lookback_*' columns found.")
        try:
            cols_sorted = sorted(cols, key=lambda c: int(c.split("_")[1]))
        except Exception:
            cols_sorted = sorted(cols)  # fallback alphabetical
        return cols_sorted

    
    def folds(self, 
              df_master: pd.DataFrame | None = None):
        """
        Yield (Xtr, ytr, Xv, yv, Xte, yte) for each fold.
        If df_master is provided, it must have columns: feature_*, y, window.
        """
        # pick master df and normalize/check once
        base = (df_master.copy() if df_master is not None else self.df_master.copy())


        base = self._normalize_window_col(base)
        feat_cols = self._feature_cols(base)
        if self.config.lookback is not None and self.config.lookback > 0:
            lookback_cols = self._lookback_columns(base)
            output_cols = lookback_cols + ["y"]
        else:
            output_cols = ["y"]

        self.console_logger.debug(f"output_cols: {output_cols}")
        self.console_logger.debug(f"self.folds_count: {self.folds_count}")

        for fold in range(self.folds_count):
            df_train, df_val, df_test = self.obtain_datasets_fold(fold, df_master=base)

            if df_train.empty or df_val.empty or df_test.empty:
                self.console_logger.warning(f"Skipping fold {fold} due to empty split.")
                continue

            # build arrays in consistent column order
            Xtr = df_train[feat_cols].to_numpy(dtype=np.float64) 
            ytr = df_train[[*output_cols]].to_numpy(dtype=np.float64) # ensure its a list
            Xv = df_val[feat_cols].to_numpy(dtype=np.float64)   
            yv = df_val[[*output_cols]].to_numpy(dtype=np.float64) # ensure its a list
            Xte = df_test[feat_cols].to_numpy(dtype=np.float64)
            yte = df_test[[*output_cols]].to_numpy(dtype=np.float64) # ensure its a list

            
            if self.config.clip is not None and self.config.clip !=0:
                self.console_logger.debug(f'self.config.clip: {self.config.clip}')
                # Calculate percentiles threshold from training data only
                # TODO: MIGHT BE LEAKING, CHECK THIS!
                X_flat = Xtr.flatten()
                y_flat = ytr.flatten()
                lower_threshold_x = np.percentile(X_flat, self.config.clip)  # 0.5th percentile
                upper_threshold_x = np.percentile(X_flat, 100-self.config.clip)  # 99.5th percentile
                lower_threshold_y = np.percentile(y_flat, self.config.clip)  # 0.5th percentile
                upper_threshold_y = np.percentile(y_flat, 100-self.config.clip)  # 99.5th percentile
                
                
                lower_threshold = min(lower_threshold_x, lower_threshold_y)
                upper_threshold = max(upper_threshold_x, upper_threshold_y)
                
                
                train_mask_x = np.all((Xtr >= lower_threshold) & (Xtr <= upper_threshold), axis=1)
                train_mask_y = np.all((ytr >= lower_threshold) & (ytr <= upper_threshold), axis=1)
                train_mask = train_mask_y & train_mask_x # AND, keep only if both are True i.e. in bounds 
                
                val_mask_x = np.all((Xv >= lower_threshold) & (Xv <= upper_threshold), axis=1)
                val_mask_y = np.all((yv >= lower_threshold) & (yv <= upper_threshold), axis=1)
                val_mask = val_mask_y & val_mask_x
                
                test_mask_x = np.all((Xte >= lower_threshold) & (Xte <= upper_threshold), axis=1)
                test_mask_y = np.all((yte >= lower_threshold) & (yte <= upper_threshold), axis=1)
                test_mask = test_mask_y & test_mask_x
                
                self.console_logger.debug(f'Shapes of Xtr, Xv, Xte before clipping:  {Xtr.shape}, {Xv.shape}, {Xte.shape}')
                
                # Apply masks to both X and y
                Xtr, ytr = Xtr[train_mask], ytr[train_mask]
                Xv, yv = Xv[val_mask], yv[val_mask]
                Xte, yte = Xte[test_mask], yte[test_mask]
                
                self.console_logger.debug(
                    f"Fold {fold} after clipping (y): train={ytr.shape}, val={yv.shape}, test={yte.shape}"
                )                
                self.console_logger.debug(
                    f"Fold {fold} after clipping (X): train={Xtr.shape}, val={Xv.shape}, test={Xte.shape}"
                )                
            else:
                self.console_logger.debug(
                    f"Fold {fold}: train={len(Xtr)}, val={len(Xv)}, test={len(Xte)}"
                )                


            if self.scale:
                result = scale_split(
                    Xtr, ytr, Xv, yv, Xte, yte,
                    scale_type=self.scale_type,
                    merge=True
                )
                (
                    Xtr, ytr, Xv, yv, Xte, yte,
                    Xtr_val, ytr_val, Xte_merged, yte_merged
                ) = result
            else:
                # Create unscaled merged arrays
                Xtr_val = np.concatenate([Xtr, Xv], axis=0)
                ytr_val = np.concatenate([ytr, yv], axis=0)
                Xte_merged = Xte.copy()
                yte_merged = yte.copy()

            self.console_logger.debug(f'Generating fold: {fold}')
            self.console_logger.debug(f'Merged arrays shapes: Xtr_val={Xtr_val.shape}, ytr_val={ytr_val.shape}')
            self.console_logger.debug(f'Merged test shapes: Xte_merged={Xte_merged.shape}, yte_merged={yte_merged.shape}')

            yield Xtr, ytr, Xv, yv, Xte, yte, Xtr_val, ytr_val, Xte_merged, yte_merged



