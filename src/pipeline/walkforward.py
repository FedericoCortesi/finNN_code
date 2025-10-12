from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


from typing import Callable, Dict, Tuple, List

from pipeline.preprocessing import preprocess
from pipeline.wf_config import WFConfig

from utils.custom_formatter import setup_logger


class WFCVGenerator:
    def __init__(
        self,
        config:WFConfig,
        id_col: str = "permno", 
        df_long: pd.DataFrame = pd.DataFrame({}),                  # preprocessed long df
        time_col: str = "t",
        value_cols: List[str] = ["ret", "volume"],      
        target_col: str = "ret",                        # e.g. "ret" or "close_lead1"
        scaler_factory: Callable = StandardScaler.fit_transform,
    ):
        self.df = df_long.copy()
        self.id_col, self.time_col = id_col, time_col
        self.value_cols, self.target_col = value_cols, target_col
        self.config = config
        self.scaler_factory = scaler_factory

        # build wide dict once (cheap, readable)
        if df_long.size == 0:
            self.df = preprocess()[["t", "ret", "permno"]].copy()

        # define number of trading days
        self.T = self.df["t"].nunique()    

        # Call important functions
        self.stamps_and_windows_array = self._make_windows()
        self.df_master = self._build_master_df() 

        # setup info logger
        self.info_logger = setup_logger("WFCVGenerator", "INFO")
        self.info_logger.debug(self.config.summary())



    def _walk_forward(self):
        # self.config.lagsast train index
        t_0 = 0

        result = []

        folds_count = 0  

        while True:
            a, b = t_0 , t_0 + self.config.T_train # train
            c = b + self.config.T_val # validation
            d = c + self.config.T_test # test
            if d > self.T: 
                break
        
            result.append([slice(a,b), slice(b,c), slice(c,d)])
            t_0 += self.config.step
            folds_count += 1
        self.folds_count = folds_count
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
            t_col="t" 
            ) -> pd.DataFrame:
        
        # Obtain windows 
        windows = self.stamps_and_windows_array[6].copy()


        # 1) Wide once: rows=permno, cols=t (sorted) 
        # Values has to be ret!!!!!!!!!!!!!
        W = (self.df.pivot(index=self.id_col, columns=self.time_col, values=self.target_col)
                    .sort_index(axis=1))
        col_index = W.columns  # Int64Index of t's
        W = W.astype("float64")  # otherwise numpy doesnt coerce pd nans
        V = W.values           # ndarray (n_permno × n_time), avoids per-iteration DataFrame ops

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
            block = block[ok_rows]

            # 5) Build a small DataFrame for this window; rename cols
            #    First L columns → features, last col → y
            df_block = pd.DataFrame(block, columns=out_cols)

            # add window informatio
            df_block["window"] = [(a,b)]*len(df_block)

            # (optional) If you want to keep which permno each row is:
            # keep_ids = W.index.to_numpy()[ok_rows]
            # df_block.insert(0, "permno", keep_ids)
            # df_block.insert(1, "t_end", b)

            out_frames.append(df_block)

        # 6) Single concat at the end
        if out_frames:
            df_base = pd.concat(out_frames, ignore_index=True)
        else:
            df_base = pd.DataFrame(columns=out_cols)

        return df_base


    def obtain_datasets_fold(self, fold:int):
        """
        Given a number of fold, returns the train val and test dataframe  
        """

        # Back out windows
        train_stamps, train_windows, val_stamps, val_windows, test_stamps, test_windows, all_windows = self.stamps_and_windows_array

        # Build master df
        df_master = self.df_master.copy()

        # make the dataframes
        df_train = df_master[df_master["window"].isin(train_windows[fold])].drop(columns="window")
        df_val = df_master[df_master["window"].isin(val_windows[fold])].drop(columns="window")
        df_test = df_master[df_master["window"].isin(test_windows[fold])].drop(columns="window")

        return df_train, df_val, df_test
    
    def folds(self):
        """Yield train/val/test arrays for each fold."""
        for fold in range(self.folds_count):
            df_train, df_val, df_test = self.obtain_datasets_fold(fold)

            Xtr, ytr = df_train.drop(columns=["y"]).values, df_train["y"].values
            Xv, yv   = df_val.drop(columns=["y"]).values,   df_val["y"].values
            Xte, yte = df_test.drop(columns=["y"]).values,  df_test["y"].values

            yield Xtr, ytr, Xv, yv, Xte, yte




