import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers #type:ignore
from tensorflow.keras.regularizers import l2 #type:ignore

from utils.paths import MODELS_DIR
from utils.gpu_test import gpu_test
from pipeline.walkforward import WFCVGenerator, WFConfig

# ----------------------------- Utilities -------------------------------- #

def to_f32(x):  # enforce float32 (helps TF perf & avoids NaN surprises)
    return np.asarray(x, dtype=np.float32)

def assert_finite(name, arr):
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))[:5].ravel()
        raise ValueError(f"{name} has non-finite values at indices: {bad}")
    
class VerboseLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_total = time.time()
    def on_epoch_begin(self, epoch, logs=None):
        self.start_epoch = time.time()
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_epoch
        now = time.strftime("%H:%M:%S", time.localtime())
        loss = logs.get("loss", float("nan"))
        vloss = logs.get("val_loss", float("nan"))
        print(f"[{now}] Epoch {epoch+1:03d} | loss={loss:.8f} | val_loss={vloss:.8f} | {elapsed:.2f}s")


es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
verbose_loss = VerboseLoss()

# ------------------------------- Model ---------------------------------- #

def build_model(input_shape, 
                         n_hidden_layers=2, 
                         n_neurons=32, 
                         dropout_rate=0.2, 
                         activation='relu',
                         l2_reg=0.001,
                         learning_rate=0.001):
    model = keras.Sequential()
    
    model.add(layers.Input(shape=(input_shape,)))
    
    for _ in range(n_hidden_layers):
        model.add(layers.Dense(
            n_neurons, 
            activation=activation,
            kernel_regularizer=l2(l2_reg) 
        ))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
            
    model.add(layers.Dense(1))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

if __name__ == "__main__":
    gpu_test()  # prints GPU info once

    # configuration and splits
    config = WFConfig()
    wfcv = WFCVGenerator(config=config)
    
    all_fold_results = []

    param_grid = {
        'n_hidden_layers': [1, 2, 3],
        'n_neurons': [16, 32, 64],
        'dropout_rate': [0.0, 0.2, 0.5],
        'activation': ['relu', 'tanh'],
        'l2_reg': [0.0, 0.001, 0.01],
        'learning_rate': [0.01, 0.001, 0.0001]
    }

    for fold in range(config.folds):
        # Just because it stopped there last iteration

        print(f"\n===== Processing Fold {fold} =====\n")
        
        df_train, df_val, df_test = wfcv.obtain_datasets_fold(fold)

        X_train = df_train.drop(['y'], axis=1).values
        y_train = df_train['y'].values
        X_val = df_val.drop(['y'], axis=1).values
        y_val = df_val['y'].values

        best_val_mse = float('inf')
        best_params = None
        
        # --- Batch-size sweep (speed test) ---
        from statistics import mean

        class EpochTimer(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self._times = []
            def on_epoch_begin(self, epoch, logs=None):
                self._t0 = time.time()
            def on_epoch_end(self, epoch, logs=None):
                self._times.append(time.time() - self._t0)

        BATCHES = [2048, 4096, 8192, 16384, 32768, 65536]  # grow until OOM / no further speedup
        WARMUP_EPOCHS = 1      # allow XLA + autotune to settle
        MEASURE_EPOCHS = 3     # average over a few epochs

        results = []  # (bs, steps, sec_per_epoch, val_mse)

        for bs in BATCHES:
            print(f"\nFold {fold} | Trying batch_size={bs}")
            steps = int(np.ceil(len(X_train) / bs))
            print(f"  steps/epoch ≈ {steps}")

            keras.backend.clear_session()
            model = build_model(
                input_shape=X_train.shape[1],
                n_neurons=32,           # fixed for speed test
                learning_rate=1e-3      # fixed for speed test
            )

            timer = EpochTimer()
            # 1) warmup (not timed)
            model.fit(
                X_train, y_train,
                epochs=WARMUP_EPOCHS,
                batch_size=bs,
                shuffle=False,
                verbose=0
            )
            # 2) timed epochs
            model.fit(
                X_train, y_train,
                epochs=MEASURE_EPOCHS,
                batch_size=bs,
                shuffle=False,
                callbacks=[timer],
                verbose=0
            )
            sec_per_epoch = mean(timer._times)
            val_mse = model.evaluate(X_val, y_val, verbose=0)

            print(f"  avg sec/epoch: {sec_per_epoch:.3f}s | val_mse: {val_mse:.6f}")
            results.append((bs, steps, sec_per_epoch, val_mse))

        # Pick the fastest batch that keeps validation reasonable
        results.sort(key=lambda x: x[2])  # sort by sec/epoch asc
        best_bs = results[0][0]
        print("\nBatch-size timing summary:")
        for bs, steps, sec, mse in results:
            print(f"  bs={bs:6d} | steps={steps:4d} | {sec:6.3f}s/epoch | val_mse={mse:.6f}")
        print(f"\n=> Fastest batch_size chosen for final training: {best_bs}")

        best_params = {'n_neurons': 32, 'learning_rate': 1e-3}
