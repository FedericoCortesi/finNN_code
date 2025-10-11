import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from paths import MODELS_DIR
from gpu_test import gpu_test
from pipeline.walkforward import WFCVTrainer, WFConfig


# ----------------------------- Variables -------------------------------- #

BATCH_SIZE = 2**7
MAX_EPOCHS = 100

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


es = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
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
    wfcv = WFCVTrainer(config=config)
    
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

        print(f"\n===== Processing Fold {fold} =====\n")
        
        df_train, df_val, df_test = wfcv.obtain_datasets_fold(fold)

        X_train = df_train.drop(['y'], axis=1).values
        y_train = df_train['y'].values
        X_val = df_val.drop(['y'], axis=1).values
        y_val = df_val['y'].values

        best_val_mse = float('inf')
        best_params = None
        
        print(f"\n--- Tuning hyperparameters for Fold {fold} ---\n")
        for n_neurons in param_grid['n_neurons']:
            for learning_rate in param_grid['learning_rate']:

                print(f"\nFold: {fold} | n_neurons: {n_neurons} | learning_rate: {learning_rate}")

                model = build_model(
                    input_shape=X_train.shape[1],
                    n_neurons=n_neurons,
                    learning_rate=learning_rate
                )


                model.fit(
                    X_train, y_train,
                    epochs=MAX_EPOCHS, 
                    batch_size=BATCH_SIZE,
                    callbacks=[es, verbose_loss],
                    shuffle=False, #time series?
                    verbose=0 
                )
                
                val_mse = model.evaluate(X_val, y_val, verbose=1)
                
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_params = {'n_neurons': n_neurons, 'learning_rate': learning_rate}

        print(f"Best hyperparameters found: {best_params}")

        print(f"--- Refitting model on combined training & validation data for Fold {fold} ---")
        
        full_train_df = pd.concat([df_train, df_val])
        X_full_train = full_train_df.drop(['y'], axis=1).values
        y_full_train = full_train_df['y'].values

        X_test = df_test.drop(['y'], axis=1).values
        y_test = df_test['y'].values
        
        final_model = build_model(
            input_shape=X_full_train.shape[1],
            **best_params
        )

        final_es = keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True)


        final_model.fit(
            X_full_train, y_full_train,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[final_es, verbose_loss],
            verbose=0
        )

        mse_in_sample = final_model.evaluate(X_full_train, y_full_train, verbose=0)
        rmse_in_sample = np.sqrt(mse_in_sample)
        y_pred_in_sample = final_model.predict(X_full_train, verbose=0).flatten()
        dir_acc_in_sample = np.mean(np.sign(y_pred_in_sample) == np.sign(y_full_train)) * 100

        mse_out_of_sample = final_model.evaluate(X_test, y_test, verbose=0)
        rmse_out_of_sample = np.sqrt(mse_out_of_sample)
        y_pred_out_of_sample = final_model.predict(X_test, verbose=0).flatten()
        dir_acc_out_of_sample = np.mean(np.sign(y_pred_out_of_sample) == np.sign(y_test)) * 100
        
        print(f"In-sample RMSE: {rmse_in_sample:.6f} | In-sample Directional Accuracy: {dir_acc_in_sample:.2f}%")
        print(f"Out-of-sample RMSE: {rmse_out_of_sample:.6f} | Out-of-sample Directional Accuracy: {dir_acc_out_of_sample:.2f}%")

        model_path = MODELS_DIR / f"model_fold_{fold}.keras"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)  # make sure dir exists
        final_model.save(model_path)        

        fold_results = {
            'fold': fold,
            'best_hyperparameters': best_params,
            'in_sample_rmse': rmse_in_sample,
            'out_of_sample_rmse': rmse_out_of_sample,
            'in_sample_dir_acc': dir_acc_in_sample,
            'out_of_sample_dir_acc': dir_acc_out_of_sample,
            'model_path': model_path
        }
        all_fold_results.append(fold_results)

    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv('model_results.csv', index=False)
