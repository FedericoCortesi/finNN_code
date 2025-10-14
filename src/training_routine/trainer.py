import os, time
import numpy as np
from tensorflow import keras
from .metrics import directional_accuracy_pct

from utils.logging_utils import ExperimentLogger
from utils.custom_formatter import setup_logger
from .callbacks import VerboseLoss

class Trainer:
    def __init__(self, cfg: dict, logger: ExperimentLogger):
        self.cfg = cfg 
        self.logger = logger
        self.console_logger = setup_logger("Trainer", level="INFO")

    # compiles model to make it ready to fit data 
    def compile(self, model):
        # handle dir_acc as its a custom metric
        cfg_metrics = list(self.cfg["trainer"]["metrics"])

        resolved = []
        for m in cfg_metrics:
            if m == "dir_acc" or (isinstance(m, str) and m.lower() == "dir_acc"):
                resolved.append(directional_accuracy_pct)   # inject the function
            else:
                resolved.append(m)

        self.console_logger.debug(f"metrics: {resolved}")

        model.compile(optimizer=keras.optimizers.Adam(self.cfg["trainer"]["lr"]),
                      loss=self.cfg["trainer"]["loss"],
                      metrics=resolved)
        

    def fit_eval_fold(self, 
                      model, 
                      data:tuple, 
                      fold:int,
                      trial:int=0,
                      merge_train_val:bool=False): 
        
        # TODO: fit on test + val after
        # TODO: define in sample/oos metrics only

        # Get data from input and assing to arrays
        Xtr, ytr, Xv, yv, Xte, yte = data
        
        # identify directory for the trial 
        fold_dir = self.logger.path(f"trial_{trial:03d}/fold_{fold:03d}/")
        trial_dir = self.logger.path(f"trial_{trial:03d}/")
        
        # define callbacks
        es = keras.callbacks.EarlyStopping(
                monitor=self.cfg["experiment"]["monitor"],
                mode=self.cfg["experiment"]["mode"],
                patience=25,
                min_delta = 1e-12, # loss is very small
                restore_best_weights=True)
        
        ckpt = keras.callbacks.ModelCheckpoint(
        filepath=self.logger.path(f"trial_{trial:03d}/fold_{fold:03d}/model_best.keras"),
        monitor=self.cfg["experiment"]["monitor"],
        mode=self.cfg["experiment"]["mode"],
        # This works only for hyperparams tuning, does NOT carry weights
        # From one fold to the other
        save_best_only=True, 
        save_weights_only=False,  # set True if you only want weights (smaller file)
        verbose=0)
        
        cb = [es, ckpt, VerboseLoss()]
        
        # compile model passed in the class using built in function
        self.compile(model)
        
        # initial time
        t0 = time.time()

        print()
        self.console_logger.info(
            f"Fitting fold {fold:02d} (trial {trial:02d}) "
            f"on {Xtr.shape[0]} samples, val={Xv.shape[0]}, test={Xte.shape[0]}...")
        print()
        
        # fit model
        hist = model.fit(Xtr, ytr, 
                  validation_data=(Xv,yv), # not a problem since temporal order is preserved when building the df, every epoch
                  epochs=self.cfg["trainer"]["epochs"],
                  batch_size=self.cfg["trainer"]["batch_size"],
                  callbacks=cb, 
                  verbose=0, 
                  shuffle=False) # maybe doesn't matter for time series?
        
        # elapsed time
        secs = time.time()-t0

        self.console_logger.info("Evaluating...")
        # evaluate on validation and test 
        tr = model.evaluate(Xtr, ytr, verbose=0, return_dict=True) 
        v  = model.evaluate(Xv,  yv,  verbose=0, return_dict=True) 
        te = model.evaluate(Xte, yte, verbose=0, return_dict=True)

        # prefix keys right off the dicts (no manual metric-name plumbing)
        trmap = {f"tr_{k}": v for k, v in tr.items()}
        vmap  = {f"val_{k}": v for k, v in v.items()}
        temap = {f"test_{k}": v for k, v in te.items()}

        # ----- Best epoch from history (align with your monitor + mode)
        monitor_key = self.cfg["experiment"]["monitor"]
        mode = self.cfg["experiment"]["mode"].lower()
        hist_vals = np.array(hist.history[monitor_key])
        if mode == "min":
            best_epoch = int(np.argmin(hist_vals)) + 1  # Keras epochs are 1-based in logs
        else:
            best_epoch = int(np.argmax(hist_vals)) + 1

 
        # Print fold results
        print()        
        self.console_logger.info(
        f"\n[selection] tr_loss={trmap.get('tr_loss', np.nan):.6f} | "
        f"val_loss={vmap.get('val_loss', np.nan):.6f} | "
        f"test_loss={temap.get('test_loss', np.nan):.6f} | "
        f"\ntr_mae={trmap.get('tr_mae', np.nan):.6f} | "
        f"val_mae={vmap.get('val_mae', np.nan):.6f} | "
        f"test_mae={temap.get('test_mae', np.nan):.6f} | "
        f"\ntr_diracc={trmap.get('tr_directional_accuracy_pct', np.nan):.2f}% | "
        f"val_diracc={vmap.get('val_directional_accuracy_pct', np.nan):.2f}% | "
        f"test_diracc={temap.get('test_directional_accuracy_pct', np.nan):.2f}% | "
        f"\nbest_epoch={best_epoch}")
        print()        

        model_path = f"trial_{trial:03d}/fold_{fold:03d}/model_best.keras"
        self.console_logger.info(f"Saving model at {model_path}")
        self.logger.append_result(
        trial=trial, fold=fold,
        seconds=secs,
        model_path=self.logger.path(model_path),
        **trmap, **vmap, **temap
        )
        
        
        return trmap, vmap, temap
