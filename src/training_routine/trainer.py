import os, time
import numpy as np
from tensorflow import keras

from utils.logging_utils import ExperimentLogger
from utils.custom_formatter import setup_logger
from .callbacks import VerboseLoss

class Trainer:
    def __init__(self, cfg: dict, logger: ExperimentLogger):
        self.cfg = cfg 
        self.logger = logger
        self.console_logger = setup_logger("Trainer")

    # compiles model to make it ready to fit data 
    def compile(self, model):
        model.compile(optimizer=keras.optimizers.Adam(self.cfg["trainer"]["lr"]),
                      loss=self.cfg["trainer"]["loss"],
                      metrics=self.cfg["trainer"]["metrics"])
        

    def fit_eval_fold(self, 
                      model, 
                      data:tuple, 
                      fold:int,
                      trial:int=0): 
        # TODO: fit on test + val after 
        # TODO: define directional accuracy as a callback method

        # Get data from input and assing to arrays
        Xtr, ytr, Xv, yv, Xte, yte = data
        
        # identify directory for the trial 
        fold_dir = self.logger.path(f"trial_{trial:03d}/fold_{fold:03d}/")
        trial_dir = self.logger.path(f"trial_{trial:03d}/")
        
        # define callbacks
        es = keras.callbacks.EarlyStopping(
                monitor=self.cfg["experiment"]["monitor"],
                mode=self.cfg["experiment"]["mode"],
                patience=10,
                min_delta = 1e-8, # loss is very small
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
        model.fit(Xtr, ytr, 
                  validation_data=(Xv,yv), # not a problem since temporal order is preserved when building the df, every epoch
                  epochs=self.cfg["trainer"]["epochs"],
                  batch_size=self.cfg["trainer"]["batch_size"],
                  callbacks=cb, 
                  verbose=0, 
                  shuffle=False) # maybe doesn't matter for time series?
        
        # elapsed time
        secs = time.time()-t0

        # evaluate on validation and test 
        tr = model.evaluate(Xtr,ytr,verbose=0) 
        v = model.evaluate(Xv,yv,verbose=0) 
        te = model.evaluate(Xte,yte,verbose=0)

        # create dictionaries with metrics
        names = ["loss"] + [m if isinstance(m,str) else m.name for m in model.metrics]
        trmap = dict(zip(["tr_"+n for n in names], tr))
        vmap = dict(zip(["val_"+n for n in names], v))
        temap = dict(zip(["test_"+n for n in names], te))

        # add directional accuracy
        yhat_tr = model.predict(Xtr, verbose=0).squeeze()
        yhat_v = model.predict(Xv, verbose=0).squeeze()
        yhat_te = model.predict(Xte, verbose=0).squeeze()

        trmap["tr_diracc"] = np.mean(np.sign(yhat_tr) == np.sign(ytr)) * 100
        vmap["val_diracc"] = np.mean(np.sign(yhat_v) == np.sign(yv)) * 100
        temap["test_diracc"] = np.mean(np.sign(yhat_te) == np.sign(yte)) * 100
        
        # Print fold results
        print()        
        self.console_logger.info(
            f" tr_loss={trmap['tr_loss']:.6f} | val_loss={vmap['val_loss']:.6f} | test_loss={temap['test_loss']:.6f}"
            f" | tr_diracc={trmap['tr_diracc']:.6f} | val_diracc={vmap['val_diracc']:.6f} | test_diracc={temap['test_diracc']:.6f}"
        )
        print()        

        self.console_logger.info("Saving model")
        self.logger.append_result(
        trial=trial, fold=fold,
        tr_loss=trmap["tr_loss"], val_loss=vmap["val_loss"], test_loss=temap["test_loss"],
        tr_mae=trmap.get("tr_mae",""), val_mae=vmap.get("val_mae",""), test_mae=temap.get("test_mae",""),
        tr_diracc=trmap.get("tr_diracc",""), val_diracc=vmap.get("val_diracc",""), test_diracc=temap.get("test_diracc",""),
        seconds=secs,
        model_path=self.logger.path(f"trial_{trial:03d}/fold_{fold:03d}/model_best.keras"))
        
        return trmap, vmap, temap
