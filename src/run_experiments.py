import argparse
from functools import partial
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from config.config_types import AppConfig
from utils.logging_utils import ExperimentLogger
from training_routine.trainer import Trainer            
from pipeline.walkforward import WFCVGenerator
#from pipeline.wf_config import WFConfig
from hyperparams_search.search_utils import optuna_objective, sample_hparams_into_cfg
#from hyperparams_search.torch_estimator import TorchFoldEstimator
#from hyperparams_search.randomsearch import RandomSearch

from utils.gpu_test import gpu_test
from utils.paths import CONFIG_DIR, DATA_DIR
from utils.custom_formatter import setup_logger
from models import create_model 

import time



def main():
    # -------- argparse setup --------
    parser = argparse.ArgumentParser(description="Run a walk-forward training experiment.")
    parser.add_argument("--config", type=str, default=str("default.yaml"),
                        help="Path to YAML config file.")
    parser.add_argument("--data", type=str, default=None,
                        help="Optional dataset path to override config[data][df_path].")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Optional name to override config[experiment][name].")
    args = parser.parse_args()


    # setup logger
    console_logger = setup_logger("Experiment", level="DEBUG")

    # --- GPU check (PyTorch) ---
    gpu_test()
    console_logger.info("GPU check complete.")

    # -------- load config --------
    cfg = AppConfig.from_dict(f"{CONFIG_DIR}/{args.config}")

    # Override config with CLI args if provided
    if args.data:
        cfg.data["df_path"] = args.data
    if args.exp_name:
        cfg.experiment["name"] = args.exp_name


    # -------- data + components --------
    #logger = ExperimentLogger(cfg)
    #data_path = cfg.data.get("df_path")

    df_long = cfg.data.get("df_long", None)  

    if args.data:
        df = args.data
        wf = WFCVGenerator(df_long=df, 
                           config=cfg.walkforward)
    else:
        wf = WFCVGenerator(df_long=df_long,
                           config=cfg.walkforward)

    console_logger.debug(f'cfg.walkforward: {cfg.walkforward}')
    

    # instantiate trainer (PyTorch)
    #trainer = Trainer(cfg, logger)

    # model input size: number of lags (columns are constant across folds)
    input_shape = cfg.walkforward.lags            # int is fine; build_model handles it
    min_folds = cfg.walkforward.min_folds
    max_folds = cfg.walkforward.max_folds

    console_logger.debug(f"cfg.model.name: {cfg.model.name}")

    # define shapes
    def make_input_shape(c):
        if cfg.model.name.lower() == "mlp":
            return (c.walkforward.lags, )
        elif cfg.model.name.lower() == "simplecnn":
            return (1, cfg.walkforward.lags)  # (C, L)
        elif cfg.model.name.lower() == "lstm":
            return (cfg.walkforward.lags, 1)
        else:
            console_logger.warning(f"Model: {cfg.model.name} not recognized!")
            raise ValueError

    # ouput shape is constant
    output_shape = cfg.walkforward.lookback + 1 if cfg.walkforward.lookback is not None else 1


    # Get bool for search
    hyperparams_search = cfg.experiment.hyperparams_search
    console_logger.debug(f' hyperparams_search {hyperparams_search}')
    # Create logger and trainer if just one specification
    if not hyperparams_search:
        console_logger.warning("No hyperparams search for this experiment")
        logger = ExperimentLogger(cfg) 
        trainer = Trainer(cfg, logger)
        logger.begin_trial()

    # See if a there is a df_master in the config file
    if cfg.data["df_master"] is not None:
        df_master_path =  cfg.data["df_master"]
        df_master = pd.read_parquet(f"{DATA_DIR}/{df_master_path}")
        console_logger.debug(f"provided df master: {df_master_path}\n{df_master.head()}")
    else:
        df_master = None # needed for wf.folds()
        console_logger.debug(f'df_master:\n{df_master}')

    # Warning before starting
    if cfg.walkforward.scale == True and cfg.trainer.hparams["loss"].lower() == "qlike":
        console_logger.warning(f"Attention: scaling data and 'qlike' loss might lead to bad results because of skewness.")

    # -------- train per fold --------
    for fold, data in enumerate(wf.folds(df_master=df_master)):
        console_logger.warning(f'Fold: {fold}')
        
        if min_folds is not None and fold <  min_folds:
            continue
        if max_folds is not None and fold >= max_folds:
            break  # allow running subset of folds

        if hyperparams_search:
            # ---- pick a single fold 'data' and run Optuna on it ----
            direction = "minimize" if cfg.experiment.mode.lower() == "min" else "maximize"
            n_trials  = getattr(cfg.experiment, "n_trials", 50)
            n_jobs    = 1  # for a single fold on one machine; raise if you parallelize

            # Compute (fixed) input shape for this base cfg; it won't change with hparams

            input_shape = make_input_shape(cfg)

            # ---- build study (TPE + ASHA) and run for this fold ----
            study = optuna.create_study(
                direction=direction,
                sampler=TPESampler(seed=cfg.experiment.random_state, multivariate=True),
                pruner=SuccessiveHalvingPruner(min_resource=3, reduction_factor=3),  # ASHA-like pruning
            )

            # Define a "bound" objective function with extra args pre-filled
            # the logger is instatiated in this function so that it can save 
            # the random parameters
            objective_fn = partial(optuna_objective,
                                config=cfg,
                                fold_data=data,
                                n_fold=fold,
                                input_shp=input_shape,
                                output_shp=output_shape)
            
            study.optimize(objective_fn, 
                           n_trials=n_trials, 
                           n_jobs=n_jobs)

            # ---- log & (optionally) retrain best-once for the fold ----
            best_number = study.best_trial.number
            console_logger.info(f"[Fold {fold}] Best trial number:  {study.best_trial.number:3d}")
            console_logger.info(f"[Fold {fold}] Best params: {study.best_params}")

            # Optional: rebuild a cfg from best params and do a final “full” run (e.g., full epochs)
            best_cfg = sample_hparams_into_cfg(cfg, optuna.trial.FixedTrial(study.best_params))
            best_cfg = AppConfig.from_dict(best_cfg) # make it a dict
            console_logger.debug(f"best_cfg: {best_cfg}")
            
            # instantiate logger and navigate to the trial 
            best_logger  = ExperimentLogger(best_cfg)
            best_logger.begin_trial(name="search_best")

            best_trainer = Trainer(best_cfg, best_logger)
            best_model   = create_model(best_cfg.model, input_shape, output_shape)

            # If you keep shorter epochs during search, you can override here:
            # best_cfg.trainer.params["epochs"] = cfg.trainer.params["epochs"]  # full epochs

            _ = best_trainer.fit_eval_fold(best_model,
                                            data, 
                                            fold=fold, 
                                            trial=-1,
                                            merge_train_val=True,
                                            report_cb=None)

            # --- log best results ---
            #console_logger.info(f"[Fold {fold}] Best params: {result['best_params']}")
            #console_logger.info(f"[Fold {fold}] Best {cfg.experiment.monitor}: {result['best_selection_score']:.6f}")
            
            #time.sleep(10)


        else:
            input_shape = make_input_shape(cfg)
            
            # Keep model creation inside the loop to avoid weight leakage across folds
            model = create_model(cfg.model, input_shape, output_shape)       

            if fold == 0:
                console_logger.debug(f"model: {model}")

            trainer.fit_eval_fold(model, data, trial=0, fold=fold)

    console_logger.warning("Training completed!")

if __name__ == "__main__":
    main()
