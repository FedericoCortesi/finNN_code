import argparse
import yaml
import pandas as pd
from pathlib import Path

from utils.logging_utils import ExperimentLogger
from training_routine.trainer import Trainer
from pipeline.walkforward import WFCVGenerator
from pipeline.wf_config import WFConfig
from models.mlp import build_model
from utils.paths import CONFIG_DIR
from utils.gpu_test import gpu_test
from utils.custom_formatter import setup_logger



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
    logger = setup_logger("Experiment", level="INFO")

    # --- run GPU test once ---
    gpu_test()
    logger.info("GPU check complete.")


    # -------- load config --------
    cfg = yaml.safe_load(open(CONFIG_DIR / args.config))

    # Override config with CLI args if provided
    if args.data:
        cfg["data"]["df_path"] = args.data
    if args.exp_name:
        cfg["experiment"]["name"] = args.exp_name

    # -------- data + components --------
    logger = ExperimentLogger(cfg)

    # use data path if present
    data_path = cfg.get("data", {}).get("df_path")

    # import wfcv
    wf_config = WFConfig(**cfg["walkforward"])
    
    if data_path:
        df = pd.read_parquet(data_path)
        wf = WFCVGenerator(df_long=df, config=wf_config)
    else:
        wf = WFCVGenerator(config=wf_config)

    # instantiate trainer    
    trainer = Trainer(cfg, logger)

    # Only number of rows change, columns stay constant 
    input_shape = cfg["walkforward"]["lags"]

    max_folds = cfg["walkforward"]["max_folds"]

    # -------- train per fold --------
    for fold, data in enumerate(wf.folds()):
        if max_folds is not None and fold >= max_folds:
            break  # allow running subset of folds

        # can i take this out the for loop or do i risk leakage?
        # better to leave it here so im sure weights are initialized 
        # at each fold
        model = build_model(cfg["model"]["hparams"], input_shape)

        trainer.fit_eval_fold(model, data, trial=0, fold=fold)


if __name__ == "__main__":
    main()
