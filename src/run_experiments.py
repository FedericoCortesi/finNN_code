# main_torch.py
import argparse
import yaml
import pandas as pd
from pathlib import Path

from utils.logging_utils import ExperimentLogger
from training_routine.trainer import Trainer            
from pipeline.walkforward import WFCVGenerator
from pipeline.wf_config import WFConfig
from utils.gpu_test import gpu_test
from utils.paths import CONFIG_DIR
from utils.custom_formatter import setup_logger

from models import create_model 

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
    console_logger = setup_logger("Experiment", level="INFO")

    # --- GPU check (PyTorch) ---
    gpu_test()
    console_logger.info("GPU check complete.")

    # -------- load config --------
    cfg = yaml.safe_load(open(CONFIG_DIR / args.config))

    # Override config with CLI args if provided
    if args.data:
        cfg["data"]["df_path"] = args.data
    if args.exp_name:
        cfg["experiment"]["name"] = args.exp_name

    # -------- data + components --------
    logger = ExperimentLogger(cfg)

    data_path = cfg.get("data", {}).get("df_path")

    # walk-forward config/generator
    wf_config = WFConfig(**cfg["walkforward"])
    if data_path:
        df = pd.read_parquet(data_path)
        wf = WFCVGenerator(df_long=df, config=wf_config)
    else:
        wf = WFCVGenerator(config=wf_config)

    # instantiate trainer (PyTorch)
    trainer = Trainer(cfg, logger)

    # model input size: number of lags (columns are constant across folds)
    input_shape = cfg["walkforward"]["lags"]            # int is fine; build_model handles it
    max_folds = cfg["walkforward"]["max_folds"]

    if cfg["model"]["name"].lower() == "cnn1d":
        input_shape = (1, cfg["walkforward"]["lags"])  # (C, L)
    elif cfg["model"]["name"].lower() == "mlp":
        input_shape = (cfg["walkforward"]["lags"],)
    else:
        console_logger.warning(f"Model: {cfg["model"]["name"]} not recognized!")

    logger.begin_trial()

    # -------- train per fold --------
    for fold, data in enumerate(wf.folds()):
        if max_folds is not None and fold >= max_folds:
            break  # allow running subset of folds

        # Keep model creation inside the loop to avoid weight leakage across folds
        model = create_model(cfg["model"], input_shape)       

        console_logger.critical(f"model: {model}")

        trainer.fit_eval_fold(model, data, trial=0, fold=fold)

    console_logger.warning("Training completed!")

if __name__ == "__main__":
    main()
