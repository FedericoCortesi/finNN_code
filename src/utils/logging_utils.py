import os, json, time, socket, platform
from datetime import datetime
from pathlib import Path

from .paths import SRC_DIR
from .custom_formatter import setup_logger

console_logger = setup_logger("Experiment", level="INFO")

class ExperimentLogger:
    def __init__(self, cfg: dict):
        
        # instantiate cfg
        self.cfg = cfg

        # create/find for specific type dir
        exp_type = self.cfg["experiment"]["type"]
        TYPE_EXPERIMENTS_DIR = SRC_DIR / exp_type / "experiments"
        
        if not TYPE_EXPERIMENTS_DIR.exists():
            TYPE_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
            console_logger.warning(f"Directory {TYPE_EXPERIMENTS_DIR} doesn't exist. Check type of experiment")

        # find number id first
        existing = [
            d for d in os.listdir(TYPE_EXPERIMENTS_DIR)
            if os.path.isdir(os.path.join(TYPE_EXPERIMENTS_DIR, d)) and d.startswith("exp_")
        ]
        # extract numeric prefix, e.g. exp_012 -> 12
        nums = []
        for d in existing:
            parts = d.split("_")
            try:
                nums.append(int(parts[1]))
            except Exception:
                pass
        next_id = max(nums) + 1 if nums else 1

        # timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # create experiment directory
        exp_name = self.cfg["experiment"]["name"]
        self.exp_dir = os.path.join(TYPE_EXPERIMENTS_DIR, f"exp_{next_id:03d}_{ts}_{exp_name}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # file paths
        self.results_csv = os.path.join(self.exp_dir, "results.csv")
        self.meta_path = os.path.join(self.exp_dir, "metadata.jsonl")

        # header for your new table structure
        with open(self.results_csv, "w") as f:
            f.write(
                "trial,fold,"
                "tr_loss,val_loss,test_loss,"
                "tr_mae,val_mae,test_mae,"
                "tr_diracc,val_diracc,test_diracc,"
                "seconds,model_path\n"
            )
        # save config + small env stamp
        with open(os.path.join(self.exp_dir, "config_snapshot.json"), "w") as f:
            json.dump(
                {
                    "cfg": cfg,
                    "env": {
                        "host": socket.gethostname(),
                        "platform": platform.platform(),
                        "time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    },
                },
                f,
                indent=2,
            )

    
    def append_result(self, **kw):
            """
            Appends one fold's summary to results.csv.
            Expected keys:
            trial, fold,
            tr_loss, val_loss, test_loss,
            tr_mae,  val_mae,  test_mae,
            tr_diracc, val_diracc, test_diracc,
            seconds, model_path
            """
            line = (
                f"{kw.get('trial',0)},"
                f"{kw['fold']},"
                f"{kw.get('tr_loss','')},{kw['val_loss']},{kw['test_loss']},"
                f"{kw.get('tr_mae','')},{kw.get('val_mae','')},{kw.get('test_mae','')},"
                f"{kw.get('tr_diracc','')},{kw.get('val_diracc','')},{kw.get('test_diracc','')},"
                f"{kw.get('seconds','')},{kw.get('model_path','')}\n"
            )
            with open(self.results_csv, "a") as f:
                f.write(line)


    def log(self, obj: dict):
        with open(self.meta_path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def path(self, *parts):
        p = os.path.join(self.exp_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
