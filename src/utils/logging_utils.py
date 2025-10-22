import os
import json
import socket
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from utils.custom_formatter import setup_logger

# Adjust this to your repo root import that defines SRC_DIR
from utils.paths import SRC_DIR


class ExperimentLogger:
    """
    Folder structure:
      {SRC_DIR}/{exp_type}/experiments/
        exp_{id:03d}_{exp_name}/
          trial_{YYYYMMDD_HHMMSS}/
            results.csv
            metadata.jsonl
            config_snapshot.json
            fold_000/...
            fold_001/...

    Usage:
        logger = ExperimentLogger(cfg)
        trial_dir = logger.begin_trial()   # creates trial_YYYYMMDD_HHMMSS
        # ... during training:
        path = logger.path(f"fold_{fold:03d}/model_best.pt")
        logger.append_result(trial=0, fold=fold, ...)
        logger.log({"anything": "you want"})
    """

    def __init__(self, cfg: Dict[str, Any], console_logger=None):
        self.cfg = cfg
        self.console_logger = console_logger or setup_logger("ExperimentLogger", level="INFO")

        # Base experiments dir by type
        self.exp = self.cfg.experiment
        exp_type = self.exp.type
        self.type_PRICE_EXPERIMENTS_DIR: Path = Path(SRC_DIR) / exp_type / "experiments"
        self.type_PRICE_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        # get name
        exp_name = self.exp.name

        # ---- try to reuse an existing experiment with the same name ----
        existing_dirs = [
            d for d in self.type_PRICE_EXPERIMENTS_DIR.iterdir()
            if d.is_dir() and d.name.startswith("exp_") and d.name.endswith(f"_{exp_name}")
        ]

        if existing_dirs:
            # If multiple match (unlikely), pick the one with the highest numeric id
            def _extract_num(p: Path) -> int:
                try:
                    return int(p.name.split("_")[1])
                except Exception:
                    return -1
            existing_dirs.sort(key=_extract_num, reverse=True)
            self.exp_dir: Path = existing_dirs[0]
            self.console_logger.warning(
                f"Experiment with name '{exp_name}' already exists: {self.exp_dir}. "
                f"If the experiment is of type `search`, 'trial_serch_best' will be overwritten!"
                f"Creating a new trial under this experiment."
            )
        else:
            # ---- Original behavior: allocate a new exp_{id:03d}_{name} ----
            existing = [
                d.name for d in self.type_PRICE_EXPERIMENTS_DIR.iterdir()
                if (self.type_PRICE_EXPERIMENTS_DIR / d).is_dir() and d.name.startswith("exp_")
            ]
            nums = []
            for dn in existing:
                parts = dn.split("_")
                try:
                    nums.append(int(parts[1]))
                except Exception:
                    pass
            next_id = max(nums) + 1 if nums else 1

            self.exp_dir: Path = self.type_PRICE_EXPERIMENTS_DIR / f"exp_{next_id:03d}_{exp_name}"
            self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Trial-scoped paths (set by begin_trial)
        self.trial_dir: Optional[Path] = None
        self.results_csv: Optional[Path] = None
        self.meta_path: Optional[Path] = None

        self.console_logger.info(f"Experiment directory: {self.exp_dir}")

    # ---- trial lifecycle -----------------------------------------------------

    def begin_trial(self, name= None) -> str:
        """
        Create a new trial subdir 'trial_{YYYYMMDD_HHMMSS}' and initialize
        results.csv, metadata.jsonl, config_snapshot.json.
        if `name` is passed, then creates a subdir 'trial_{name}' with appropriate formatting
        Returns trial directory path (str).
        """
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name is None: 
            id =  time 
        else: 
            if isinstance(name, str):
                id = name.lower().replace(" ", "_")
            elif isinstance(name, int):
                id = format(name, "03d")
            else:
                self.console_logger.warning(f"Type {type(name)} not supported for name: {name}")
                raise ValueError
            
        self.trial_dir = self.exp_dir / f"trial_{id}"
        self.trial_dir.mkdir(parents=True, exist_ok=True)

        # File paths for this trial
        self.results_csv = self.trial_dir / "results.csv"
        self.meta_path = self.trial_dir / "metadata.jsonl"

        # Initialize results.csv header
        if not os.path.exists(self.results_csv):
            with open(self.results_csv, "w") as f:
                f.write(
                    "trial,fold,"
                    "tr_loss,val_loss,test_loss,"
                    "tr_mae,val_mae,test_mae,"
                    "tr_directional_accuracy_pct,val_directional_accuracy_pct,test_directional_accuracy_pct,"
                    "seconds,model_path\n"
                )

        # Save config + small env stamp (per trial)
        # Moved to path 
        self.console_logger.info(f"Trial directory: {self.trial_dir}")
        return str(self.trial_dir)

    # ---- writers -------------------------------------------------------------

    def append_result(self, **kw):
        """
        Appends one fold's summary to this trial's results.csv.

        Expected keys:
          trial, fold,
          tr_loss, val_loss, test_loss,
          tr_mae,  val_mae,  test_mae,
          tr_directional_accuracy_pct, val_directional_accuracy_pct, test_directional_accuracy_pct,
          seconds, model_path
        """
        if self.results_csv is None:
            raise RuntimeError("begin_trial() must be called before append_result().")

        line = (
            f"{kw.get('trial',0)},"
            f"{kw['fold']},"
            f"{kw.get('tr_loss','')},{kw['val_loss']},{kw['test_loss']},"
            f"{kw.get('tr_mae','')},{kw.get('val_mae','')},{kw.get('test_mae','')},"
            f"{kw.get('tr_directional_accuracy_pct','')},{kw.get('val_directional_accuracy_pct','')},{kw.get('test_directional_accuracy_pct','')},"
            f"{kw.get('seconds','')},{kw.get('model_path','')}\n"
        )
        with open(self.results_csv, "a") as f:
            f.write(line)

    def log(self, obj: dict):
        """Append a JSON line to this trial's metadata.jsonl."""
        if self.meta_path is None:
            raise RuntimeError("begin_trial() must be called before log().")
        with open(self.meta_path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    # ---- path helper ---------------------------------------------------------

    def path(self, *parts) -> str:
        """
        Build a path **inside the current trial** and ensure parent folders exist.
        Example:
            logger.path(f"fold_{fold:03d}/model_best.pt")
        """
        if self.trial_dir is None:
            raise RuntimeError("begin_trial() must be called before path().")
        
        p = self.trial_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.console_logger.debug(f"fold_dir: {os.path.basename(p)}")
        if str(p).endswith("/") or p.suffix == "":
            p.mkdir(parents=True, exist_ok=True)
        # Save config + small env stamp (per trial)
        if "fold" in str(os.path.basename(p)): 
            time = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(p / "config_snapshot.json", "w") as f:
                cfg_dict = asdict(self.cfg)
                json.dump(
                    {
                        "cfg": cfg_dict,
                        "env": {
                            "host": socket.gethostname(),
                            "platform": platform.platform(),
                            "time": time,
                        },
                    },
                    f,
                    indent=2,
                )



        return str(p)
