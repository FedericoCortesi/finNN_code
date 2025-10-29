# We use this file to run optuna hyperparameters optimization.
# Given that optuna expects callables and dictionaries, we have
# to parse the dataclass config, make it into a dict that has the 
# same configuration of default.yaml, and later feed that as a 
# AppConfig dataclass to the model and trainer during hyperparams
# search.  

import optuna

import copy
from dataclasses import is_dataclass, asdict

from config.config_types import AppConfig
from utils.logging_utils import ExperimentLogger
from training_routine.trainer import Trainer            
from models import create_model 
from utils.custom_formatter import setup_logger


console_logger = setup_logger("Search_utils", "INFO")

def to_dict(cfg):
    if is_dataclass(cfg):
        return copy.deepcopy(asdict(cfg))
    # AppConfig likely has .to_dict(); if so prefer that:
    if hasattr(cfg, "to_dict"):
        return copy.deepcopy(cfg.to_dict())
    return copy.deepcopy(cfg)  # assume dict-like

def _get_attr(spec, key, default=None):
    return spec.get(key, default) if isinstance(spec, dict) else getattr(spec, key, default)

def _spec_type(spec):
    t = _get_attr(spec, "type")
    #console_logger.debug(f"t: {t}")
    if t is None:
        cls = spec.__class__.__name__.lower()
        if "float" in cls: return "float"
        if "int"   in cls: return "int"
        if "cat"   in cls or "categor" in cls: return "cat"
    return str(t).lower()

def _suggest_from_spec(trial, name, spec):
    t = _spec_type(spec)
    if t == "float":
        low = float(_get_attr(spec, "low")); high = float(_get_attr(spec, "high"))
        log = bool(_get_attr(spec, "log", False))
        return trial.suggest_float(name, low, high, log=log)
    if t == "int":
        low = int(_get_attr(spec, "low")); high = int(_get_attr(spec, "high"))
        return trial.suggest_int(name, low, high)
    if t in ("cat","categorical"):
        return trial.suggest_categorical(name, list(_get_attr(spec, "choices")))
    raise ValueError(f"Unknown spec type for {name}")

def sample_hparams_into_cfg(base_cfg, trial):
    """
    Read *.search blocks from YAML/dataclass, sample with Optuna, and
    return a NEW **dict** config with sampled values written into
    model.params[...] / trainer.params[...].
    """
    cfg = to_dict(base_cfg)
    
    # make out config to store configuration for models
    parsed_config = {}
    parsed_config["data"] = copy.deepcopy(cfg["data"])
    parsed_config["trainer"] = copy.deepcopy(cfg["trainer"])
    parsed_config["experiment"] = copy.deepcopy(cfg["experiment"])
    parsed_config["walkforward"] = copy.deepcopy(cfg["walkforward"])
    parsed_config["model"] =  copy.deepcopy(cfg["model"])
    parsed_config["model"]["name"] = copy.deepcopy(cfg["model"]["name"])
    model_keys = [k for k in cfg["model"]["search"].keys()] + [k for k in cfg["model"]["hparams"].keys()]
    console_logger.debug(f"model_keys: {model_keys}")
    console_logger.debug(f'cfg["model"]: {cfg["model"]}')
    for k in model_keys:
        parsed_config["model"]["hparams"][k] = None

    console_logger.debug(f'cfg["model"] after the cycle: {cfg["model"]}')
    console_logger.debug(f'parsed_config["model"]: {parsed_config["model"]}')


    # trainer.search → trainer.params (mirror to top-level if you also read there)
    tr_search = cfg.get("trainer", {}).get("search", {}) or {}
    tr_params = cfg.get("trainer", {}).get("hparams", {}) or {}
    tr_params_keys = tr_params.keys()
    for key, spec in tr_search.items():
        val = _suggest_from_spec(trial, f"trainer.{key}", spec)
        parsed_config["trainer"]["hparams"][key] = val
        if key in tr_params_keys:
            parsed_config["trainer"]["hparams"][key] = val

    # model.search → model.params
    mdl_search = cfg.get("model", {}).get("search", {}) or {}
    console_logger.debug(f"mdl_search: {mdl_search}")
    #mdl_hparams = cfg.get("model", {}).get("hparams", {}) or {}
    
    #sampled = {k: _suggest_from_spec(trial, f"model.{k}", spec) for k, spec in mdl_search.items()}

    # 1) Sample global model choices (activation, dropout) once
    if "activation" in mdl_search:
        parsed_config["model"]["hparams"]["activation"] = _suggest_from_spec(trial, "model.activation", mdl_search["activation"])
    if "dropout_rate" in mdl_search:
        parsed_config["model"]["hparams"]["dropout_rate"] = float(
            _suggest_from_spec(trial, "model.dropout_rate", mdl_search["dropout_rate"])
        )

    # 2) How many layers? (variable length)
    #    Optuna handles conditionals fine: later width_i params only exist if n_layers >= i+1
    if "n_layers" in mdl_search:
        n_layers = int(_suggest_from_spec(trial, "model.n_layers", mdl_search["n_layers"]))
    else:
        # fallback: if user didn't specify n_layers, deduce from current hidden_sizes length
        n_layers = len(cfg["model"]["hparams"].get("hidden_sizes", [128, 64]))
    parsed_config["model"]["hparams"]["n_layers"] = n_layers

    # 3) Which spec to use for each layer's width
    #    Prefer 'width' if provided; fallback to legacy 'n_hidden'
    width_spec = mdl_search.get("width") or mdl_search.get("n_hidden")
    if width_spec is None:
        # If neither width nor n_hidden is present, keep existing hidden_sizes
        # but ensure length equals n_layers (repeat first element if needed).
        current = cfg["model"]["hparams"].get("hidden_sizes", [128, 64])
        if len(current) != n_layers:
            base = current[0] if current else 128
            parsed_config["model"]["hparams"]["hidden_sizes"] = [int(base)] * max(1, n_layers)
    else:
        # 4) Sample a *separate* width for each layer
        #    Use distinct parameter names so Optuna can learn per-depth structure.
        hidden_sizes = []
        for i in range(n_layers):
            # Name matters: unique & stable keys help the sampler learn correlations.
            # e.g., model.width_0, model.width_1, ...
            wi = int(_suggest_from_spec(trial, f"model.width_{i}", width_spec))
            hidden_sizes.append(wi)
        parsed_config["model"]["hparams"]["n_hidden"] = hidden_sizes # pass the value so that it doesnt raise in 5)
        parsed_config["model"]["hparams"]["hidden_sizes"] = hidden_sizes
        
    # 5) Residual for items that have not been searched.
    # Harder than Trainer class since we have 2 effective levels.
    for k,v in parsed_config["model"]["hparams"].items():
        if v is None:
            console_logger.debug(f"k,v: {k, v}")
            console_logger.debug(f'cfg["model"]["hparams"][k]: {cfg["model"]["hparams"][k]}')
            parsed_config["model"]["hparams"][k] = cfg["model"]["hparams"][k]
    console_logger.debug(parsed_config)
    return parsed_config

def _make_report_cb(trial, mode: str = "min", patience: int = 5):
    """
    Returns a function: (epoch:int, val_metric:float) -> bool
    that reports to Optuna every call, and only allows pruning
    after 'patience' consecutive non-improvements.
    """
    best = float("inf") if mode == "min" else -float("inf")
    bad = 0

    def better(a, b):
        return a < b if mode == "min" else a > b

    def cb(epoch: int, val_metric: float) -> bool:
        nonlocal best, bad
        trial.report(val_metric, step=epoch)  # monotonically increasing step

        if better(val_metric, best):
            best = val_metric
            bad = 0
        else:
            bad += 1

        # Only prune if patience exceeded AND pruner agrees
        return bad > patience and trial.should_prune()

    return cb

# TODO: log results in trial_{n_fold}_{n_fold} so that you can 
# save just one conifg json and per fold and not bloat the other dirs.
def optuna_objective(trial: optuna.trial.Trial, 
                     config: AppConfig,
                     fold_data,
                     n_fold,
                     input_shp,
                     output_shp)-> float:
    # 1) sample hparams → a NEW cfg (dict or dataclass, depending on your function)

    trial_cfg = sample_hparams_into_cfg(config, trial)  # returns same "type" you pass in

    trial_cfg = AppConfig.from_dict(trial_cfg)

    console_logger.critical(f"Model params: {trial_cfg.model.hparams}")
    console_logger.critical(f"Trainer params: {trial_cfg.trainer.hparams}")


    # 2) fresh trainer per trial (avoid any state carry-over)
    trial_logger = ExperimentLogger(trial_cfg)
    name = f"{n_fold:03d}_{trial.number:03d}"
    trial_logger.begin_trial(name)

    trial_trainer = Trainer(trial_cfg, trial_logger)


    # 4) build a fresh model for this trial
    model = create_model(trial_cfg.model, input_shp, output_shp)

    # 5) epoch-wise reporting so ASHA can prune early
    mode = trial_cfg.experiment.mode.lower()  # "min" or "max"
    patience = trial_cfg.trainer.hparams.get("optuna_patience",10)
    report_cb = _make_report_cb(trial, 
                                mode=mode, 
                                patience=patience)
     
    
    # 6) fit/eval only on THIS fold’s (train,val). Your `data` tuple already contains them.
    fit_result = trial_trainer.fit_eval_fold(model,
                                             fold_data, 
                                             fold=n_fold,
                                             trial=trial.number, 
                                             merge_train_val=False,
                                             report_cb=report_cb)

    # 7) return the scalar according to direction
    val_dict = fit_result[1]
    val_metric = val_dict[config.experiment.monitor]
    return val_metric  # study direction handles min/max

