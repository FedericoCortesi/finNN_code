# TODO: make this for walkforward
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import yaml
import copy

# ---- Search spec types ----
@dataclass
class FloatSpec:
    low: float
    high: float
    log: bool = False
    type: str = "float"

@dataclass
class IntSpec:
    low: int
    high: int
    type: str = "int"

@dataclass
class CatSpec:
    choices: List[Any]
    type: str = "cat"

SearchSpec = Union[FloatSpec, IntSpec, CatSpec]

def _parse_spec(d: Dict[str, Any]) -> SearchSpec:
    t = d["type"]
    if t == "float": return FloatSpec(**d)
    if t == "int":   return IntSpec(**d)
    if t == "cat":   return CatSpec(**d)
    raise ValueError(f"Unknown search type: {t}")

# ---- Model / Trainer blocks ----
@dataclass
class ModelConfig:
    name: str
    # model kwargs (e.g., hidden sizes, activation, etc.)
    hparams: Dict[str, Any] = field(default_factory=dict)
    # hyperparam search space for model__*
    search: Dict[str, SearchSpec] = field(default_factory=dict)



@dataclass
class TrainerConfig:
    # trainer kwargs (lr, wd, batch_size, patience, etc.)
    hparams: Dict[str, Any] = field(default_factory=dict)
    # hyperparam search space for trainer__*
    search: Dict[str, SearchSpec] = field(default_factory=dict)

@dataclass
class WFConfig:
    """Configuration for Walk-Forward Cross-Validation."""

    target_col: str = "ret"    # target column for regression
    lookback: int = 0          # Last N observations to include in the y vector 
    ratio_train: int = 3       # x test
    ratio_val: int = 1         # x test
    ratio_test: int = 1        # base value = step
    step: int = 251            # trading days per 'year'
    lags: int = 20             # number of past days as features
    max_folds: Optional[int] = None  # optional cap on folds
    scale: bool = False  # optional scale or not
    clip: bool = False

    def __post_init__(self):
        # Derived absolute lengths
        self.T_train = self.ratio_train * self.step
        self.T_val   = self.ratio_val   * self.step
        self.T_test  = self.ratio_test  * self.step

        # Validation
        for name in ["ratio_train", "ratio_val", "ratio_test", "step", "lags"]:
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive integer, got {v}")

        v = getattr(self, "lookback")
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {v}")
            

        assert self.lookback < self.lags, f"Lags must be longer than lookback, got {self.lookback} and {self.lags} instead"

    def summary(self) -> str:
        return (
            f"WFConfig(train={self.T_train}d, val={self.T_val}d, "
            f"test={self.T_test}d, lags={self.lags}, step={self.step}, "
            f"max_folds={self.max_folds})"
        )

@dataclass
class ExperimentConfig:
    name: str = "exp"
    hyperparams_search: bool = False
    monitor: str = "val_loss"
    mode: str = "min"
    type: str = "price_prediction"
    n_trials: int = 20
    random_state: Optional[int] = None

# ---- Root config ----
@dataclass
class AppConfig:
    model: ModelConfig
    trainer: TrainerConfig
    walkforward: WFConfig
    experiment: ExperimentConfig
    data: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(path_or_dict: str) -> "AppConfig":
        if isinstance(path_or_dict, dict):
            raw = path_or_dict
        if isinstance(path_or_dict, str):
            if path_or_dict.endswith("yaml"):
                with open(path_or_dict, "r") as f:
                    raw = yaml.safe_load(f)

        # parse model
        m = raw["model"]
        m_search = {k: _parse_spec(v) for k, v in m.get("search", {}).items()}
        model = ModelConfig(name=m["name"], hparams=m.get("hparams", {}), search=m_search)

        # parse trainer
        t = raw.get("trainer", {})
        t_search = {k: _parse_spec(v) for k, v in t.get("search", {}).items()}
        trainer = TrainerConfig(hparams=t.get("hparams", {}), search=t_search)

        wf = WFConfig(**raw["walkforward"])
        exp = ExperimentConfig(**raw.get("experiment", {}))
        data = raw.get("data", {})

        return AppConfig(model=model, trainer=trainer, walkforward=wf, experiment=exp, data=data)
    
    def to_dict(self) -> dict:
        from dataclasses import asdict
        return copy.deepcopy(asdict(self))
