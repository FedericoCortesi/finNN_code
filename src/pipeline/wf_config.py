from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class WFConfig:
    """Configuration for Walk-Forward Cross-Validation."""

    ratio_train: int = 3       # x test
    ratio_val: int = 1         # x test
    ratio_test: int = 1        # base value = step
    step: int = 251            # trading days per 'year'
    lags: int = 20             # number of past days as features
    max_folds: Optional[int] = None  # optional cap on folds

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

    def summary(self) -> str:
        return (
            f"WFConfig(train={self.T_train}d, val={self.T_val}d, "
            f"test={self.T_test}d, lags={self.lags}, step={self.step}, "
            f"max_folds={self.max_folds})"
        )
