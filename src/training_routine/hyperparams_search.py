# hyperparam_search.py
import math, random
import numpy as np
from typing import Callable, Dict, Any

class HyperparamSearch:
    """
    Orchestrates random search. Delegates *all training* to a provided Trainer instance.
    """
    def __init__(
        self,
        trainer,                               # <- your Trainer instance (already constructed)
        cfg: dict,
        build_model_fn: Callable[[dict, tuple], Any],
        input_shape: tuple,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.trainer = trainer
        self.cfg = cfg
        self.build_model = build_model_fn
        self.input_shape = input_shape
        self.monitor = monitor
        self.mode = mode.lower()
        self.space = cfg["model"]["search"]

    # ---------- sampling ----------
    def _sample_float(self, spec, rng):
        lo, hi = spec["low"], spec["high"]
        if spec.get("log"):
            u = rng.random()
            return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        return lo + rng.random() * (hi - lo)

    def _sample_int(self, spec, rng):
        lo, hi = spec["low"], spec["high"]
        return rng.randint(lo, hi)

    def _sample_cat(self, spec, rng):
        return rng.choice(spec["choices"])

    def _sample_hparams(self, seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        # quick helper because random.Random has no choice on plain object
        rng.choice = lambda seq: seq[int(rng.random() * len(seq)) % len(seq)]
        hp = {}
        for k, spec in self.space.items():
            t = spec["type"]
            if t == "float":
                hp[k] = self._sample_float(spec, rng)
            elif t == "int":
                hp[k] = self._sample_int(spec, rng)
            elif t == "cat":
                hp[k] = self._sample_cat(spec, rng)
            else:
                raise ValueError(f"Unknown search type {t} for '{k}'")
        return hp

    # ---------- compare ----------
    def _is_better(self, a, b):
        if b is None: return True
        return (a < b) if self.mode == "min" else (a > b)

    # ---------- public: run a search on one fold ----------
    def run(self, data, fold: int, n_trials: int = 20):
        best_score, best_trial, best_hp, best_sel = None, None, None, None

        for t in range(n_trials):
            # reproducible per (fold, trial)
            hp = self._sample_hparams(seed=(fold + 1) * 10_000 + t)

            # let LR flow into trainer.compile via trainer.cfg (if you use it there)
            if "learning_rate" in hp:
                self.trainer.cfg["trainer"]["lr"] = float(hp["learning_rate"])

            # build and train (selection only)
            model = self.build_model(hp, self.input_shape)
            out = self.trainer.fit_eval_fold(
                model=model,
                data=data,
                fold=fold,
                trial=t,
                do_refit=False  # only selection during search
            )

            # pick score from selection val metrics
            sel_val = out["selection"]["val"]
            score = sel_val.get(self.monitor, None)
            if score is None:
                raise RuntimeError(f"Monitor '{self.monitor}' not found in selection metrics keys: {list(sel_val.keys())}")

            if self._is_better(score, best_score):
                best_score, best_trial, best_hp, best_sel = score, t, hp, out["selection"]

        # Refit once with the *winner*
        if "learning_rate" in best_hp:
            self.trainer.cfg["trainer"]["lr"] = float(best_hp["learning_rate"])
        best_model = self.build_model(best_hp, self.input_shape)

        refit_out = self.trainer.fit_eval_fold(
            model=best_model,
            data=data,
            fold=fold,
            trial=best_trial,
            do_refit=True   # this triggers train+val refit + final test logging
        )

        return {
            "best_trial": best_trial,
            "best_hparams": best_hp,
            "best_selection_score": best_score,
            "selection": best_sel,
            "refit": refit_out["refit"],  # contains train+val and test after refit
        }
