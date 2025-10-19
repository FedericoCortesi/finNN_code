import os, time
import numpy as np
from typing import Tuple, Dict, Any, Callable, Optional

import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.optim as optim
import optuna 
#import warnings
#warnings.simplefilter(action="ignore", module=torch)

from utils.logging_utils import ExperimentLogger
from utils.custom_formatter import setup_logger
from config.config_types import AppConfig

from .metrics import directional_accuracy_pct as _directional_accuracy_pct
from .metrics import mse as _mse
from .metrics import mae as _mae


class Trainer:
    def __init__(self, 
                 cfg: AppConfig, 
                 logger: ExperimentLogger):
        # attirbutes
        self.cfg = cfg
        self.logger = logger
        
        # console logging, mainly fro debugging
        self.console_logger = setup_logger("Trainer", level="INFO")
        
        self.device = torch.device("cuda")
        # Fail if cuda (=GPU) not available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This trainer requires a GPU.")
        
        # will be set in compile()
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    # Build optimizer/loss 
    def compile(self, model: torch.nn.Module):
        self.model = model.to(self.device)

        self.console_logger.debug(f"self.cfg.trainer:\n{self.cfg.trainer}")
        lr = float(self.cfg.trainer.hparams["lr"])
        weight_decay = float(self.cfg.trainer.hparams["weight_decay"])

        # loss mapping similar to Keras strings
        loss_name = str(self.cfg.trainer.hparams["loss"]).lower()
        if loss_name in ("mse", "mean_squared_error"):
            self.loss_fn = nn.MSELoss()
        elif loss_name in ("mae", "mean_absolute_error", "l1"):
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss '{loss_name}'")

        # Separate parameters into two groups:
        #  - decay: weights of Linear/Conv layers (apply L2 regularization)
        #  - no_decay: biases and normalization params (skip weight decay)
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                # 1D params
                if p.dim() == 1 or n.endswith("bias"):
                    no_decay.append(p)
                else:
                    decay.append(p)

        # Apply weight decay only to 'decay' group for cleaner regularization
        self.optimizer = optim.Adam(
            [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}],
            lr=lr,
            fused=True
        )
        try:
            self.model = torch.compile(self.model)
        except Exception:
            pass

    def _batch_iter(self, X: torch.Tensor, y: torch.Tensor, batch_size: int):
        n = X.shape[0]
        for i in range(0, n, batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

    @torch.no_grad()
    def _evaluate_fold(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Fuction to evaluate the performance at the end of a fold
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        pred = self.model(X_t).squeeze(-1)
        loss_val = self.loss_fn(pred, y_t).item()

        return {
            "loss": loss_val,
            "mae": _mae(pred, y_t),
            "mse": _mse(pred, y_t),
            "directional_accuracy_pct": _directional_accuracy_pct(pred, y_t),
        }

    def fit_eval_fold(
        self,
        model: torch.nn.Module,
        data: tuple,
        fold: int,
        trial: int = 0,
        merge_train_val: bool = False,  # Should we allow this? If no search we're losing info but unfair comparison with other model
        report_cb: Optional[Callable] = None  
    ):
        # Unpack numpy arrays
        Xtr, ytr, Xv, yv, Xte, yte = data

        # Paths
        fold_dir = self.logger.path(f"fold_{fold:03d}/")

        # “Early stopping” via best-on-val checkpointing
        monitor_key = self.cfg.experiment.monitor  # e.g. 'val_loss', 'val_mae'
        mode = str(self.cfg.experiment.mode).lower()  # 'min' or 'max'
        if not merge_train_val:
            # In search mode we require a validation metric for checkpointing/pruning
            assert str(monitor_key).startswith("val_"), \
                "monitor should be a validation metric (e.g., 'val_loss') when merge_train_val=False"
        val_every = int(self.cfg.trainer.hparams["val_every"])

        # Prepare tensors
        if merge_train_val:
            Xtrv = np.concatenate([Xtr, Xv], axis=0)
            ytrv = np.concatenate([ytr, yv], axis=0)

            # Same name so the notation is less cumbersome
            Xtr_tensor = torch.as_tensor(Xtrv, dtype=torch.float32, device=self.device) 
            ytr_tensor = torch.as_tensor(ytrv, dtype=torch.float32, device=self.device)
            # No validation tensors in merged mode
            Xv_tensor = yv_tensor = None
            Xte_tensor  = torch.as_tensor(Xte,  dtype=torch.float32, device=self.device)
            yte_tensor  = torch.as_tensor(yte,  dtype=torch.float32, device=self.device)
        else:
            Xtr_tensor = torch.as_tensor(Xtr, dtype=torch.float32, device=self.device)
            ytr_tensor = torch.as_tensor(ytr, dtype=torch.float32, device=self.device)
            Xv_tensor  = torch.as_tensor(Xv,  dtype=torch.float32, device=self.device)
            yv_tensor  = torch.as_tensor(yv,  dtype=torch.float32, device=self.device)
            Xte_tensor  = torch.as_tensor(Xte,  dtype=torch.float32, device=self.device)
            yte_tensor  = torch.as_tensor(yte,  dtype=torch.float32, device=self.device)

        # Compile (optimizer/loss/device)
        self.compile(model)

        # define sizes
        epochs = int(self.cfg.trainer.hparams["epochs"])
        batch_size = int(self.cfg.trainer.hparams["batch_size"])

        # Pre-split batches once (cuts per-iter slicing cost)
        xb_chunks = torch.split(Xtr_tensor, batch_size, dim=0)
        yb_chunks = torch.split(ytr_tensor, batch_size, dim=0)

        # AMP is optional, keep it off for minimalism; enable if you want:
        use_amp = True
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

        best_val = np.inf if mode == "min" else -np.inf
        history = []  # store monitor per epoch for compatibility
        grad_history = []  # store grads 

        t0 = time.time()
        if Xv_tensor is not None:
            msg = f"Fitting fold {fold:02d} (trial {trial:02d}) " \
                f"on {Xtr_tensor.shape} samples, val {Xv_tensor.shape}, test {Xte_tensor.shape}"
        else:
            msg = f"Fitting fold {fold:02d} (trial {trial:02d}) " \
                f"on train+val {Xtr_tensor.shape} samples, test {Xte_tensor.shape}"

        self.console_logger.info(msg)


        # iterate over epochs
        for epoch in range(1, epochs + 1):
            start_epoch_time = time.time()
            self.model.train()
            epoch_loss = 0.0
            seen = 0

            # manual batching
            for xb, yb in zip(xb_chunks, yb_chunks):
                # set gradient to zero otherwise they accumulate
                self.optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    pred = self.model(xb).squeeze(-1)
                    loss = self.loss_fn(pred, yb)
                loss.backward()
                self.optimizer.step()
                bs = xb.shape[0]
                epoch_loss += loss.detach() * bs
                seen += bs

            # compute loss
            tr_loss_avg = (epoch_loss / max(seen, 1)).item()

            # Validate only every k epochs (default k=10)
            # k=1 slower but good for debugging 
            # TODO: take out if not merge_train_val, less
            # computation on cpu is better. This is 
            # slow as hell.        
            if not merge_train_val:
                if epoch % val_every == 0 or epoch == epochs or epoch == 1:
                    self.model.eval()
                    with torch.no_grad(), amp_ctx:
                        vpred = self.model(Xv_tensor).squeeze(-1)
                        vloss = self.loss_fn(vpred, yv_tensor).item()
                        # average train loss once per epoch (one CPU sync)

                    history.append({"tr_loss":tr_loss_avg, "val_loss":vloss})
                    self.console_logger.info(
                        f"Epoch {epoch:03d} | loss={epoch_loss/max(seen,1):.12f} "
                        f"| val_loss={vloss:.6f} | time: {time.time()-start_epoch_time:.3f}s"
                    )
                    # optional pruning callback only in search mode
                    if callable(report_cb):
                        # report the monitored validation metric
                        if monitor_key == "val_loss":
                            should_prune = report_cb(epoch, vloss)
                        else:
                            # If you track additional val metrics in history, pass them here.
                            should_prune = report_cb(epoch, vloss)
                        if should_prune:
                            raise optuna.TrialPruned()                    
            else:
                # merged mode: no val, just log train loss
                history.append({"tr_loss": tr_loss_avg})
                self.console_logger.info(
                    f"Epoch {epoch:03d} | loss={tr_loss_avg:.12f} "
                    f"| time: {time.time()-start_epoch_time:.3f}s"
                )

            # debug grads
            if epoch % 1 == 0:  # only every 10 epochs to avoid overhead
                with torch.no_grad():
                    grad_means = {}
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            grad_means[name] = p.grad.abs().mean().item()
                    grad_history.append({"epoch": epoch, **grad_means})


        # get best and write it to disk
        if merge_train_val:
            # No validation-based selection: take the LAST epoch as "best"
            best_epoch = epochs
            best_val = history[-1]["tr_loss"]
        else:
            val_history = [history[e].get("val_loss") for e in range(epochs)]
            best_epoch = int(np.argmin(np.array(val_history))) + 1 if mode == "min" else int(np.argmax(np.array(val_history))) + 1
            best_val = history[best_epoch - 1].get("val_loss")

        torch.save(
            {
                "epoch": best_epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "monitor": best_val,
                "history":history,
                "grad_history":grad_history
            },
            os.path.join(fold_dir, "model_best.pt"),
        )

        secs = time.time() - t0
        self.console_logger.info("Evaluating...")

        # Load best before final eval (optional but consistent with “restore_best_weights”)
        state = torch.load(os.path.join(fold_dir, "model_best.pt"), map_location=self.device)
        self.model.load_state_dict(state["model_state"])

        # Evaluate full splits
        if merge_train_val:
            tr = self._evaluate_fold(Xtr_tensor, ytr_tensor) # in this case is already merged w/ validation
        else:
            tr = self._evaluate_fold(Xtr_tensor, ytr_tensor)
            v = self._evaluate_fold(Xv_tensor,  yv_tensor)
        te = self._evaluate_fold(Xte_tensor, yte_tensor)

        # Prefix keys
        trmap = {f"tr_{k}": v for k, v in tr.items()}
        temap = {f"test_{k}": v for k, v in te.items()}
        if not merge_train_val: 
            vmap  = {f"val_{k}": v for k, v in v.items()}
            # Derive best_epoch from tracked history
            if mode == "min":
                best_epoch_from_hist = int(np.argmin(np.array(val_history))) + 1
            else:
                best_epoch_from_hist = int(np.argmax(np.array(val_history))) + 1
        else:
            vmap = {"val_loss": np.nan, 
                    "val_mae": np.nan, 
                    "val_mse": np.nan, 
                    "val_directional_accuracy_pct": np.nan}  # placeholder
            best_epoch_from_hist = -1



        print()
        self.console_logger.info(
            f"tr_loss={trmap.get('tr_loss', np.nan):.6f} | "
            f"val_loss={vmap.get('val_loss', np.nan):.6f} | "
            f"test_loss={temap.get('test_loss', np.nan):.6f} | "
            #f"tr_mae={trmap.get('tr_mae', np.nan):.6f} | "
            #f"val_mae={vmap.get('val_mae', np.nan):.6f} | "
            #f"test_mae={temap.get('test_mae', np.nan):.6f} | "
            f"tr_diracc={trmap.get('tr_directional_accuracy_pct', np.nan):.2f}% | "
            f"val_diracc={vmap.get('val_directional_accuracy_pct', np.nan):.2f}% | "
            f"test_diracc={temap.get('test_directional_accuracy_pct', np.nan):.2f}% | "
            f"best_epoch={best_epoch_from_hist}"
        )
        print()

        model_path = f"{fold_dir}/model_best.pt"
        self.console_logger.info(f"Saving model at {model_path}")
        self.logger.append_result(
            trial=trial,
            fold=fold,
            seconds=secs,
            model_path=self.logger.path(model_path),
            **trmap, **vmap, **temap
        )

        return trmap, vmap, temap
