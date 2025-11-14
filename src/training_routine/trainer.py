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
from .metrics import undershooting_pct as _undershooting_pct
from .metrics import QLikeLoss
from .training_utils import early_stopping_step


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

        lr = float(self.cfg.trainer.hparams["lr"])
        weight_decay = float(self.cfg.trainer.hparams["weight_decay"])

        # loss mapping similar to Keras strings
        loss_name = str(self.cfg.trainer.hparams["loss"]).lower()
        if loss_name in ("mse", "mean_squared_error"):
            self.loss_fn = nn.MSELoss()
        elif loss_name in ("mae", "mean_absolute_error", "l1"):
            self.loss_fn = nn.L1Loss()
        elif loss_name in ("qlike"):
            self.loss_fn = QLikeLoss()
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
            name = self.cfg.model.name.lower()
            if name in ("lstm", "gru", "rnn"):        
                # safest: skip compile for recurrent nets
                pass
                # or, if you want some speed but safe graph breaks:
                # self.model = torch.compile(self.model, backend="eager", fullgraph=False)
            elif name == "simplecnn":
                self.model = torch.compile(self.model, backend="eager", fullgraph=False)
            else:
                self.model = torch.compile(self.model)
        except Exception:
            pass

    def _batch_iter(self, X: torch.Tensor, y: torch.Tensor, batch_size: int):
        n = X.shape[0]
        for i in range(0, n, batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

    @torch.inference_mode()
    def _evaluate_fold(self, X: np.ndarray, y: np.ndarray, eval_bs: int = 8192) -> Dict[str, float]:
        """
        Memory-safe evaluation: iterate in batches and stream the metrics.
        """
        self.model.eval()

        n = 0
        loss_sum = 0.0
        mae_sum  = 0.0
        mse_sum  = 0.0
        diracc_correct = 0
        undershooting = 0

        # You can safely use bf16 autocast for *inference* to trim memory
        name = self.cfg.model.name.lower()
        use_amp_eval = (name not in ("lstm", "gru", "rnn"))  # if you want to keep LSTM eval in fp32, set False
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp_eval)

        self.console_logger.debug(f'X shape: {X.shape}')
        self.console_logger.debug(f'y shape: {y.shape}')

        for i in range(0, len(X), eval_bs):
            xb = torch.as_tensor(X[i:i+eval_bs], dtype=torch.float32, device=self.device)
            yb = torch.as_tensor(y[i:i+eval_bs], dtype=torch.float32, device=self.device)
            if yb.ndim == 2 and yb.size(-1) == 1:
                yb = yb.squeeze(-1)

            with amp_ctx:
                pb = self.model(xb).squeeze(-1)
                lb = self.loss_fn(pb, yb)

            b = yb.numel()
            loss_sum += float(lb) * b
            mae_sum  += torch.nn.functional.l1_loss(pb, yb, reduction="sum").item()
            mse_sum  += torch.nn.functional.mse_loss(pb, yb, reduction="sum").item()
            diracc_correct += (torch.sign(pb) == torch.sign(yb)).sum().item()
            undershooting += _undershooting_pct(pb, yb)

            n += b

        return {
            "loss": loss_sum / n,
            "mae":  mae_sum  / n,
            "mse":  mse_sum  / n,
            "directional_accuracy_pct": 100.0 * diracc_correct / n,
            "undershooting_pct": 100.0 * undershooting / n,
        }

    def _y_to_tensor(self, y):
        t = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        if t.ndim == 2 and t.size(-1) == 1:
            # single output, squeeze to (N,)
            t = t.squeeze(-1)
        elif t.ndim == 2 and t.size(-1) > 1:
            # multi-output target, keep as (N, M)
            pass
        elif t.ndim != 1:
            raise ValueError(f"y must be 1-D, (N,1), or (N,M); got {tuple(t.shape)}")

        return t

    def fit_eval_fold(
        self,
        model: torch.nn.Module,
        data: tuple,
        fold: int,
        trial: int = 0,
        merge_train_val: bool = False,  # Should we allow this? If no search we're losing info but unfair comparison with other model
        report_cb: Optional[Callable] = None  
    ):
        self.console_logger.debug(f"report_cb: {report_cb}")

        # Unpack numpy arrays
        Xtr, ytr, Xv, yv, Xte, yte = data

        # Paths
        fold_dir = self.logger.path(f"fold_{fold:03d}/")

        # “Early stopping” via best-on-val checkpointing
        monitor_key = self.cfg.experiment.monitor  # e.g. 'val_loss', 'val_mae'
        mode = str(self.cfg.experiment.mode).lower()  # 'min' or 'max'
        if not merge_train_val:
            # In search mode we require a validation metric for checkpointing/pruning
            if not str(monitor_key).startswith("val_"):
                msg = f"monitor should be a validation metric (e.g., 'val_loss') when merge_train_val=False, got {monitor_key} instead"
                self.console_logger.warning(msg)
        val_every = int(self.cfg.trainer.hparams["val_every"])

        # Prepare tensors
        if merge_train_val:
            Xtrv = np.concatenate([Xtr, Xv], axis=0)
            ytrv = np.concatenate([ytr, yv], axis=0)

            # Same name so the notation is less cumbersome
            Xtr_tensor = torch.as_tensor(Xtrv, dtype=torch.float32, device=self.device) 
            ytr_tensor = self._y_to_tensor(ytrv)
            # No validation tensors in merged mode
            Xv_tensor = yv_tensor = None
            Xte_tensor  = torch.as_tensor(Xte,  dtype=torch.float32, device=self.device)
            yte_tensor  = self._y_to_tensor(yte)
        else:
            Xtr_tensor = torch.as_tensor(Xtr, dtype=torch.float32, device=self.device)
            ytr_tensor = self._y_to_tensor(ytr)
            Xv_tensor  = torch.as_tensor(Xv,  dtype=torch.float32, device=self.device)
            yv_tensor  = self._y_to_tensor(yv)
            Xte_tensor  = torch.as_tensor(Xte,  dtype=torch.float32, device=self.device)
            yte_tensor  = self._y_to_tensor(yte)

        # Compile (optimizer/loss/device)
        self.compile(model)

        # define sizes
        epochs = int(self.cfg.trainer.hparams["epochs"])
        batch_size = int(self.cfg.trainer.hparams["batch_size"])

        # assume Xtr_tensor: (N, D), ytr_tensor: (N,)
        N = Xtr_tensor.size(0)
        g = torch.Generator(device="cpu").manual_seed(42)  # fixed seed for reproducibility

        # generate a random permutation of indices
        perm = torch.randperm(N, generator=g)

        # shuffle both X and y with the same permutation
        Xtr_shuffled = Xtr_tensor[perm]
        ytr_shuffled = ytr_tensor[perm]

        # Pre-split batches once (cuts per-iter slicing cost)
        xb_chunks = torch.split(Xtr_shuffled, batch_size, dim=0)
        yb_chunks = torch.split(ytr_shuffled, batch_size, dim=0)

        # AMP is optional, keep it off for minimalism; enable if you want:
        use_amp = False # Sometime is better off
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

        # define variables for grad descent
        best_val = np.inf if mode == "min" else -np.inf
        history = []  # store monitor per epoch for compatibility
        grad_history = []  # store grads 

        # early stopping
        patience  = self.cfg.trainer.hparams.get("torch_patience")
        patience = int(patience) if patience is not None else patience
        min_delta = float(self.cfg.trainer.hparams.get("min_delta", 1e-4))
        stalled = 0
        best_state = None

        t0 = time.time()
        if Xv_tensor is not None:
            msg = f"Fitting fold {fold:02d} (trial {trial:02d}) " \
                f"on {Xtr_tensor.shape} samples, val {Xv_tensor.shape}, test {Xte_tensor.shape}." \
                f" Y's shapes: train {ytr_tensor.shape}, val {yv_tensor.shape}, test {yte_tensor.shape}" \
            
        else:
            msg = f"Fitting fold {fold:02d} (trial {trial:02d}) " \
                f"on train+val {Xtr_tensor.shape} samples, test {Xte_tensor.shape}" \
                f"Y's shapes: train {ytr_tensor.shape}, test {yte_tensor.shape}"

        self.console_logger.info(msg)

        # Info on datasets
        percentiles = [0, 5, 25, 50, 75, 95, 100]
        self.console_logger.debug(f"np.var(ytr),np.var(yv),np.var(yte): {np.var(ytr):.8f},{np.var(yv):.8f},{np.var(yte):.8f}")
        self.console_logger.debug(f"Y train, val, test {percentiles}:\n{np.percentile(ytr, percentiles)}\n{np.percentile(yv, percentiles)}\n{np.percentile(yte, percentiles)}")
        self.console_logger.debug(f"np.var(Xtr),np.var(Xv),np.var(Xte): {np.var(Xtr):.8f},{np.var(Xv):.8f},{np.var(Xte):.8f}")
        self.console_logger.debug(f"X train, val, test {percentiles}:\n{np.percentile(Xtr, percentiles)}\n{np.percentile(Xv, percentiles)}\n{np.percentile(Xte, percentiles)}")
        
        # debug optimizer and model
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        opt_params   = sum(p.numel() for g in self.optimizer.param_groups for p in g["params"] if p.requires_grad)
        self.console_logger.debug(f"param_count model={model_params} optimizer={opt_params}")

        effective_epochs = 0
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
            # TODO: remove the ifs and make this more streamlined.        
            if not merge_train_val:
                if epoch % val_every == 0 or epoch == epochs or epoch == 1:
                    self.model.eval()
                    with torch.inference_mode():
                        eval_out = self._evaluate_fold(Xv_tensor, yv_tensor)
                        # average train loss once per epoch (one CPU sync)
                    vloss = float(eval_out.get("loss", np.nan))

                    history.append({"tr_loss":tr_loss_avg, "val_loss":vloss})
                    self.console_logger.info(
                        f"Epoch {epoch:03d} | loss={epoch_loss/max(seen,1):.12f} "
                        f"| val_loss={vloss:.12f} | time: {time.time()-start_epoch_time:.3f}s"
                    )
                    
                    if patience is not None:
                        # early stopping logic (on validation checkpoints only)
                        # be very mindful with this, setting a high patience makes the
                        # GPU mmeory implode
                        es_result = early_stopping_step(epoch=epoch,
                                                        val_loss=vloss,
                                                        best_val=best_val,
                                                        stalled=stalled,
                                                        patience=patience,
                                                        min_delta=min_delta,
                                                        mode=mode,
                                                        model=self.model,
                                                        optimizer=self.optimizer)
                        
                        best_val, stalled, best_state, should_stop = es_result
                        
                        if should_stop:
                            self.console_logger.info(
                                f"Early stopping at epoch {epoch} "
                                f"(no val improvement in {patience} validations)."
                            )
                            # break out of training loop
                            break


                    # optional pruning callback only in search mode
                    if callable(report_cb):
                        self.console_logger.debug(f"in report_cb")

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
                    total_norm = 0
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            grad_means[name] = p.grad.abs().mean().item()
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item()**2
                    total_norm = total_norm ** 0.5
                    self.console_logger.debug(f'Grad L2 norm: {total_norm}')
                    grad_history.append({"epoch": epoch, **grad_means})

            effective_epochs += 1

        # get best and write it to disk
        if merge_train_val:
            # No validation-based selection: take the LAST epoch as "best"
            best_epoch = epochs
            best_val = history[-1]["tr_loss"]
        else:
            val_entries = [h for h in history if "val_loss" in h]
            val_history = np.array([h["val_loss"] for h in val_entries])
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
            vmap = {f"val_{k}": np.nan for k, v in tr.items()}  # placeholder, use tr since the metrics are identical
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
