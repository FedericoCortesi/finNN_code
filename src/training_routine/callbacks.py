import time
from tensorflow import keras

from utils.custom_formatter import setup_logger

logger = setup_logger("Trainer")

class VerboseLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_total = time.time()
    def on_epoch_begin(self, epoch, logs=None):
        self.start_epoch = time.time()
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_epoch
        #now = time.strftime("%H:%M:%S", time.localtime())
        loss = logs.get("loss", float("nan"))
        vloss = logs.get("val_loss", float("nan"))
        da = logs.get("directional_accuracy_pct")
        vda = logs.get("val_directional_accuracy_pct")

        msg = (f"Epoch {epoch+1:03d} | loss={loss:.12f}"
               f" | val_loss={vloss:.12f}")
        if da is not None and vda is not None:
            msg += f" | diracc={da:.2f}% | val_diracc={vda:.2f}%"
        msg += f" | {elapsed:.2f}s"

        logger.info(msg)
