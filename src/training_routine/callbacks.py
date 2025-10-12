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
        now = time.strftime("%H:%M:%S", time.localtime())
        loss = logs.get("loss", float("nan"))
        vloss = logs.get("val_loss", float("nan"))
        logger.info(f"Epoch {epoch+1:03d} | loss={loss:.12f} | val_loss={vloss:.12f} | {elapsed:.2f}s")
