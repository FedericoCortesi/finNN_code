# src/training/metrics.py
import tensorflow as tf
from tensorflow import keras

@keras.utils.register_keras_serializable()
def directional_accuracy_pct(y_true, y_pred):
    y_true_sign = tf.sign(y_true)
    y_pred_sign = tf.sign(y_pred)
    matches = tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)
    return 100.0 * tf.reduce_mean(matches)
