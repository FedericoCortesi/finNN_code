from .custom_formatter import setup_logger


info_logger = setup_logger("Trainer")

import ctypes
import sys
import tensorflow as tf
from utils.custom_formatter import setup_logger


def gpu_test():
    logger = setup_logger("GPU-Test", level="DEBUG")

    logger.debug("=" * 20 + " Running GPU test " + "=" * 20)
    logger.debug(f"TensorFlow version: {tf.__version__}")

    visible_gpus = tf.config.list_physical_devices('GPU')
    logger.debug(f"Visible GPUs: {visible_gpus if visible_gpus else '❌ None detected'}")

    built_with_cuda = tf.test.is_built_with_cuda()
    logger.debug(f"Built with CUDA?: {built_with_cuda}")

    build_info = tf.sysconfig.get_build_info()
    logger.debug(f"Build info:\n{build_info}")

    try:
        ctypes.CDLL('libcuda.so.1')
        logger.debug("libcuda.so.1 is loadable ✅")
    except OSError as e:
        logger.error(f"libcuda.so.1 NOT found ❌ ({e})")
        sys.exit(1)

    logger.debug("GPU test completed successfully.")


if __name__ == "__main__":
    gpu_test()

    