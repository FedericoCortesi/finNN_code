import torch
import ctypes
import sys
from utils.custom_formatter import setup_logger


def gpu_test():
    logger = setup_logger("GPU-Test", level="DEBUG")

    logger.debug("=" * 20 + " Running GPU test " + "=" * 20)
    logger.debug(f"PyTorch version: {torch.__version__}")
    logger.debug(f"CUDA runtime version: {torch.version.cuda}")

    # Visible GPU devices
    visible_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    logger.debug(f"Visible GPUs: {visible_gpus if visible_gpus else 'None detected'}")

    # CUDA build info
    cuda_built = torch.cuda.is_available()
    logger.debug(f"Built with CUDA?: {cuda_built}")

    if cuda_built:
        driver_version = torch.version.cuda
        logger.debug(f"CUDA driver/runtime version: {driver_version}")
        current_device = torch.cuda.current_device()
        logger.debug(f"Active device ID: {current_device}")
        logger.debug(f"Active device name: {torch.cuda.get_device_name(current_device)}")
    else:
        logger.error("CUDA not available (torch.cuda.is_available() == False)")

    # Low-level library test
    try:
        ctypes.CDLL('libcuda.so.1')
        logger.debug("libcuda.so.1 is loadable")
    except OSError as e:
        logger.error(f"libcuda.so.1 NOT found ({e})")
        sys.exit(1)

    logger.debug("GPU test completed successfully")


if __name__ == "__main__":
    gpu_test()
