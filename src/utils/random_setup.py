import random
import os
import numpy as np
import torch

def set_global_seed(seed: int):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure 
    reproducibility for a specific run.
    """
    # 1. Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # 3. Deterministic Operations (Critical for Logic)
    # Forces deterministic algorithms (throws error if an op is non-deterministic)
    torch.use_deterministic_algorithms(True) 
    
    # 4. CuDNN Benchmarking (Critical for Performance/Stability)
    # benchmark=False: Prevents selecting different algorithms based on hardware benchmarking
    # deterministic=True: Selects only deterministic algorithms
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Required for torch.use_deterministic_algorithms when using CUDA 10.2+
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"