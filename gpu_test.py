import tensorflow as tf
print("TF:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA?:", tf.test.is_built_with_cuda())
print(tf.sysconfig.get_build_info())

import ctypes, sys
try:
    ctypes.CDLL('libcuda.so.1')
    print("libcuda.so.1 is loadable ✅")
except OSError as e:
    print("libcuda.so.1 NOT found ❌", e)
    sys.exit(1)