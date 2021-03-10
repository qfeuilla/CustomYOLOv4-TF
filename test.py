import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

from model.YOLOv4 import YOLOv4
import numpy as np
import gc
from tqdm import tqdm
from gpuinfo import GPUInfo

x = 608
y = 608
inp = np.ones((4, x, y, 3), dtype=np.float64)
small, medium, large = YOLOv4(inp, 20)
print((x, y), " work and give :", small.shape, medium.shape, large.shape)
