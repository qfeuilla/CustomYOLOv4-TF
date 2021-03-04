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

import model.backbone as back
from model.body import YOLOv4_body
import numpy as np
from model.head import YOLOv4_head
import gc

x = np.ones((8, 608, 608, 3), dtype=np.float64)
print(x.shape)

small, medium, large = YOLOv4_head(YOLOv4_body(x), 15)
gc.collect()
print(small.shape, medium.shape, large.shape)