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

x = 320
y = 320
inp = np.ones((4, x, y, 3), dtype=np.float64)
network = YOLOv4(side=x)
network.compile(loss="mse", optimizer="sgd", metrics="acc")

small, medium, large = network(inp)
print(GPUInfo.get_info())
print((x, y), " work and give :", small.shape, medium.shape, large.shape)
network.summary()
print(network.layers[-1].output_shape)

from dataset import Dataset
dataset = Dataset(4, network.layers[-1].output_shape, shape=network.layers[0].input_shape[1: 3])
bbox = dataset.yolo_output_to_bbox((small, medium, large))

from loss import YOLO_loss

loss = YOLO_loss(dataset)

X, y = dataset.get_X_y(type_data="test")
y_hat = network(np.expand_dims(X, 0))
dataset.preprocess_y(y)