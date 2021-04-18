import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from timer import Timer
t = Timer()

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
print("input shape is :", dataset.shape)

image_path = dataset.get_dir("test")
labels = dataset.get_label_data("test")

print(np.array(image_path).shape)
print(labels[0])

dataloader = tf.data.Dataset.from_tensor_slices((image_path, labels))
dataloader = dataloader.map(dataset.preprocess_X_y, num_parallel_calls = tf.data.experimental.AUTOTUNE)

'''
bbox = dataset.yolo_output_to_bbox((small, medium, large))

from loss import YOLO_loss

loss = YOLO_loss(dataset)

Xs, ys = dataset.get_batch()
y_hat = network(Xs)
bboxes = dataset.yolo_output_to_bbox(y_hat)
t.start()
print('loss :', loss(ys, y_hat))
t.stop()
'''