from model import head
from model import body
import tensorflow.keras as keras

class YOLOv4(keras.Model):
    def __init__(self, n_classes=10, **kwargs):
        super(YOLOv4, self).__init__(self, **kwargs)
        self.body = body.YOLOv4_body()
        self.head = head.YOLOv4_head(n_classes)
    
    def call(self, input_data):
        out = self.body(input_data)
        out = self.head(out)
        return out
