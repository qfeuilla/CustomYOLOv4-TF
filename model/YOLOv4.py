from model import head
from model import body
import tensorflow.keras as keras

class YOLOv4(keras.Model):
    def __init__(self, n_classes=10, name="YOLOv4", side=320,  **kwargs):
        super(YOLOv4, self).__init__(name=name, **kwargs)
        self.body = body.YOLOv4_body((side, side, 3))
        self.head = head.YOLOv4_head(self.body.output_shape, n_classes)
        self.side = side
    
    def call(self, input_data):
        out = self.body(input_data)
        out = self.head(out)
        return out

    def model(self):
        x = keras.layers.Input(shape=(self.side, self.side, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))