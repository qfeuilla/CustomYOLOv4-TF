import tensorflow.keras.layers as L
import tensorflow as tf
import model.common as cm
import model.backbone as back
import tensorflow.keras as keras


class SPP(L.Layer):
    def __init__(self, name="SPP", **kwargs):
        super(SPP, self).__init__(self, name=name, **kwargs)
        self.pool1 = L.MaxPool2D(5, strides=1, padding='same')
        self.pool2 = L.MaxPool2D(9, strides=1, padding='same')
        self.pool3 = L.MaxPool2D(13, strides=1, padding='same')

    def call(self, input_data):
        pool5 = self.pool1(input_data)
        pool9 = self.pool2(input_data)
        pool13 = self.pool3(input_data)
        
        return tf.concat([pool13, pool9, pool5, input_data], axis=-1)

def YOLOv4_body(input_shape):
    inp = tf.keras.layers.Input(shape=input_shape)
    route1, route2, x = back.CSPDarkNet53()(inp)

    x = cm.ConvBlock((1, 1, 1024, 512))(x)
    x = cm.ConvBlock((3, 3, 512, 1024))(x)
    x = cm.ConvBlock((1, 1, 1024, 512))(x)

    x = SPP()(x)

    x = cm.ConvBlock((1, 1, 2048, 512))(x)
    x = cm.ConvBlock((3, 3, 512, 1024))(x)
    x = cm.ConvBlock((1, 1, 1024, 512), name="neck3")(x)

    route3 = x

    x = cm.ConvBlock((1, 1, 512, 256))(route3)
    upsampled = L.UpSampling2D()(x)

    x = cm.ConvBlock((1, 1, 512, 256))(route2)
    x = tf.concat([x, upsampled], axis=-1)

    x = cm.ConvBlock((1, 1, 512, 256))(x)
    x = cm.ConvBlock((3, 3, 256, 512))(x)
    x = cm.ConvBlock((1, 1, 512, 256))(x)
    x = cm.ConvBlock((3, 3, 256, 512))(x)
    x = cm.ConvBlock((1, 1, 512, 256), name="neck2")(x)

    route2 = x

    x = cm.ConvBlock((1, 1, 256, 128))(route2)
    upsampled = L.UpSampling2D()(x)

    x = cm.ConvBlock((1, 1, 256, 128))(route1)
    x = tf.concat([x, upsampled], axis=-1)

    x = cm.ConvBlock((1, 1, 256, 128))(x)
    x = cm.ConvBlock((3, 3, 128, 256))(x)
    x = cm.ConvBlock((1, 1, 256, 128))(x)
    x = cm.ConvBlock((3, 3, 128, 256))(x)
    x = cm.ConvBlock((1, 1, 256, 128), name="neck1")(x)

    route1 = x

    return keras.Model(inputs=[inp], outputs=(route1, route2, route3), name="Body")
