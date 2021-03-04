import tensorflow.keras.layers as L
import tensorflow as tf
import model.common as cm
import model.backbone as back


def SPP(input_data):
    pool5 = L.MaxPool2D(5, strides=1, padding='same')(input_data)
    pool9 = L.MaxPool2D(9, strides=1, padding='same')(input_data)
    pool13 = L.MaxPool2D(13, strides=1, padding='same')(input_data)

    conc = tf.concat([pool13, pool9, pool5, input_data], axis=-1)
    return conc

def YOLOv4_body(input_data):
    route1, route2, x = back.CSPDarkNet53(input_data)

    x = cm.ConvBlock(x, (1, 1, 1024, 512))
    x = cm.ConvBlock(x, (3, 3, 512, 1024))
    x = cm.ConvBlock(x, (1, 1, 1024, 512))

    x = SPP(x)

    x = cm.ConvBlock(x, (1, 1, 2048, 512))
    x = cm.ConvBlock(x, (3, 3, 512, 1024))
    x = cm.ConvBlock(x, (1, 1, 1024, 512))

    route3 = x

    x = cm.ConvBlock(route3, (1, 1, 512, 256))
    upsampled = L.UpSampling2D()(x)

    x = cm.ConvBlock(route2, (1, 1, 512, 256))
    x = tf.concat([x, upsampled], axis=-1)

    x = cm.ConvBlock(x, (1, 1, 512, 256))
    x = cm.ConvBlock(x, (3, 3, 256, 512))
    x = cm.ConvBlock(x, (1, 1, 512, 256))
    x = cm.ConvBlock(x, (3, 3, 256, 512))
    x = cm.ConvBlock(x, (1, 1, 512, 256))

    route2 = x

    x = cm.ConvBlock(route2, (1, 1, 256, 128))
    upsampled = L.UpSampling2D()(x)

    x = cm.ConvBlock(route1, (1, 1, 256, 128))
    x = tf.concat([x, upsampled], axis=-1)

    x = cm.ConvBlock(x, (1, 1, 256, 128))
    x = cm.ConvBlock(x, (3, 3, 128, 256))
    x = cm.ConvBlock(x, (1, 1, 256, 128))
    x = cm.ConvBlock(x, (3, 3, 128, 256))
    x = cm.ConvBlock(x, (1, 1, 256, 128))

    route1 = x

    return route1, route2, route3
