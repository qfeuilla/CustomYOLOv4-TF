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

class YOLOv4_body(L.Layer):
    def __init__(self, name="YOLOv4_body", **kwargs):
        super(YOLOv4_body, self).__init__(self, name=name, **kwargs)

        self.darknet = back.CSPDarkNet53()

        self.conv1_1 = cm.ConvBlock((1, 1, 1024, 512))
        self.conv1_2 = cm.ConvBlock((3, 3, 512, 1024))
        self.conv1_3 = cm.ConvBlock((1, 1, 1024, 512))
    
        self.spp = SPP()

        self.conv2_1 = cm.ConvBlock((1, 1, 2048, 512))
        self.conv2_2 = cm.ConvBlock((3, 3, 512, 1024))
        self.conv2_3 = cm.ConvBlock((1, 1, 1024, 512))

        self.conv3 = cm.ConvBlock((1, 1, 512, 256))
        self.upsample3 = L.UpSampling2D()

        self.conv4 = cm.ConvBlock((1, 1, 512, 256))

        self.conv5_1 = cm.ConvBlock((1, 1, 512, 256))
        self.conv5_2 = cm.ConvBlock((3, 3, 256, 512))
        self.conv5_3 = cm.ConvBlock((1, 1, 512, 256))
        self.conv5_4 = cm.ConvBlock((3, 3, 256, 512))
        self.conv5_5 = cm.ConvBlock((1, 1, 512, 256))

        self.conv6 = cm.ConvBlock((1, 1, 256, 128))
        self.upsample6 = L.UpSampling2D()

        self.conv7 = cm.ConvBlock((1, 1, 256, 128))

        self.conv8_1 = cm.ConvBlock((1, 1, 256, 128))
        self.conv8_2 = cm.ConvBlock((3, 3, 128, 256))
        self.conv8_3 = cm.ConvBlock((1, 1, 256, 128))
        self.conv8_4 = cm.ConvBlock((3, 3, 128, 256))
        self.conv8_5 = cm.ConvBlock((1, 1, 256, 128))
    
    def call(self, input_data):
        route1, route2, x = self.darknet(input_data)

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        x = self.spp(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        route3 = x
        
        x = self.conv3(route3)
        upsampled = self.upsample3(x)

        x = self.conv4(route2)
        x = tf.concat([x, upsampled], axis=-1)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)

        route2 = x

        x = self.conv6(route2)
        upsampled = self.upsample6(x)

        x = self.conv7(route1)
        x = tf.concat([x, upsampled], axis=-1)

        x = self.conv8_1(x)
        x = self.conv8_2(x)
        x = self.conv8_3(x)
        x = self.conv8_4(x)
        x = self.conv8_5(x)

        route1 = x

        return route1, route2, route3

'''
def YOLOv4_body(input_data):
    route1, route2, x = back.CSPDarkNet53()(input_data)

    x = cm.ConvBlock((1, 1, 1024, 512))(x)
    x = cm.ConvBlock((3, 3, 512, 1024))(x)
    x = cm.ConvBlock((1, 1, 1024, 512))(x)

    x = SPP()(x)

    x = cm.ConvBlock((1, 1, 2048, 512))(x)
    x = cm.ConvBlock((3, 3, 512, 1024))(x)
    x = cm.ConvBlock((1, 1, 1024, 512))(x)

    route3 = x

    x = cm.ConvBlock((1, 1, 512, 256))(route3)
    upsampled = L.UpSampling2D()(x)

    x = cm.ConvBlock((1, 1, 512, 256))(route2)
    x = tf.concat([x, upsampled], axis=-1)

    x = cm.ConvBlock((1, 1, 512, 256))(x)
    x = cm.ConvBlock((3, 3, 256, 512))(x)
    x = cm.ConvBlock((1, 1, 512, 256))(x)
    x = cm.ConvBlock((3, 3, 256, 512))(x)
    x = cm.ConvBlock((1, 1, 512, 256))(x)

    route2 = x

    x = cm.ConvBlock((1, 1, 256, 128))(route2)
    upsampled = L.UpSampling2D()(x)

    x = cm.ConvBlock((1, 1, 256, 128))(route1)
    x = tf.concat([x, upsampled], axis=-1)

    x = cm.ConvBlock((1, 1, 256, 128))(x)
    x = cm.ConvBlock((3, 3, 128, 256))(x)
    x = cm.ConvBlock((1, 1, 256, 128))(x)
    x = cm.ConvBlock((3, 3, 128, 256))(x)
    x = cm.ConvBlock((1, 1, 256, 128))(x)

    route1 = x

    return route1, route2, route3
'''