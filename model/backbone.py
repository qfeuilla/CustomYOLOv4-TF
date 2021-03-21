import model.common as cm
import tensorflow as tf
import tensorflow.keras as keras

class CSPDarkNet53(keras.Model):
    def __init__(self, name="CSPDarkNet53", **kwargs):
        super(CSPDarkNet53, self).__init__(self, name=name, **kwargs)

        self.conv1_1 = cm.ConvBlock((3, 3, 3, 32), activate_type="mish")
        self.conv1_2 = cm.ConvBlock((3, 3, 32, 64), downsample=True, activate_type="mish")
        
        self.conv2_1 = cm.ConvBlock((1, 1, 64, 64), activate_type="mish")
        self.conv2_2 = cm.ConvBlock((1, 1, 64, 64), activate_type="mish")
        self.res_blocks1 = []
        for _ in range(1):
            self.res_blocks1.append(cm.ResBlock(64, 32, 64, activate_type="mish"))
        self.conv2_3 = cm.ConvBlock((1, 1, 64, 64), activate_type="mish")

        self.conv3_1 = cm.ConvBlock((1, 1, 128, 64), activate_type="mish")
        self.conv3_2 = cm.ConvBlock((3, 3, 64, 128), downsample=True, activate_type="mish")

        self.conv4_1 = cm.ConvBlock((1, 1, 128, 64), activate_type="mish")
        self.conv4_2 = cm.ConvBlock((1, 1, 128, 64), activate_type="mish")
        self.res_blocks2 = []
        for _ in range(2):
            self.res_blocks2.append(cm.ResBlock(64, 64, 64, activate_type="mish"))
        self.conv4_3 = cm.ConvBlock((1, 1, 64, 64), activate_type="mish")
        
        self.conv5_1 = cm.ConvBlock((1, 1, 128, 128), activate_type="mish")
        self.conv5_2 =  cm.ConvBlock((3, 3, 128, 256), downsample=True, activate_type="mish")

        self.conv6_1 = cm.ConvBlock((1, 1, 256, 128), activate_type="mish")
        self.conv6_2 = cm.ConvBlock((1, 1, 256, 128), activate_type="mish")
        self.res_blocks3 = []
        for _ in range(8):
            self.res_blocks3.append(cm.ResBlock(128, 128, 128, activate_type="mish"))
        self.conv6_3 = cm.ConvBlock((1, 1, 128, 128), activate_type="mish")
        
        self.conv7_1 = cm.ConvBlock((1, 1, 256, 256), activate_type="mish")
        self.conv7_2 = cm.ConvBlock((3, 3, 256, 512), downsample=True, activate_type="mish")

        self.conv8_1 = cm.ConvBlock((1, 1, 512, 256), activate_type="mish")
        self.conv8_2 = cm.ConvBlock((1, 1, 512, 256), activate_type="mish")
        self.res_blocks4 = []
        for _ in range(8):
            self.res_blocks4.append(cm.ResBlock(256, 256, 256, activate_type="mish"))
        self.conv8_3 = cm.ConvBlock((1, 1, 256, 256), activate_type="mish")
        
        self.conv9_1 = cm.ConvBlock((1, 1, 512, 512), activate_type="mish")
        self.conv9_2 = cm.ConvBlock((3, 3, 512, 1024), downsample=True, activate_type="mish")

        self.conv10_1 = cm.ConvBlock((1, 1, 1024, 512), activate_type="mish")
        self.conv10_2 = cm.ConvBlock((1, 1, 1024, 512), activate_type="mish")
        self.res_blocks5 = []
        for _ in range(4):
            self.res_blocks5.append(cm.ResBlock(512, 512, 512, activate_type="mish"))
        self.conv10_3 = cm.ConvBlock((1, 1, 512, 512), activate_type="mish")
        
        self.conv_last = cm.ConvBlock((1, 1, 1024, 1024), activate_type="mish")

    def call(self, input_data):
        output = self.conv1_1(input_data)
        output = self.conv1_2(output)

        route = self.conv2_1(output)
        output = self.conv2_2(output)
        for res in self.res_blocks1:
            output = res(output)
        output = self.conv2_3(output)
        output = tf.concat([output, route], axis=-1)

        output = self.conv3_1(output)
        output = self.conv3_2(output)

        route = self.conv4_1(output)
        output = self.conv4_2(output)
        for res in self.res_blocks2:
            output = res(output)
        output = self.conv4_3(output)
        output = tf.concat([output, route], axis=-1)

        output = self.conv5_1(output)
        output = self.conv5_2(output)

        route = self.conv6_1(output)
        output = self.conv6_2(output)
        for res in self.res_blocks3:
            output = res(output)
        output = self.conv6_3(output)
        output = tf.concat([output, route], axis=-1)

        output = self.conv7_1(output)
        route1 = output
        output = self.conv7_2(output)

        route = self.conv8_1(output)
        output = self.conv8_2(output)
        for res in self.res_blocks4:
            output = res(output)
        output = self.conv8_3(output)
        output = tf.concat([output, route], axis=-1)

        output = self.conv9_1(output)
        route2 = output
        output = self.conv9_2(output)

        route = self.conv10_1(output)
        output = self.conv10_2(output)
        for res in self.res_blocks5:
            output = res(output)
        output = self.conv10_3(output)
        output = tf.concat([output, route], axis=-1)

        output = self.conv_last(output)
        return route1, route2, output