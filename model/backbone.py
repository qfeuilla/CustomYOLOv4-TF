import model.common as cm
import tensorflow as tf

def CSPDarkNet53(input_data):
    input_data = cm.ConvBlock(input_data, (3, 3, 3, 32), activate_type="mish")
    input_data = cm.ConvBlock(input_data, (3, 3, 32, 64), downsample=True,  activate_type="mish")

    route = cm.ConvBlock(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 64, 64), activate_type="mish")
    for _ in range(1):
        input_data = cm.ResBlock(input_data, 64, 32, 64, activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cm.ConvBlock(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = cm.ConvBlock(input_data, (3, 3, 64, 128), downsample=True,  activate_type="mish")

    route = cm.ConvBlock(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 128, 64), activate_type="mish")
    for _ in range(2):
        input_data = cm.ResBlock(input_data, 64, 64, 64, activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cm.ConvBlock(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = cm.ConvBlock(input_data, (3, 3, 128, 256), downsample=True,  activate_type="mish")

    route = cm.ConvBlock(input_data, (1, 1, 256, 128), activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 256, 128), activate_type="mish")
    for _ in range(8):
        input_data = cm.ResBlock(input_data, 128, 128, 128, activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cm.ConvBlock(input_data, (1, 1, 256, 256), activate_type="mish")
    route1 = input_data
    input_data = cm.ConvBlock(input_data, (3, 3, 256, 512), downsample=True,  activate_type="mish")

    route = cm.ConvBlock(input_data, (1, 1, 512, 256), activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 512, 256), activate_type="mish")
    for _ in range(8):
        input_data = cm.ResBlock(input_data, 256, 256, 256, activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cm.ConvBlock(input_data, (1, 1, 512, 512), activate_type="mish")
    route2 = input_data
    input_data = cm.ConvBlock(input_data, (3, 3, 512, 1024), downsample=True,  activate_type="mish")

    route = cm.ConvBlock(input_data, (1, 1, 1024, 512), activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 1024, 512), activate_type="mish")
    for _ in range(4):
        input_data = cm.ResBlock(input_data, 512, 512, 512, activate_type="mish")
    input_data = cm.ConvBlock(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cm.ConvBlock(input_data, (1, 1, 1024, 1024), activate_type="mish")
    return route1, route2, input_data