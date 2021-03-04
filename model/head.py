import tensorflow.keras.layers as L
import tensorflow as tf
import model.common as cm


def YOLOv4_head(input_data, NUM_CLASS):
    route1, route2, route3 = input_data

    conv_small_box = cm.ConvBlock(route1, (3, 3, 128, 256))
    conv_small_box = cm.ConvBlock(conv_small_box, (3, 3, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    x = cm.ConvBlock(route1, (3, 3, 128, 256), downsample=True)
    x = tf.concat([x, route2], axis=-1)

    x = cm.ConvBlock(x, (1, 1, 512, 256))
    x = cm.ConvBlock(x, (3, 3, 256, 512))
    x = cm.ConvBlock(x, (1, 1, 512, 256))
    x = cm.ConvBlock(x, (3, 3, 256, 512))
    x = cm.ConvBlock(x, (1, 1, 512, 256))

    route2 = x

    conv_medium_box = cm.ConvBlock(x, (3, 3, 256, 512))
    conv_medium_box = cm.ConvBlock(conv_medium_box, (3, 3, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    x = cm.ConvBlock(route2, (3, 3, 256, 512), downsample=True)
    x = tf.concat([x, route3], axis=-1)

    x = cm.ConvBlock(x, (1, 1, 1024, 512))
    x = cm.ConvBlock(x, (3, 3, 512, 1024))
    x = cm.ConvBlock(x, (1, 1, 1024, 512))
    x = cm.ConvBlock(x, (3, 3, 512, 1024))
    x = cm.ConvBlock(x, (1, 1, 1024, 512))

    conv_big_box = cm.ConvBlock(x, (3, 3, 512, 1024))
    conv_big_box = cm.ConvBlock(conv_big_box, (3, 3, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return conv_small_box, conv_medium_box, conv_big_box
