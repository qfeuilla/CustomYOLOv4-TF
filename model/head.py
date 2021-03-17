import tensorflow.keras.layers as L
import tensorflow as tf
import model.common as cm


def YOLOv4_head(input_data, NUM_CLASS, bbox_per_head=2):
    route1, route2, route3 = input_data

    conv_small_box = cm.ConvBlock(route1, (3, 3, 128, 256))
    confidence = cm.ConvBlock(conv_small_box, (3, 3, 256, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)
    box_params = cm.ConvBlock(conv_small_box, (3, 3, 256, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)
    classes = []
    for i in range(bbox_per_head):
        classes.append(cm.ConvBlock(conv_small_box, (3, 3, 256, NUM_CLASS), activate=True, activate_type="softmax", bn=False))
    conv_small_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

    x = cm.ConvBlock(route1, (3, 3, 128, 256), downsample=True)
    x = tf.concat([x, route2], axis=-1)

    x = cm.ConvBlock(x, (1, 1, 512, 256))
    x = cm.ConvBlock(x, (3, 3, 256, 512))
    x = cm.ConvBlock(x, (1, 1, 512, 256))
    x = cm.ConvBlock(x, (3, 3, 256, 512))
    x = cm.ConvBlock(x, (1, 1, 512, 256))

    route2 = x

    conv_medium_box = cm.ConvBlock(x, (3, 3, 256, 512))
    confidence = cm.ConvBlock(conv_medium_box, (3, 3, 512, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)
    box_params = cm.ConvBlock(conv_medium_box, (3, 3, 512, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)
    classes = []
    for i in range(bbox_per_head):
        classes.append(cm.ConvBlock(conv_medium_box, (3, 3, 512, NUM_CLASS), activate=True, activate_type="softmax", bn=False))
    conv_medium_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

    x = cm.ConvBlock(route2, (3, 3, 256, 512), downsample=True)
    x = tf.concat([x, route3], axis=-1)

    x = cm.ConvBlock(x, (1, 1, 1024, 512))
    x = cm.ConvBlock(x, (3, 3, 512, 1024))
    x = cm.ConvBlock(x, (1, 1, 1024, 512))
    x = cm.ConvBlock(x, (3, 3, 512, 1024))
    x = cm.ConvBlock(x, (1, 1, 1024, 512))

    conv_big_box = cm.ConvBlock(x, (3, 3, 512, 1024))
    confidence = cm.ConvBlock(conv_big_box, (3, 3, 1024, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)
    box_params = cm.ConvBlock(conv_big_box, (3, 3, 1024, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)
    classes = []
    for i in range(bbox_per_head):
        classes.append(cm.ConvBlock(conv_big_box, (3, 3, 1024, NUM_CLASS), activate=True, activate_type="softmax", bn=False))
    conv_big_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

    return conv_small_box, conv_medium_box, conv_big_box
