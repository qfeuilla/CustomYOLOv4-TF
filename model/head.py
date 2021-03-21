import tensorflow.keras.layers as L
import tensorflow as tf
import model.common as cm

def YOLOv4_head(input_data, NUM_CLASS, bbox_per_head=2):
    shape1, shape2, shape3 = input_data
    route1 = tf.keras.layers.Input(shape=shape1[1:], name="head1")
    route2 = tf.keras.layers.Input(shape=shape2[1:], name="head2")
    route3 = tf.keras.layers.Input(shape=shape3[1:], name="head3")

    conv_small_box = cm.ConvBlock((3, 3, 128, 256))(route1)
    confidence = cm.ConvBlock((3, 3, 256, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)(conv_small_box)
    box_params = cm.ConvBlock((3, 3, 256, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)(conv_small_box)
    classes = []
    for i in range(bbox_per_head):
        classes.append(cm.ConvBlock((3, 3, 256, NUM_CLASS), activate=True, activate_type="softmax", bn=False)(conv_small_box))
    conv_small_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

    x = cm.ConvBlock((3, 3, 128, 256), downsample=True)(route1)
    x = tf.concat([x, route2], axis=-1)

    x = cm.ConvBlock((1, 1, 512, 256))(x)
    x = cm.ConvBlock((3, 3, 256, 512))(x)
    x = cm.ConvBlock((1, 1, 512, 256))(x)
    x = cm.ConvBlock((3, 3, 256, 512))(x)
    x = cm.ConvBlock((1, 1, 512, 256))(x)

    skip = x

    conv_medium_box = cm.ConvBlock((3, 3, 256, 512))(x)
    confidence = cm.ConvBlock((3, 3, 512, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)(conv_medium_box)
    box_params = cm.ConvBlock((3, 3, 512, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)(conv_medium_box)
    classes = []
    for i in range(bbox_per_head):
        classes.append(cm.ConvBlock((3, 3, 512, NUM_CLASS), activate=True, activate_type="softmax", bn=False)(conv_medium_box))
    conv_medium_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

    x = cm.ConvBlock((3, 3, 256, 512), downsample=True)(skip)
    x = tf.concat([x, route3], axis=-1)

    x = cm.ConvBlock((1, 1, 1024, 512))(x)
    x = cm.ConvBlock((3, 3, 512, 1024))(x)
    x = cm.ConvBlock((1, 1, 1024, 512))(x)
    x = cm.ConvBlock((3, 3, 512, 1024))(x)
    x = cm.ConvBlock((1, 1, 1024, 512))(x)

    conv_big_box = cm.ConvBlock((3, 3, 512, 1024))(x)
    confidence = cm.ConvBlock((3, 3, 1024, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)(conv_big_box)
    box_params = cm.ConvBlock((3, 3, 1024, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)(conv_big_box)
    classes = []
    for i in range(bbox_per_head):
        classes.append(cm.ConvBlock((3, 3, 1024, NUM_CLASS), activate=True, activate_type="softmax", bn=False)(conv_big_box))
    conv_big_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

    return tf.keras.Model(inputs=[route1, route2, route3], outputs=(conv_small_box, conv_medium_box, conv_big_box), name="Heads")
