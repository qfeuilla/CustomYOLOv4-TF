import tensorflow.keras.layers as L
import tensorflow as tf
import model.common as cm


class YOLOv4_head(L.Layer):
    def __init__(self, NUM_CLASS, bbox_per_head=2, name="YOLOv4_head", **kwargs):
        super(YOLOv4_head, self).__init__(self, name=name, **kwargs)
        
        self.conv_sm = cm.ConvBlock((3, 3, 128, 256))
        self.confidence_sm = cm.ConvBlock((3, 3, 256, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)
        self.box_params_sm = cm.ConvBlock((3, 3, 256, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)
        self.classes_sm = []
        for _ in range(bbox_per_head):
            self.classes_sm.append(cm.ConvBlock((3, 3, 256, NUM_CLASS), activate=True, activate_type="softmax", bn=False))
        
        self.conv1 = cm.ConvBlock((3, 3, 128, 256), downsample=True)
        
        self.conv2_1 = cm.ConvBlock((1, 1, 512, 256))
        self.conv2_2 = cm.ConvBlock((3, 3, 256, 512))
        self.conv2_3 = cm.ConvBlock((1, 1, 512, 256))
        self.conv2_4 = cm.ConvBlock((3, 3, 256, 512))
        self.conv2_5 = cm.ConvBlock((1, 1, 512, 256))

        self.conv_md = cm.ConvBlock((3, 3, 256, 512))
        self.confidence_md = cm.ConvBlock((3, 3, 512, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)
        self.box_params_md = cm.ConvBlock((3, 3, 512, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)
        self.classes_md = []
        for _ in range(bbox_per_head):
            self.classes_md.append(cm.ConvBlock((3, 3, 512, NUM_CLASS), activate=True, activate_type="softmax", bn=False))
        
        self.conv3 = cm.ConvBlock((3, 3, 256, 512), downsample=True)
        
        self.conv4_1 = cm.ConvBlock((1, 1, 1024, 512))
        self.conv4_2 = cm.ConvBlock((3, 3, 512, 1024))
        self.conv4_3 = cm.ConvBlock((1, 1, 1024, 512))
        self.conv4_4 = cm.ConvBlock((3, 3, 512, 1024))
        self.conv4_5 = cm.ConvBlock((1, 1, 1024, 512))

        self.conv_lg = cm.ConvBlock((3, 3, 512, 1024))
        self.confidence_lg = cm.ConvBlock((3, 3, 1024, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)
        self.box_params_lg = cm.ConvBlock((3, 3, 1024, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)
        self.classes_lg = []
        for _ in range(bbox_per_head):
            self.classes_lg.append(cm.ConvBlock((3, 3, 1024, NUM_CLASS), activate=True, activate_type="softmax", bn=False))

    def call(self, input_data):
        route1, route2, route3 = input_data

        small_box = self.conv_sm(route1)
        confidence = self.confidence_sm(small_box)
        box_params = self.box_params_sm(small_box)
        classes = []
        for classif in self.classes_sm:
            classes.append(classif(small_box))
        small_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

        x = self.conv1(route1)
        x = tf.concat([x, route2], axis=-1)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        x = self.conv2_5(x)

        route2 = x

        medium_box = self.conv_md(x)
        confidence = self.confidence_md(medium_box)
        box_params = self.box_params_md(medium_box)
        classes = []
        for classif in self.classes_md:
            classes.append(classif(medium_box))
        medium_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

        x = self.conv3(route2)
        x = tf.concat([x, route3], axis=-1)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)

        big_box = self.conv_lg(x)
        confidence = self.confidence_lg(big_box)
        box_params = self.box_params_lg(big_box)
        classes = []
        for classif in self.classes_lg:
            classes.append(classif(big_box))
        big_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

        return small_box, medium_box, big_box
        
'''

def YOLOv4_head(input_data, NUM_CLASS, bbox_per_head=2):
    route1, route2, route3 = input_data

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

    route2 = x

    conv_medium_box = cm.ConvBlock((3, 3, 256, 512))(x)
    confidence = cm.ConvBlock((3, 3, 512, bbox_per_head), activate=True, activate_type="sigmoid", bn=False)(conv_medium_box)
    box_params = cm.ConvBlock((3, 3, 512, bbox_per_head * 4), activate=True, activate_type="partial_sigmoid", bn=False)(conv_medium_box)
    classes = []
    for i in range(bbox_per_head):
        classes.append(cm.ConvBlock((3, 3, 512, NUM_CLASS), activate=True, activate_type="softmax", bn=False)(conv_medium_box))
    conv_medium_box = tf.concat([confidence, box_params, tf.concat(classes, axis=-1)], axis=-1)

    x = cm.ConvBlock((3, 3, 256, 512), downsample=True)(route2)
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

    return conv_small_box, conv_medium_box, conv_big_box
'''