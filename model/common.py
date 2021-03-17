import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow as tf

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def ConvBlock(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = L.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padd = 'valid'
        strid = 2
    else:
        padd = 'same'
        strid = 1
    conv = L.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strid, padding=padd, 
                        use_bias=not bn, kernel_regularizer=keras.regularizers.l2(0.0005),
                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                        bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = L.BatchNormalization()(conv) # Problem of batchnorm fixed in new version of tf so no need to do a custom Batchnorm
    if activate:
        if activate_type == "leaky":
            conv = L.LeakyReLU(alpha=0.1)(conv)
        elif activate_type == "mish":
            conv = L.Activation(mish)(conv)
        elif activate_type == "softmax":
            conv = L.Activation("softmax")(conv)
        elif activate_type == "sigmoid":
            conv = L.Activation("sigmoid")(conv)
        elif activate_type == "partial_sigmoid":
            conv = L.Activation("sigmoid")(conv) - (1 / 2)
    return conv

def ResBlock(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    conv = ConvBlock(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = ConvBlock(conv, filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    res_out = conv + input_layer
    return res_out