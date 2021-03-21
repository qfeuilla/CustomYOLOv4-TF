import tensorflow.keras as keras
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow as tf

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def partial_sigmoid(x):
    return tf.math.sigmoid(x) - (1 / 2)

class ConvBlock(keras.layers.Layer):
    def __init__(self, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky', **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.downsample = downsample
        if (downsample):
            padd = 'valid'
            strid = 2
            self.input_layer = L.ZeroPadding2D(((1, 0), (1, 0)))
        else:
            padd = 'same'
            strid = 1
        self.conv = L.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strid, padding=padd, 
                        use_bias=not bn, kernel_regularizer=keras.regularizers.l2(0.0005),
                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                        bias_initializer=tf.constant_initializer(0.))
        self.bn = bn
        self.batch_normalization = L.BatchNormalization()
        self.activate = activate
        if activate:
            if activate_type == "leaky":
                self.activation = L.LeakyReLU(alpha=0.1)
            elif activate_type == "mish":
                self.activation = L.Activation(mish)
            elif activate_type == "partial_sigmoid":
                self.activation = L.Activation(partial_sigmoid)
            else:
                self.activation = L.Activation(activate_type)
    
    def call(self, input_data):
        out = input_data
        if self.downsample:
            out = self.input_layer(out)
        out = self.conv(out)
        if self.bn:
            out = self.batch_normalization(out)
        if self.activate:
            out = self.activation(out)
        return out

class ResBlock(keras.layers.Layer):
    def __init__(self, input_channel, filter_num1, filter_num2, activate_type='leaky', **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.conv1 = ConvBlock(filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
        self.conv2 = ConvBlock(filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type)
    
    def call(self, input_data):
        res_out = self.conv1(input_data)
        res_out = self.conv2(res_out)
        res_out += input_data
        return res_out