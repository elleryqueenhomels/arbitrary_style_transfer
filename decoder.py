# Decoder mostly mirrors the encoder with all pooling layers replaced by nearest
# up-sampling to reduce checker-board effects.
# Decoder has no BN/IN layers.

import tensorflow as tf


WEIGHT_INIT_STDDEV = 0.1


class Decoder(object):

    def __init__(self):
        self.weight_vars = []

        with tf.variable_scope('decoder'):
            self.weight_vars.append(self._create_variables(512, 256, 3, scope='conv4_1'))

            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_4'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_3'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_2'))
            self.weight_vars.append(self._create_variables(256, 128, 3, scope='conv3_1'))

            self.weight_vars.append(self._create_variables(128, 128, 3, scope='conv2_2'))
            self.weight_vars.append(self._create_variables(128,  64, 3, scope='conv2_1'))

            self.weight_vars.append(self._create_variables( 64,  64, 3, scope='conv1_2'))
            self.weight_vars.append(self._create_variables( 64,   3, 3, scope='conv1_1'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape  = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias   = tf.Variable(tf.zeros([output_filters]), name='bias')
            return (kernel, bias)

    def decode(self, image):
        # upsampling after 'conv4_1', 'conv3_1', 'conv2_1'
        upsample_indices = (0, 4, 6)
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, use_relu=False)
            else:
                out = conv2d(out, kernel, bias)
            
            if i in upsample_indices:
                out = upsample(out)

        return out


def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    if use_relu:
        out = tf.nn.relu(out)

    return out


def upsample(x, scale=2):
    height = tf.shape(x)[1] * scale
    width  = tf.shape(x)[2] * scale
    output = tf.image.resize_images(x, [height, width], 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output

