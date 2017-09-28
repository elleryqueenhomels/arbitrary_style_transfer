# Decoder mostly mirrors the encoder with all pooling layers replaced by nearest
# up-sampling to reduce checker-board effects.
# Decoder has no BN/IN layers.

import tensorflow as tf


WEIGHT_INIT_STDDEV = 0.1


def decode(image):

    with tf.variable_scope('decoder'):
        conv4_1 = conv2d(image, 512, 256, 3, 1, 'conv4_1')
        conv4_1 = upsample(conv4_1)

        conv3_4 = conv2d(conv4_1, 256, 256, 3, 1, 'conv3_4')
        conv3_3 = conv2d(conv3_4, 256, 256, 3, 1, 'conv3_3')
        conv3_2 = conv2d(conv3_3, 256, 256, 3, 1, 'conv3_2')
        conv3_1 = conv2d(conv3_2, 256, 128, 3, 1, 'conv3_1')
        conv3_1 = upsample(conv3_1)

        conv2_2 = conv2d(conv3_1, 128, 128, 3, 1, 'conv2_2')
        conv2_1 = conv2d(conv2_2, 128,  64, 3, 1, 'conv2_1')
        conv2_1 = upsample(conv2_1)

        conv1_2 = conv2d(conv2_1, 64, 64, 3, 1, 'conv1_2')
        conv1_1 = conv2d(conv1_2, 64,  3, 3, 1, 'conv1_1')

    return conv1_1


def conv2d(x, input_filters, output_filters, kernel_size, strides, scope='conv'):
    with tf.variable_scope(scope):
        # define variables
        shape  = [kernel_size, kernel_size, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='weight')
        bias   = tf.Variable(tf.zeros([output_filters]), name='bias')

        # padding image with reflection mode
        padding  = int(kernel_size // 2)
        x_padded = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode='REFLECT')

        # conv and add bias
        out = tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID')
        out = tf.nn.bias_add(out, bias)

        return tf.nn.relu(out)


def upsample(x, strides=2):
    height = tf.shape(x)[1] * strides
    width  = tf.shape(x)[2] * strides
    output = tf.image.resize_images(x, [height, width], 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output

