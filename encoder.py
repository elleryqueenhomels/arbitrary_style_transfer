# Encoder is fixed to the first few layers (up to relu4_1)
# of VGG-19 (pre-trained on ImageNet)
# This code is a modified version of Anish Athalye's vgg.py
# https://github.com/anishathalye/neural-style/blob/master/vgg.py

import numpy as np
import tensorflow as tf


ENCODER_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1'
)


class Encoder(object):

    def __init__(self, weights_path):
        # load weights (kernel and bias) from npz file
        weights = np.load(weights_path)

        idx = 0
        self.weight_vars = []

        # create the TensorFlow variables
        with tf.variable_scope('encoder'):
            for layer in ENCODER_LAYERS:
                kind = layer[:4]

                if kind == 'conv':
                    kernel = weights['arr_%d' % idx].transpose([2, 3, 1, 0])
                    bias   = weights['arr_%d' % (idx + 1)]
                    kernel = kernel.astype(np.float32)
                    bias   = bias.astype(np.float32)
                    idx += 2

                    with tf.variable_scope(layer):
                        W = tf.Variable(kernel, trainable=False, name='kernel')
                        b = tf.Variable(bias,   trainable=False, name='bias')

                    self.weight_vars.append((W, b))

    def encode(self, image):
        # create the computational graph
        idx = 0
        layers = {}
        current = image

        for layer in ENCODER_LAYERS:
            kind = layer[:4]

            if kind == 'conv':
                kernel, bias = self.weight_vars[idx]
                idx += 1
                current = conv2d(current, kernel, bias)

            elif kind == 'relu':
                current = tf.nn.relu(current)

            elif kind == 'pool':
                current = pool2d(current)

            layers[layer] = current

        assert(len(layers) == len(ENCODER_LAYERS))

        enc = layers[ENCODER_LAYERS[-1]]

        return enc, layers

    def preprocess(self, image, mode='BGR'):
        if mode == 'BGR':
            return image - np.array([103.939, 116.779, 123.68])
        else:
            return image - np.array([123.68, 116.779, 103.939])

    def deprocess(self, image, mode='BGR'):
        if mode == 'BGR':
            return image + np.array([103.939, 116.779, 123.68])
        else:
            return image + np.array([123.68, 116.779, 103.939])


def conv2d(x, kernel, bias):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    return out


def pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

