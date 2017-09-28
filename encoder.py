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
        self.weights = load_vgg_weights(weights_path)

    def encode(self, image):
        idx = 0
        layers = {}
        current = image

        with tf.variable_scope('encoder'):
            for name in ENCODER_LAYERS:
                kind = name[:4]

                if kind == 'conv':
                    kernel, bias = self.weights[idx]
                    idx += 1
                    current = conv2d(current, kernel, bias, name)
                elif kind == 'pool':
                    current = pool2d(current)

                layers[name] = current

            assert(len(layers) == len(ENCODER_LAYERS))

            enc = layers[ENCODER_LAYERS[-1]]

            return enc, layers

    def preprocess(self, image, mode='BGR'):
        if mode == 'BGR':
            return image - np.array([103.939, 116.779, 123.68])
        else:
            return image - np.array([123.68, 116.779, 103.939])


def conv2d(x, weight, bias, scope='conv'):
    with tf.variable_scope(scope):
        # define variables
        W = tf.Variable(weight, trainable=False, name='weight')
        b = tf.Variable(bias,   trainable=False, name='bias')

        # padding image with reflection mode
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        # conv and add bias
        out = tf.nn.conv2d(x_padded, W, strides=[1, 1, 1, 1], padding='VALID')
        out = tf.nn.bias_add(out, b)

        return tf.nn.relu(out)


def pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def load_vgg_weights(weights_path):
    kind = weights_path[-3:]
    if kind == 'npz':
        weights = load_from_npz(weights_path)
    elif kind == 'mat':
        weights = load_from_mat(weights_path)
    else:
        weights = None
        print('Unrecognized file type: %s' % kind)
    return weights


def load_from_npz(weights_path):
    params = np.load(weights_path)
    count = int(params['arr_0']) + 1
    weights = []
    for i in range(1, count, 2):
        kernel = params['arr_%s' % i]
        bias = params['arr_%s' % (i + 1)]
        weights.append((kernel, bias))
    return weights


def load_from_mat(weights_path):
    from scipy.io import loadmat
    data = loadmat(weights_path)
    if not all(i in data for i in ('layers', 'classes', 'normalization')):
        raise ValueError('You are using the wrong VGG-19 data.')
    params = data['layers'][0]

    weights = []
    for i, name in enumerate(VGG19_LAYERS):
        if name[:4] == 'conv':
            # matlabconv: [width, height, in_channels, out_channels]
            # tensorflow: [height, width, in_channels, out_channels]
            kernel, bias = params[i][0][0][0][0]
            kernel = np.transpose(kernel, [1, 0, 2, 3])
            bias = bias.reshape(-1) # flatten
            weights.append((kernel, bias))
    return weights

