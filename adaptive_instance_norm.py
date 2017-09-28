# Adaptive Instance Normalization

import tensorflow as tf


def AdaIN(content, style, epsilon=1e-5):
    meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
    meanS, varS = tf.nn.moments(style,   [1, 2], keep_dims=True)

    sigmaC = tf.sqrt(tf.add(varC, epsilon))
    sigmaS = tf.sqrt(tf.add(varS, epsilon))
    
    return (content - meanC) * sigmaS / sigmaC + meanS

