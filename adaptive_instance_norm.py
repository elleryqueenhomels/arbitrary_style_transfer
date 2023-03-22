# Adaptive Instance Normalization

import tensorflow as tf

def AdaIN(content, style, epsilon=1e-5):
    mean_c, var_c = tf.nn.moments(content, [1, 2], keep_dims=True)
    mean_s, var_s = tf.nn.moments(style,   [1, 2], keep_dims=True)

    sigma_c = tf.sqrt(tf.add(var_c, epsilon))
    sigma_s = tf.sqrt(tf.add(var_s, epsilon))
    
    return (content - mean_c) * sigma_s / sigma_c + mean_s
