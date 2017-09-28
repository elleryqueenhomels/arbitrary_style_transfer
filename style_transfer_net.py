# Style Transfer Network
# Encoder -> AdaIN -> Decoder

import tensorflow as tf

from decoder import decode
from encoder import Encoder
from adaptive_instance_norm import AdaIN


class StyleTransferNet(object):

    def __init__(self, encoder_weights_path):
        self.encoder_weights_path = encoder_weights_path

    def transform(self, content, style):
        # create encoder
        encoder = Encoder(self.encoder_weights_path)
        self.encoder = encoder

        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])
        style   = tf.reverse(style,   axis=[-1])

        # preprocess image
        content = encoder.preprocess(content)
        style   = encoder.preprocess(style)

        # encode image
        enc_c, enc_c_layers = encoder.encode(content)
        enc_s, enc_s_layers = encoder.encode(style)

        self.encoded_content_layers = enc_c_layers
        self.encoded_style_layers   = enc_s_layers

        # pass the encoded images to AdaIN
        target_features = AdaIN(enc_c, enc_s)
        self.target_features = target_features

        # decode the target features back to image
        generated_img = decode(target_features)

        # deprocess image
        generated_img = encoder.deprocess(generated_img)

        # switch BGR back to RGB
        output = tf.reverse(generated_img, axis=[-1])

        return output

