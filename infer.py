# Use a trained Image Transform Net to generate
# a style transferred image with a specific style

import tensorflow as tf

from style_transfer_net import StyleTransferNet
from utils import get_images, save_images


def stylize(contents_path, styles_path, output_dir, encoder_path, model_path, 
            resize_height=None, resize_width=None, suffix=None):

    if isinstance(contents_path, str):
        contents_path = [contents_path]
    if isinstance(styles_path, str):
        styles_path = [styles_path]

    with tf.Graph().as_default(), tf.Session() as sess:
        # build the dataflow graph
        content = tf.placeholder(
            tf.float32, shape=(1, None, None, 3), name='content')
        style   = tf.placeholder(
            tf.float32, shape=(1, None, None, 3), name='style')

        stn = StyleTransferNet(encoder_path)

        output_image = stn.transform(content, style)

        sess.run(tf.global_variables_initializer())

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        outputs = []
        for content_path in contents_path:

            content_img = get_images(content_path, 
                height=resize_height, width=resize_width)

            for style_path in styles_path:

                style_img   = get_images(style_path)

                result = sess.run(output_image, 
                    feed_dict={content: content_img, style: style_img})

                outputs.append(result[0])

    save_images(outputs, contents_path, styles_path, output_dir, suffix=suffix)

    return outputs

