# Use a trained Image Transform Net to generate
# a style transferred image with a specific style

import tensorflow as tf

from style_transfer_net import StyleTransferNet
from utils import get_images, save_images


def generate(contents_path, style_path, encoder_path, model_path, 
    is_same_size=False, resize_height=None, resize_width=None, 
    output_path=None, prefix='stylized-', suffix=None):

    if isinstance(contents_path, str):
        contents_path = [contents_path]
    if isinstance(style_path, list):
        style_path = style_path[0]

    if is_same_size or (resize_height is not None and resize_width is not None):
        outputs = _handler1(contents_path, style_path, encoder_path, model_path, 
            resize_height=resize_height, resize_width=resize_width, 
            output_path=output_path, prefix=prefix, suffix=suffix)
        return list(outputs)
    else:
        outputs = _handler2(contents_path, style_path, encoder_path, model_path, 
            output_path=output_path, prefix=prefix, suffix=suffix)
        return outputs


def _handler1(content_path, style_path, encoder_path, model_path, 
    resize_height=None, resize_width=None, output_path=None, 
    prefix=None, suffix=None):

    # get the actual image data, output shape:
    # (num_images, height, width, color_channels)
    content_img = get_images(content_path, resize_height, resize_width)
    style_img   = get_images(style_path)

    with tf.Graph().as_default(), tf.Session() as sess:
        # build the dataflow graph
        content = tf.placeholder(
            tf.float32, shape=content_img.shape, name='content')
        style   = tf.placeholder(
            tf.float32, shape=style_img.shape, name='style')

        stn = StyleTransferNet(encoder_path)

        output_image = stn.transform(content, style)

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        output = sess.run(output_image, 
            feed_dict={content: content_img, style: style_img})

    if output_path is not None:
        save_images(content_path, output, output_path, 
            prefix=prefix, suffix=suffix)

    return output


def _handler2(content_path, style_path, encoder_path, model_path, 
    output_path=None, prefix=None, suffix=None):

    style_img = get_images(style_path)

    with tf.Graph().as_default(), tf.Session() as sess:
        # build the dataflow graph
        content = tf.placeholder(
            tf.float32, shape=(1, None, None, 3), name='content')
        style   = tf.placeholder(
            tf.float32, shape=style_img.shape, name='style')

        stn = StyleTransferNet(encoder_path)

        output_image = stn.transform(content, style)

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        output = []
        for path in content_path:
            content_img = get_images(path)
            result = sess.run(output_image, 
                feed_dict={content: content_img, style: style_img})
            output.append(result[0])

    if output_path is not None:
        save_images(content_path, output, output_path, 
            prefix=prefix, suffix=suffix)

    return output

