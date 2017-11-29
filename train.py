# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import tensorflow as tf

from style_transfer_net import StyleTransferNet
from utils import get_train_images


STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

TRAINING_IMAGE_SHAPE = (256, 256, 3) # (height, width, color_channels)

EPOCHS = 16
EPSILON = 1e-5
BATCH_SIZE = 8
LEARNING_RATE = 1e-4


def train(style_weight, content_imgs_path, style_imgs_path, encoder_path, save_path, debug=False, logging_period=100):
    if debug:
        from datetime import datetime
        start_time = datetime.now()

    # guarantee the size of content and style images to be a multiple of BATCH_SIZE
    num_imgs = min(len(content_imgs_path), len(style_imgs_path))
    content_imgs_path = content_imgs_path[:num_imgs]
    style_imgs_path   = style_imgs_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        content_imgs_path = content_imgs_path[:-mod]
        style_imgs_path   = style_imgs_path[:-mod]

    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
        style   = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')

        # create the style transfer net
        stn = StyleTransferNet(encoder_path)

        # pass content and style to the stn, getting the generated_img
        generated_img = stn.transform(content, style)

        # get the target feature maps which is the output of AdaIN
        target_features = stn.target_features

        # pass the generated_img to the encoder, and use the output compute loss
        generated_img = tf.reverse(generated_img, axis=[-1])  # switch RGB to BGR
        generated_img = stn.encoder.preprocess(generated_img) # preprocess image
        enc_gen, enc_gen_layers = stn.encoder.encode(generated_img)

        # compute the content loss
        content_loss = tf.reduce_sum(tf.reduce_mean(tf.square(enc_gen - target_features), axis=[1, 2]))

        # compute the style loss
        style_layer_loss = []
        for layer in STYLE_LAYERS:
            enc_style_feat = stn.encoded_style_layers[layer]
            enc_gen_feat   = enc_gen_layers[layer]

            meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
            meanG, varG = tf.nn.moments(enc_gen_feat,   [1, 2])

            sigmaS = tf.sqrt(varS + EPSILON)
            sigmaG = tf.sqrt(varG + EPSILON)

            l2_mean  = tf.reduce_sum(tf.square(meanG - meanS))
            l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

            style_layer_loss.append(l2_mean + l2_sigma)

        style_loss = tf.reduce_sum(style_layer_loss)

        # compute the total loss
        loss = content_loss + style_weight * style_loss

        # Training step
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        """Start Training"""
        step = 0
        n_batches = int(len(content_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')
            start_time = datetime.now()

        try:
            for epoch in range(EPOCHS):

                np.random.shuffle(content_imgs_path)
                np.random.shuffle(style_imgs_path)

                for batch in range(n_batches):
                    # retrive a batch of content and style images
                    content_batch_path = content_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                    style_batch_path   = style_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]

                    content_batch = get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
                    style_batch   = get_train_images(style_batch_path,   crop_height=HEIGHT, crop_width=WIDTH)

                    # run the training step
                    sess.run(train_op, feed_dict={content: content_batch, style: style_batch})

                    step += 1

                    if step % 1000 == 0:
                        saver.save(sess, save_path, global_step=step)

                    if debug:
                        is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                        if is_last_step or step % logging_period == 0:
                            elapsed_time = datetime.now() - start_time
                            _content_loss, _style_loss, _loss = sess.run([content_loss, style_loss, loss], 
                                feed_dict={content: content_batch, style: style_batch})

                            print('step: %d,  total loss: %.3f,  elapsed time: %s' % (step, _loss, elapsed_time))
                            print('content loss: %.3f' % (_content_loss))
                            print('style loss  : %.3f,  weighted style loss: %.3f\n' % (_style_loss, style_weight * _style_loss))
        except:
            tmp_save_path = save_path + '-' + str(step)
            saver.save(sess, tmp_save_path)
            print('\nSomething wrong happens! Current model is saved to <%s>\n' % tmp_save_path)

        """Done Training & Save the model"""
        saver.save(sess, save_path)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % save_path)

